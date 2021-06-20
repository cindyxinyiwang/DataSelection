import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import gc
import random
import numpy as np
from utils import *
from model import *
from rl_data_utils import RLDataUtil
from featurizer import *
from actor import *
from customAdam import *
from customSGD import *

from collections import defaultdict, deque
import copy
import time


def get_val_ppl(model, data_loader, hparams, crit, step, load_full, log):
    decode = model.hparams.decode
    model.hparams.decode = True
    model.eval()
    valid_words = 0
    valid_loss = 0
    valid_acc = 0
    n_batches = 0
    total_ppl, total_bleu = 0, 0
    valid_bleu = None
    logits_batch, labels_batch = None, None
    for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, dev_file_index, x_rank in data_loader.next_dev(
            dev_batch_size=hparams.valid_batch_size, load_full=load_full,
            log=log):
        # clear GPU memory
        #gc.collect()
        # next batch
        # do this since you shift y_valid[:, 1:] and y_valid[:, :-1]
        y_count -= batch_size
        # word count
        valid_words += y_count

        logits = model.forward(
            x,
            x_mask,
            x_len,
            x_pos_emb_idxs,
            y[:, :-1],
            y_mask[:, :-1],
            y_len,
            y_pos_emb_idxs,
            x_char,
            y_char,
            file_idx=dev_file_index,
            step=step,
            x_rank=x_rank)
        if logits_batch is None:
            logits_batch = logits
            labels_batch = y[:, 1:].contiguous()
        logits = logits.view(-1, hparams.trg_vocab_size)
        labels = y[:, 1:].contiguous().view(-1)
        val_loss, val_acc = get_performance(crit, logits, labels, hparams)
        n_batches += batch_size
        valid_loss += val_loss.item()
        valid_acc += val_acc.item()
        if eof:
            val_ppl = np.exp(valid_loss / valid_words)
            #print("ppl for dev {}".format(dev_file_index[0]))
            #print("val_step={0:<6d}".format(step))
            #print(" loss={0:<6.2f}".format(valid_loss / valid_words))
            #print(" acc={0:<5.4f}".format(valid_acc / valid_words))
            #print(" val_ppl={0:<.2f}".format(val_ppl))
            if log:
                print(" val_ppl={0:<.2f}".format(val_ppl))
                print(" acc={0:<5.4f}".format(valid_acc / valid_words))
                print(" loss={0:<6.2f}".format(valid_loss / valid_words))
            valid_words = 0
            valid_loss = 0
            valid_acc = 0
            n_batches = 0
            total_ppl = val_ppl
        if eop:
            break
    model.train()
    model.hparams.decode = decode
    return total_ppl, logits_batch, labels_batch


class ReinforceTrainer():
    def __init__(self, hparams):
        self.hparams = hparams
        self.data_loader = RLDataUtil(hparams)

        print("Training RL...")
        if self.hparams.load_model:
            self.nmt_model = torch.load(
                os.path.join(self.hparams.output_dir,
                             "bleu_final_nmt_model.pt"))
            self.nmt_optim = torch.load(
                os.path.join(self.hparams.output_dir,
                             "bleu_final_nmt_optim.pt"))
            self.actor = torch.load(
                os.path.join(self.hparams.output_dir, "bleu_actor.pt"))
            self.actor_optim = torch.load(
                os.path.join(self.hparams.output_dir, "bleu_actor_optim.pt"))
            self.featurizer = Featurizer(hparams, self.data_loader)
            self.start_time = time.time()
            [
                self.step, self.best_val_ppl, self.best_val_bleu,
                self.cur_attempt, self.lr, self.epoch
            ] = torch.load(
                os.path.join(self.hparams.output_dir,
                             "bleu_final_nmt_extras.pt"))
            if self.hparams.cuda:
                self.nmt_model = self.nmt_model.cuda()
                self.actor = self.actor.cuda()
        else:
            self.nmt_model = Seq2Seq(hparams, self.data_loader)
            if self.hparams.actor_type == "base":
                self.featurizer = Featurizer(hparams, self.data_loader)
                self.actor = Actor(hparams, self.featurizer.num_feature,
                                   self.data_loader.lan_dist_vec)
            else:
                print("actor not implemented")
                exit(0)

            trainable_params = [
                p for p in self.actor.parameters() if p.requires_grad
            ]
            num_params = count_params(trainable_params)
            print("Actor Model has {0} params".format(num_params))
            self.actor_optim = torch.optim.Adam(
                trainable_params,
                lr=self.hparams.lr_q,
                weight_decay=self.hparams.l2_reg)

            if self.hparams.imitate_episode:
                if self.hparams.imitate_type == "heuristic":
                    self.heuristic_actor = HeuristicActor(
                        hparams, self.featurizer.num_feature,
                        self.data_loader.lan_dist_vec)
                elif self.hparams.imitate_type == "init":
                    self.heuristic_actor = InitActor(
                        hparams, self.featurizer.num_feature,
                        self.data_loader.lan_dist_vec)
                else:
                    print("actor not implemented")
                    exit(0)
            elif self.hparams.not_train_score:
                if self.hparams.imitate_type == "heuristic":
                    self.actor = HeuristicActor(hparams,
                                                self.featurizer.num_feature,
                                                self.data_loader.lan_dist_vec)
                elif self.hparams.imitate_type == "init":
                    self.actor = InitActor(hparams,
                                           self.featurizer.num_feature,
                                           self.data_loader.lan_dist_vec)
                else:
                    print("uniform actor")
                    pass

            self.start_time = time.time()
            trainable_params = [
                p for p in self.nmt_model.parameters() if p.requires_grad
            ]
            num_params = count_params(trainable_params)
            print("NMT Model has {0} params".format(num_params))
            if self.hparams.model_optimizer == "SGD":
                self.nmt_optim = customSGD(
                    trainable_params, hparams, lr=self.hparams.lr)
            elif self.hparams.model_optimizer == "ADAM":
                self.nmt_optim = customAdam(
                    trainable_params,
                    hparams,
                    lr=self.hparams.lr,
                    weight_decay=self.hparams.l2_reg)
            else:
                print("optimizer not defined")
                exit(0)

            if self.hparams.cosine_schedule_max_step:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.nmt_optim, self.hparams.cosine_schedule_max_step)
            if self.hparams.cuda:
                self.nmt_model = self.nmt_model.cuda()
                self.actor = self.actor.cuda()

            if hparams.init_type == "uniform" and not hparams.model_type == "transformer":
                print("initialize uniform with range {}".format(
                    hparams.init_range))
                for p in self.nmt_model.parameters():
                    p.data.uniform_(-hparams.init_range, hparams.init_range)
            self.best_val_ppl = [
                None for _ in range(len(hparams.dev_src_file_list))
            ]
            self.best_val_bleu = [
                None for _ in range(len(hparams.dev_src_file_list))
            ]
            self.step = 0
            self.cur_attempt = 0
            self.lr = self.hparams.lr
            self.epoch = 0
            self.baseline = 0

    def train_score(self):
        step = 0
        self.nmt_optim.zero_prev_grad()
        # update the actor with graidents scaled by cosine similarity
        # first update on the base language
        if self.hparams.refresh_base_grad:
            for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train,
                 y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, lan_id,
                 eop) in self.data_loader.next_base_data():
                logits = self.nmt_model.forward(
                    x_train,
                    x_mask,
                    x_len,
                    x_pos_emb_idxs,
                    y_train[:, :-1],
                    y_mask[:, :-1],
                    y_len,
                    y_pos_emb_idxs, [], [],
                    file_idx=[],
                    step=step,
                    x_rank=[])
                logits = logits.view(-1, self.hparams.trg_vocab_size)
                labels = y_train[:, 1:].contiguous().view(-1)
                cur_nmt_loss = torch.nn.functional.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.hparams.pad_id,
                    reduction="none")
                cur_nmt_loss = cur_nmt_loss.view(batch_size, -1).sum().div_(
                    batch_size * self.hparams.update_batch)
                # save the gradients to nmt moving average
                cur_nmt_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.nmt_model.parameters(), self.hparams.clip_grad)
                self.nmt_optim.save_gradients(self.hparams.base_lan_id)
                self.nmt_optim.zero_prev_grad()
                self.nmt_optim.zero_grad()
                if eop:
                    break
        elif self.hparams.refresh_all_grad:
            for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train,
                 y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, lan_id,
                 eop) in self.data_loader.next_refresh_data():
                logits = self.nmt_model.forward(
                    x_train,
                    x_mask,
                    x_len,
                    x_pos_emb_idxs,
                    y_train[:, :-1],
                    y_mask[:, :-1],
                    y_len,
                    y_pos_emb_idxs, [], [],
                    file_idx=[],
                    step=step,
                    x_rank=[])
                logits = logits.view(-1, self.hparams.trg_vocab_size)
                labels = y_train[:, 1:].contiguous().view(-1)
                cur_nmt_loss = torch.nn.functional.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.hparams.pad_id,
                    reduction="none")
                cur_nmt_loss = cur_nmt_loss.view(batch_size, -1).sum().div_(
                    batch_size * self.hparams.update_batch)
                # save the gradients to nmt moving average
                cur_nmt_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.nmt_model.parameters(), self.hparams.clip_grad)
                self.nmt_optim.save_gradients(lan_id[0])
                self.nmt_optim.zero_prev_grad()
                self.nmt_optim.zero_grad()
                if eop:
                    break

        grad_cosine_sim = self.nmt_optim.get_cosine_sim()
        self.nmt_optim.zero_prev_grad()
        self.nmt_optim.zero_grad()
        grad_scale = torch.stack(
            [grad_cosine_sim[idx]
             for idx in range(self.hparams.lan_size)]).view(1, -1)
        print(grad_scale.data)
        if self.hparams.train_on_loaded:
            s_0_list = []
            s_1_list = []
            mask_list = []
            for s in self.data_loader.data_feature:
                step += 1
                mask = 1 - s[1].byte()
                s_0_list.append(s[0])
                s_1_list.append(s[1])
                mask_list.append(mask)
                if eop: break
            s_0 = torch.cat(s_0_list, dim=0)
            s_1 = torch.cat(s_1_list, dim=0)
            mask = torch.cat(mask_list, dim=0)
            for eps in range(self.hparams.train_score_episode):
                a_logits = self.actor([s_0, s_1])
                a_logits.masked_fill_(mask, -float("inf"))

                loss = -torch.nn.functional.log_softmax(a_logits, dim=-1)
                loss = (loss *
                        grad_scale * self.hparams.reward_scale).masked_fill_(
                            mask,
                            0.).sum().div_(self.hparams.train_score_every)
                #loss = (loss * grad_scale * self.hparams.reward_scale).sum().div_(self.hparams.train_score_every)
                cur_loss = loss.item()
                loss.backward()
                self.actor_optim.step()
                self.actor_optim.zero_grad()
        else:
            for eps in range(self.hparams.train_score_episode):
                s_0_list = []
                s_1_list = []
                mask_list = []
                for src, src_len, trg, iter_percent, eop in self.data_loader.next_raw_example_normal(
                ):
                    s = self.featurizer.get_state(src, src_len, trg)
                    step += 1
                    mask = 1 - s[1].byte()
                    s_0_list.append(s[0])
                    s_1_list.append(s[1])
                    mask_list.append(mask)
                    if eop: break
                s_0 = torch.cat(s_0_list, dim=0)
                s_1 = torch.cat(s_1_list, dim=0)
                mask = torch.cat(mask_list, dim=0)
                a_logits = self.actor([s_0, s_1])
                a_logits.masked_fill_(mask, -float("inf"))

                loss = -torch.nn.functional.log_softmax(a_logits, dim=-1)
                loss = (loss *
                        grad_scale * self.hparams.reward_scale).masked_fill_(
                            mask,
                            0.).sum().div_(self.hparams.agent_subsample_line)
                cur_loss = loss.item()
                loss.backward()
                self.actor_optim.step()
                self.actor_optim.zero_grad()
                #if step % self.hparams.print_every == 0:
                #  print("eps={}, actor loss={}".format(eps, cur_loss))

    def imitate_heuristic(self):
        data_batch, next_data_batch = [], []
        batch_count = 0
        step = 0
        cur_dev_ppl = 12
        set_lr(self.actor_optim, 0.001)
        for eps in range(self.hparams.imitate_episode):
            for src, src_len, trg, iter_percent, eop in self.data_loader.next_raw_example_normal(
            ):
                s = self.featurizer.get_state(src, src_len, trg)
                step += 1
                #if step % self.hparams.print_every == 0:
                #  print(s[1])
                a_logits = self.actor(s)
                a_target_logits = self.heuristic_actor(s)
                mask = 1 - s[1].byte()
                a_logits.masked_fill_(mask, -float("inf"))
                a_prob = torch.nn.functional.softmax(a_logits, dim=-1)
                a_target_logits.masked_fill_(mask, -float("inf"))
                a_target_prob = torch.nn.functional.softmax(
                    a_target_logits, dim=-1)
                loss = torch.nn.functional.mse_loss(a_prob, a_target_prob)
                cur_loss = loss.item()
                loss.backward()
                self.actor_optim.step()
                self.actor_optim.zero_grad()
                if step % self.hparams.print_every == 0:
                    print("step={} imitation loss={}".format(step, cur_loss))
                if eop: break

        set_lr(self.actor_optim, self.hparams.lr_q)
        return

    def init_train_nmt(self):
        self.step = 0
        self.cur_attempt = 0
        self.lr = self.hparams.lr
        self.epoch = 0

    def train_nmt_full(self, output_prob_file, n_train_epochs):
        hparams = copy.deepcopy(self.hparams)

        hparams.train_nmt = True
        hparams.output_prob_file = output_prob_file
        hparams.n_train_epochs = n_train_epochs

        model = self.nmt_model
        optim = self.nmt_optim
        #optim = torch.optim.Adam(trainable_params)
        #step = 0
        #cur_attempt = 0
        #lr = hparams.lr

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        num_params = count_params(trainable_params)
        print("Model has {0} params".format(num_params))

        print("-" * 80)
        print("start training...")
        start_time = log_start_time = time.time()
        target_words, total_loss, total_corrects = 0, 0, 0
        target_rules, target_total, target_eos = 0, 0, 0
        total_word_loss, total_rule_loss, total_eos_loss = 0, 0, 0
        model.train()
        #i = 0
        #epoch = 0
        update_batch_size = 0
        #for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, lan_id, eop) in data_util.next_nmt_train():
        for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask,
             y_count, y_len, y_pos_emb_idxs, batch_size, lan_id, eop, eob,
             save_grad) in self.data_loader.next_sample_nmt_train(
                 self.featurizer, self.actor):
            self.step += 1
            target_words += (y_count - batch_size)
            logits = model.forward(
                x_train,
                x_mask,
                x_len,
                x_pos_emb_idxs,
                y_train[:, :-1],
                y_mask[:, :-1],
                y_len,
                y_pos_emb_idxs, [], [],
                file_idx=[],
                step=self.step,
                x_rank=[])
            logits = logits.view(-1, hparams.trg_vocab_size)
            labels = y_train[:, 1:].contiguous().view(-1)

            cur_nmt_loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                ignore_index=self.hparams.pad_id,
                reduction="none")
            total_loss += cur_nmt_loss.sum().item()
            cur_nmt_loss = cur_nmt_loss.view(batch_size, -1).sum(-1).div_(
                batch_size * hparams.update_batch)

            if save_grad and not self.hparams.not_train_score:
                #save the gradients to nmt moving average
                for batch_id in range(batch_size):
                    batch_lan_id = lan_id[batch_id]
                    cur_nmt_loss[batch_id].backward(retain_graph=True)
                    optim.save_gradients(batch_lan_id)
            else:
                cur_nmt_loss = cur_nmt_loss.sum()
                cur_nmt_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       hparams.clip_grad)

            mask = (labels == hparams.pad_id)
            _, preds = torch.max(logits, dim=1)
            cur_tr_acc = torch.eq(preds, labels).int().masked_fill_(mask,
                                                                    0).sum()

            total_corrects += cur_tr_acc.item()

            if self.step % hparams.update_batch == 0:
                optim.step()
                optim.zero_grad()
                optim.zero_prev_grad()
                update_batch_size = 0
                if self.hparams.cosine_schedule_max_step:
                    self.scheduler.step()
            # clean up GPU memory
            if self.step % hparams.clean_mem_every == 0:
                gc.collect()
            if eop:
                if (self.epoch + 1) % (
                        self.hparams.agent_checkpoint_every) == 0:
                    agent_name = "actor_" + str(
                        (self.epoch + 1) //
                        self.hparams.agent_checkpoint_every) + ".pt"
                    agent_save_checkpoint(self.actor, hparams,
                                          hparams.output_dir, agent_name)
                self.epoch += 1
                if self.hparams.cosine_schedule_max_step and self.hparams.schedule_restart:
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.nmt_optim, self.hparams.cosine_schedule_max_step)
            #  get_grad_cos_all(model, data, crit)
            if (self.step / hparams.update_batch) % hparams.log_every == 0:
                curr_time = time.time()
                since_start = (curr_time - start_time) / 60.0
                elapsed = (curr_time - log_start_time) / 60.0
                log_string = "ep={0:<3d}".format(self.epoch)
                log_string += " steps={0:<6.2f}".format(
                    (self.step / hparams.update_batch) / 1000)
                if self.hparams.cosine_schedule_max_step:
                    log_string += " lr={0:<9.7f}".format(
                        self.scheduler.get_lr()[0])
                else:
                    log_string += " lr={0:<9.7f}".format(self.lr)
                log_string += " loss={0:<7.2f}".format(
                    cur_nmt_loss.sum().item())
                log_string += " |g|={0:<5.2f}".format(grad_norm)

                log_string += " ppl={0:<8.2f}".format(
                    np.exp(total_loss / target_words))
                log_string += " acc={0:<5.4f}".format(
                    total_corrects / target_words)

                log_string += " wpm(k)={0:<5.2f}".format(
                    target_words / (1000 * elapsed))
                log_string += " time(min)={0:<5.2f}".format(since_start)
                print(log_string)
            if hparams.eval_end_epoch:
                if eop:
                    eval_now = True
                else:
                    eval_now = False
            elif (self.step / hparams.update_batch) % hparams.eval_every == 0:
                eval_now = True
            else:
                eval_now = False
            if eval_now:
                based_on_bleu = hparams.eval_bleu and self.best_val_ppl[0] is not None and self.best_val_ppl[0] <= hparams.ppl_thresh
                with torch.no_grad():
                    val_ppl, val_bleu, ppl_list, bleu_list = eval(
                        model,
                        self.data_loader,
                        self.step,
                        hparams,
                        hparams,
                        eval_bleu=based_on_bleu,
                        valid_batch_size=hparams.valid_batch_size,
                        tr_logits=logits)
                for i in range(len(ppl_list)):
                    save_bleu, save_ppl = False, False
                    if based_on_bleu:
                        if self.best_val_bleu[i] is None or self.best_val_bleu[i] <= bleu_list[i]:
                            save_bleu = True
                            self.best_val_bleu[i] = bleu_list[i]
                            self.cur_attempt = 0
                        else:
                            save_bleu = False
                            self.cur_attempt += 1
                    if self.best_val_ppl[i] is None or self.best_val_ppl[i] >= ppl_list[i]:
                        save_ppl = True
                        self.best_val_ppl[i] = ppl_list[i]
                        self.cur_attempt = 0
                    else:
                        save_ppl = False
                        self.cur_attempt += 1
                    if save_bleu or save_ppl:
                        if save_bleu:
                            if len(ppl_list) > 1:
                                nmt_save_checkpoint(
                                    [
                                        self.step, self.best_val_ppl,
                                        self.best_val_bleu, self.cur_attempt,
                                        self.lr, self.epoch
                                    ],
                                    model,
                                    optim,
                                    hparams,
                                    hparams.output_dir + "dev{}".format(i),
                                    self.actor,
                                    self.actor_optim,
                                    prefix="bleu_")
                            else:
                                nmt_save_checkpoint(
                                    [
                                        self.step, self.best_val_ppl,
                                        self.best_val_bleu, self.cur_attempt,
                                        self.lr, self.epoch
                                    ],
                                    model,
                                    optim,
                                    hparams,
                                    hparams.output_dir,
                                    self.actor,
                                    self.actor_optim,
                                    prefix="bleu_")
                        if save_ppl:
                            if len(ppl_list) > 1:
                                nmt_save_checkpoint(
                                    [
                                        self.step, self.best_val_ppl,
                                        self.best_val_bleu, self.cur_attempt,
                                        self.lr, self.epoch
                                    ],
                                    model,
                                    optim,
                                    hparams,
                                    hparams.output_dir + "dev{}".format(i),
                                    self.actor,
                                    self.actor_optim,
                                    prefix="ppl_")
                            else:
                                nmt_save_checkpoint(
                                    [
                                        self.step, self.best_val_ppl,
                                        self.best_val_bleu, self.cur_attempt,
                                        self.lr, self.epoch
                                    ],
                                    model,
                                    optim,
                                    hparams,
                                    hparams.output_dir,
                                    self.actor,
                                    self.actor_optim,
                                    prefix="ppl_")
                    elif not hparams.lr_schedule and self.step >= hparams.n_warm_ups:
                        self.lr = self.lr * hparams.lr_dec
                        set_lr(optim, self.lr)
                # reset counter after eval
                log_start_time = time.time()
                target_words = total_corrects = total_loss = 0
                target_rules = target_total = target_eos = 0
                total_word_loss = total_rule_loss = total_eos_loss = 0
            if hparams.patience >= 0:
                if self.cur_attempt > hparams.patience: break
            elif hparams.n_train_epochs > 0:
                if self.epoch >= hparams.n_train_epochs: break
            else:
                if self.step > hparams.n_train_steps: break
            if eob: break

    def train_rl_and_nmt(self):
        # imitate a good policy agent first
        if self.hparams.imitate_episode and not self.hparams.load_model:
            self.imitate_heuristic()
        while True:
            self.train_nmt_full(
                self.hparams.output_prob_file,
                n_train_epochs=(
                    self.hparams.n_train_epochs // self.hparams.iteration))
            if self.hparams.patience >= 0:
                if self.cur_attempt > self.hparams.patience: break
            elif self.hparams.n_train_epochs > 0:
                if self.epoch >= self.hparams.n_train_epochs: break
            else:
                if self.step > self.hparams.n_train_steps: break
            if self.hparams.not_train_score:
                pass
            else:
                self.train_score()
