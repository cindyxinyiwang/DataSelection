import os
import sys
import time
import gc
import subprocess
import re

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def sample_action(a_logits, temp=0.01, log=False):
  if log:
    print(a_logits)
  prob = torch.nn.functional.softmax(a_logits * temp, -1)
  #prob = a_logits / a_logits.sum(dim=-1)
  a = torch.distributions.Categorical(prob).sample()
  prob = [i for i in prob.data.view(-1).cpu().numpy()]
  if log:
    print(prob)
    print(a)
  return a, prob

def memReport():
  for obj in gc.get_objects():
    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
      print(type(obj), obj.size())

def get_criterion(hparams):
  loss_reduce = False
  crit = nn.CrossEntropyLoss(ignore_index=hparams.pad_id, reduction="none")
  if hparams.cuda:
    crit = crit.cuda()
  return crit

def get_criterion_q(hparams):
  loss_reduce = False
  crit = nn.MSELoss(size_average=False, reduce=loss_reduce)
  if hparams.cuda:
    crit = crit.cuda()
  return crit

def get_performance_q(crit, logits, labels, hparams, sum_loss=True):
  loss = crit(logits, labels)
  _, preds = torch.max(logits, dim=1)
  if sum_loss: loss = loss.sum()
  return loss

def get_performance(crit, logits, labels, hparams, sum_loss=True, logits_q=None, batch_size=None, element_weight=None):
  if logits_q is not None:
    _, trg_vocab_size = logits.size()
    loss_p = crit(logits, labels).view(batch_size, -1).sum(-1)
    loss_q = crit(logits_q, labels).view(batch_size, -1).sum(-1)
    weight = torch.exp(loss_p.data - loss_q.data)
    ones = torch.FloatTensor([1]).expand_as(weight)
    if hparams.cuda: ones = ones.cuda()
    weight = torch.min(weight, ones)
    if hparams.mask_weight > 0:
      mask = weight <= hparams.mask_weight
      weight.masked_fill_(mask, 0)
    loss = loss_p.view(batch_size, -1) * weight.unsqueeze(1)
    loss = loss.view(-1) 
  else:
    loss = crit(logits, labels)
  if element_weight is not None:
    loss = loss.view(batch_size, -1)
    loss = loss * element_weight
  mask = (labels == hparams.pad_id)
  _, preds = torch.max(logits, dim=1)
  acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum()
  if sum_loss: loss = loss.sum()
  return loss, acc

def count_params(params):
  num_params = sum(p.data.nelement() for p in params)
  return num_params

def save_checkpoint(model, agent, hparams, path, save_agent=False):
  print("Saving model to '{0}'".format(path))
  torch.save(model, os.path.join(path, "model.pt"))
  if save_agent:
    torch.save(agent, os.path.join(path, "agent.pt"))
  torch.save(hparams, os.path.join(path, "hparams.pt"))

def agent_save_checkpoint(agent, hparams, path, agent_name):
  print("Saving agent to '{0}'".format(path))
  torch.save(agent, os.path.join(path, agent_name))
  torch.save(hparams, os.path.join(path, "hparams.pt"))

def nmt_save_checkpoint(extras, model, optim, hparams, path, actor, actor_optim, prefix=""):
  print("Saving model to '{0}'".format(path))
  torch.save(model, os.path.join(path, prefix+"final_nmt_model.pt"))
  torch.save(hparams, os.path.join(path, prefix+"final_nmt_hparams.pt"))
  torch.save(extras, os.path.join(path, prefix+"final_nmt_extras.pt"))
  torch.save(optim, os.path.join(path, prefix+"final_nmt_optim.pt"))
  torch.save(actor, os.path.join(path, prefix+"actor.pt"))
  torch.save(actor_optim, os.path.join(path, prefix+"actor_optim.pt"))
  #torch.save(actor, os.path.join(path, actor_name))


class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

def set_lr(optim, lr):
  for param_group in optim.param_groups:
    param_group["lr"] = lr

def init_param(p, init_type="uniform", init_range=None):
  if init_type == "xavier_normal":
    init.xavier_normal(p)
  elif init_type == "xavier_uniform":
    init.xavier_uniform(p)
  elif init_type == "kaiming_normal":
    init.kaiming_normal(p)
  elif init_type == "kaiming_uniform":
    init.kaiming_uniform(p)
  elif init_type == "uniform":
    #assert init_range is not None and init_range > 0
    init.uniform_(p, -init_range, init_range)
  else:
    raise ValueError("Unknown init_type '{0}'".format(init_type))


def get_attn_subsequent_mask(seq, pad_id=0):
  """ Get an attention mask to avoid using the subsequent info."""

  assert seq.dim() == 2
  batch_size, max_len = seq.size()
  sub_mask = torch.triu(
    torch.ones(max_len, max_len), diagonal=1).unsqueeze(0).repeat(
      batch_size, 1, 1).type(torch.ByteTensor)
  if seq.is_cuda:
    sub_mask = sub_mask.cuda()
  return sub_mask

def grad_clip(params, grad_bound=None):
  """Clipping gradients at L-2 norm grad_bound. Returns the L-2 norm."""

  params = list(filter(lambda p: p.grad is not None, params))
  total_norm = 0
  for p in params:
    if p.grad is None:
      continue
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm ** 2
  total_norm = total_norm ** 0.5

  if grad_bound is not None:
    clip_coef = grad_bound / (total_norm + 1e-6)
    if clip_coef < 1:
      for p in params:
        p.grad.data.mul_(clip_coef)
  return total_norm

def get_grad_cos(model, data, crit):
  i = 0
  step = 0 
  grads = []
  dists = [100 for _ in range(model.hparams.lan_size)]
  data_count = 0
  for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank) in data.next_train_select():
    assert file_idx[0] == i % model.hparams.lan_size
    i += 1
    target_words = (y_count - batch_size)
    logits = model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_train_char_sparse, y_train_char_sparse, file_idx=file_idx, step=step, x_rank=x_rank)
    logits = logits.view(-1, model.hparams.trg_vocab_size)
    labels = y_train[:,1:].contiguous().view(-1)
      
    cur_tr_loss, cur_tr_acc = get_performance(crit, logits, labels, model.hparams)
    total_loss = cur_tr_loss.item()
    total_corrects = cur_tr_acc.item()
    cur_tr_loss.div_(batch_size)
    cur_tr_loss.backward()
    #print(file_idx[0])
    #params = list(filter(lambda p: p.grad is not None, model.parameters()))
    params_dict = model.state_dict()
    params =  list(model.parameters())
    #for k, v in params_dict.items():
    #  print(k)
    #  print(v[0])
    #  break
    #  print(v.size())
    #for v in model.parameters():
    #  print(v.size())
    grad = {}
    d = 0
    for k, v in params_dict.items():
      if params[d].grad is not None: 
        grad[k] = params[d].grad.data.clone()
        params[d].grad.data.zero_()
      d += 1
    grads.append(grad)
    if file_idx[0] == model.hparams.lan_size-1:
      data_count += 1
      for j in range(1, model.hparams.lan_size):
        dist = 0
        if data_count == 1:
          print(data.lans[j])
        for k in grads[0].keys():
          p0 = grads[0][k]
          p1 = grads[j][k]
          p0_unit = p0 / (p0.norm(2) + 1e-10)
          p1_unit = p1 / (p1.norm(2) + 1e-10)
          cosine = (p0_unit * p1_unit).sum()

          #if "enc" in k or "decoder.attention" in k:
          if "enc" in k:
            dist = dist + cosine.item()
          if data_count == 1:
            print("{} : {}".format(k, cosine))
        dists[j] += dist
      grads = []
      if data_count == 5:
        break
  dists = [d / data_count for d in dists]
  for j in range(1, model.hparams.lan_size):
    print(data.lans[j])
    print(dists[j])
  data.update_prob_list(dists)


def get_grad_cos_all(model, data, crit):
  i = 0
  step = 0 
  grads = []
  dists = [100 for _ in range(model.hparams.lan_size)]
  data_count = 0
  for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank) in data.next_train_select_all():
    #assert file_idx[0] == (i // 2) % model.hparams.lan_size
    i += 1
    target_words = (y_count - batch_size)
    logits = model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_train_char_sparse, y_train_char_sparse, file_idx=file_idx, step=step, x_rank=x_rank)
    logits = logits.view(-1, model.hparams.trg_vocab_size)
    labels = y_train[:,1:].contiguous().view(-1)
      
    cur_tr_loss, cur_tr_acc = get_performance(crit, logits, labels, model.hparams)
    total_loss = cur_tr_loss.item()
    total_corrects = cur_tr_acc.item()
    cur_tr_loss.div_(batch_size)
    cur_tr_loss.backward()
    #print(file_idx[0])
    #params = list(filter(lambda p: p.grad is not None, model.parameters()))
    params_dict = model.state_dict()
    params =  list(model.parameters())
    #for k, v in params_dict.items():
    #  print(k)
    #  print(v[0])
    #  break
    #  print(v.size())
    #for v in model.parameters():
    #  print(v.size())
    grad = {}
    d = 0
    for k, v in params_dict.items():
      if params[d].grad is not None: 
        grad[k] = params[d].grad.data.clone()
        params[d].grad.data.zero_()
      d += 1
    grads.append(grad)
    if file_idx[0] != 0:
      data_count += 1
      data_idx = file_idx[0]
      dist = 0
      if data_count == data.ave_grad:
        print(data.lans[data_idx])
      for k in grads[0].keys():
        p0 = grads[0][k]
        p1 = grads[1][k]
        p0_unit = p0 / (p0.norm(2) + 1e-10)
        p1_unit = p1 / (p1.norm(2) + 1e-10)
        cosine = (p0_unit * p1_unit).sum()

        #if "enc" in k or "decoder.attention" in k:
        if "encoder.word_emb" in k:
          dist = dist + cosine.item()
        if data_count == data.ave_grad:
          print("{} : {}".format(k, cosine))
      dists[data_idx] += dist
      grads = []
      if file_idx[0] == model.hparams.lan_size - 1 and data_count == data.ave_grad:
        break
      if data_count == data.ave_grad: data_count = 0

  dists = [d / data_count for d in dists]
  for j in range(1, model.hparams.lan_size):
    print(data.lans[j])
    print(dists[j])
  data.update_prob_list(dists)

def eval(model, data, step, hparams, args, eval_bleu=False,
         valid_batch_size=20, tr_logits=None):
  print("Eval at step {0}. valid_batch_size={1}".format(step, valid_batch_size))
  model.hparams.decode = True
  model.eval()
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_ppl, total_bleu = 0, 0
  ppl_list, bleu_list = [], []
  valid_bleu = None
  for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, dev_file_index, x_rank in data.next_dev(dev_batch_size=valid_batch_size):
    # clear GPU memory
    gc.collect()

    # next batch
    # do this since you shift y_valid[:, 1:] and y_valid[:, :-1]
    y_count -= batch_size
    # word count
    valid_words += y_count

    logits = model.forward(
      x, x_mask, x_len, x_pos_emb_idxs,
      y[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_char, y_char, file_idx=dev_file_index, step=step, x_rank=x_rank)
    logits = logits.view(-1, hparams.trg_vocab_size)
    labels = y[:,1:].contiguous().view(-1)
    val_loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=hparams.pad_id, reduction="none").sum()
    mask = (labels == hparams.pad_id)
    _, preds = torch.max(logits, dim=1)
    val_acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum()
   
    n_batches += batch_size
    valid_loss += val_loss.item()
    valid_acc += val_acc.item()
    if eof:
      val_ppl = np.exp(valid_loss / valid_words)
      print("ppl for dev {}".format(dev_file_index[0]))
      print("val_step={0:<6d}".format(step))
      print(" loss={0:<6.2f}".format(valid_loss / valid_words))
      print(" acc={0:<5.4f}".format(valid_acc / valid_words))
      print(" val_ppl={0:<.2f}".format(val_ppl))
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      total_ppl += val_ppl
      ppl_list.append(val_ppl)
    if eop:
      break
  # BLEU eval
  if eval_bleu:
    valid_hyp_file_list = [os.path.join(args.output_dir, "dev{}.trans_{}".format(i, step)) for i in hparams.dev_file_idx_list]
    out_file = open(valid_hyp_file_list[0], 'w', encoding='utf-8')
    if args.detok:
      valid_hyp_detok_file_list = [os.path.join(args.output_dir, "dev{}.trans_{}.detok".format(i, step)) for i in hparams.dev_file_idx_list]
    dev_idx = 0
    for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, dev_file_index, x_rank in data.next_dev(dev_batch_size=1):
      hs = model.translate(
              x, x_mask, beam_size=args.beam_size, max_len=args.max_trans_len, poly_norm_m=args.poly_norm_m, x_train_char=[], y_train_char=[], file_idx=[], step=step, x_rank=x_rank)
      for h in hs:
        h_best_words = map(lambda wi: data.trg_i2w[wi],
                         filter(lambda wi: wi not in [hparams.bos_id, hparams.eos_id], h))
        if hparams.merge_bpe:
          line = ''.join(h_best_words)
          line = line.replace('▁', ' ')
        else:
          line = ' '.join(h_best_words)
        line = line.strip()
        out_file.write(line + '\n')
        out_file.flush()
      if eof:
        out_file.close()
        if args.detok:
          _ = subprocess.getoutput(
          "python src/reversible_tokenize.py --detok < {0} > {1}".format(valid_hyp_file_list[dev_idx], valid_hyp_detok_file_list[dev_idx]))

        ref_file = hparams.dev_ref_file_list[dev_idx]
        if args.detok:
          bleu_str = subprocess.getoutput(
            "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_detok_file_list[dev_idx]))
        else:
          bleu_str = subprocess.getoutput(
            "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_file_list[dev_idx]))
        print("bleu for dev {}".format(dev_idx))
        print("{}".format(bleu_str))
        bleu_str = bleu_str.split('\n')[-1].strip()
        reg = re.compile("BLEU = ([^,]*).*")
        try:
          valid_bleu = float(reg.match(bleu_str).group(1))
        except:
          valid_bleu = 0.
        print(" val_bleu={0:<.2f}".format(valid_bleu))
        total_bleu += valid_bleu
        dev_idx += 1
        bleu_list.append(valid_bleu)
        if not eop:
          out_file = open(valid_hyp_file_list[dev_idx], "w", encoding="utf-8")
      if eop:
        break
  model.hparams.decode = False
  model.train()
  return total_ppl, total_bleu, ppl_list, bleu_list

def eval_actor(model, nmt_actor, actor_optim, data, crit, step, hparams, args, eval_bleu=False,
         valid_batch_size=20, tr_logits=None):
  total_tr_loss, total_words = 0, 0
  for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, dev_file_index, x_rank in data.next_dev(dev_batch_size=valid_batch_size):
    # clear GPU memory
    gc.collect()

    # next batch
    # do this since you shift y_valid[:, 1:] and y_valid[:, :-1]
    y_count -= batch_size
    # word count
    total_words += y_count
    #logits = model.forward(
    #  x, x_mask, x_len, x_pos_emb_idxs,
    #  y[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_char, y_char, file_idx=dev_file_index, step=step, x_rank=x_rank)
    #logits = logits.view(-1, hparams.trg_vocab_size)
    #labels = y[:,1:].contiguous().view(-1)
    
    scores = nmt_actor.forward(x, x_mask, x_len, x_pos_emb_idxs, y[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs)
    #cur_tr_loss, cur_tr_acc = get_performance(crit, logits, labels, hparams, batch_size=batch_size, element_weight=scores)
    #cur_tr_loss.div_(min(scores.sum().item() * hparams.update_batch, 1))
    target = scores.clone().fill_(1.)
    cur_tr_loss = torch.nn.functional.mse_loss(scores, target)

    total_tr_loss += cur_tr_loss.item()

    cur_tr_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(nmt_actor.parameters(), args.clip_grad)
    actor_optim.step()
    actor_optim.zero_grad()

    if eof:
      #val_ppl = np.exp(total_tr_loss / total_words)
      print("train actor loss={0:<6.2f}".format(total_tr_loss))
      #print("train actor val_ppl={0:<.2f}".format(val_ppl))
    if eop:
      break

  with torch.no_grad():
    print("Eval at step {0}. valid_batch_size={1}".format(step, valid_batch_size))
    model.hparams.decode = True
    model.eval()
    valid_words = 0
    valid_loss = 0
    valid_acc = 0
    n_batches = 0
    total_ppl, total_bleu = 0, 0
    ppl_list, bleu_list = [], []
    valid_bleu = None
    for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, dev_file_index, x_rank in data.next_dev(dev_batch_size=valid_batch_size):
      # clear GPU memory
      gc.collect()

      # next batch
      # do this since you shift y_valid[:, 1:] and y_valid[:, :-1]
      y_count -= batch_size
      # word count
      valid_words += y_count

      logits = model.forward(
        x, x_mask, x_len, x_pos_emb_idxs,
        y[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_char, y_char, file_idx=dev_file_index, step=step, x_rank=x_rank)
      logits = logits.view(-1, hparams.trg_vocab_size)
      labels = y[:,1:].contiguous().view(-1)
      val_loss, val_acc = get_performance(crit, logits, labels, hparams)
      n_batches += batch_size
      valid_loss += val_loss.item()
      valid_acc += val_acc.item()

      if eof:
        val_ppl = np.exp(valid_loss / valid_words)
        print("ppl for dev {}".format(dev_file_index[0]))
        print("val_step={0:<6d}".format(step))
        print(" loss={0:<6.2f}".format(valid_loss / valid_words))
        print(" acc={0:<5.4f}".format(valid_acc / valid_words))
        print(" val_ppl={0:<.2f}".format(val_ppl))
        valid_words = 0
        valid_loss = 0
        valid_acc = 0
        n_batches = 0
        total_ppl += val_ppl
        ppl_list.append(val_ppl)
      if eop:
        break
    # BLEU eval
    if eval_bleu:
      valid_hyp_file_list = [os.path.join(args.output_dir, "dev{}.trans_{}".format(i, step)) for i in hparams.dev_file_idx_list]
      out_file = open(valid_hyp_file_list[0], 'w', encoding='utf-8')
      if args.detok:
        valid_hyp_detok_file_list = [os.path.join(args.output_dir, "dev{}.trans_{}.detok".format(i, step)) for i in hparams.dev_file_idx_list]
      dev_idx = 0
      for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, dev_file_index, x_rank in data.next_dev(dev_batch_size=1):
        hs = model.translate(
                x, x_mask, beam_size=args.beam_size, max_len=args.max_trans_len, poly_norm_m=args.poly_norm_m, x_train_char=[], y_train_char=[], file_idx=[], step=step, x_rank=x_rank)
        for h in hs:
          h_best_words = map(lambda wi: data.trg_i2w[wi],
                           filter(lambda wi: wi not in [hparams.bos_id, hparams.eos_id], h))
          if hparams.merge_bpe:
            line = ''.join(h_best_words)
            line = line.replace('▁', ' ')
          else:
            line = ' '.join(h_best_words)
          line = line.strip()
          out_file.write(line + '\n')
          out_file.flush()
        if eof:
          out_file.close()
          if args.detok:
            _ = subprocess.getoutput(
            "python src/reversible_tokenize.py --detok < {0} > {1}".format(valid_hyp_file_list[dev_idx], valid_hyp_detok_file_list[dev_idx]))

          ref_file = hparams.dev_ref_file_list[dev_idx]
          if args.detok:
            bleu_str = subprocess.getoutput(
              "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_detok_file_list[dev_idx]))
          else:
            bleu_str = subprocess.getoutput(
              "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_file_list[dev_idx]))
          print("bleu for dev {}".format(dev_idx))
          print("{}".format(bleu_str))
          bleu_str = bleu_str.split('\n')[-1].strip()
          reg = re.compile("BLEU = ([^,]*).*")
          try:
            valid_bleu = float(reg.match(bleu_str).group(1))
          except:
            valid_bleu = 0.
          print(" val_bleu={0:<.2f}".format(valid_bleu))
          total_bleu += valid_bleu
          dev_idx += 1
          bleu_list.append(valid_bleu)
          if not eop:
            out_file = open(valid_hyp_file_list[dev_idx], "w", encoding="utf-8")
        if eop:
          break
    model.hparams.decode = False
    model.train()
    return total_ppl, total_bleu, ppl_list, bleu_list


