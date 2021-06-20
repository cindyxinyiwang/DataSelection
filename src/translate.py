from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import _pickle as pickle
import shutil
import gc
import os
import sys
import time
import subprocess
import numpy as np
import re

from rl_data_utils import RLDataUtil
from hparams import *
from utils import *
from model import *

import torch
import torch.nn as nn
from torch.autograd import Variable

class TranslationHparams(HParams):
  dataset = "Translate dataset"

parser = argparse.ArgumentParser(description="Neural MT translator")

parser.add_argument("--cuda", action="store_true", help="GPU or not")
parser.add_argument("--data_path", type=str, default=None, help="path to all data")
parser.add_argument("--model_file", type=str, default="outputs", help="root directory of saved model")
parser.add_argument("--hparams_file", type=str, default="outputs", help="root directory of saved model")
parser.add_argument("--test_src_file", type=str, default=None, help="name of source test file")
parser.add_argument("--test_src_file_list", type=str, default=None, help="name of source test file")
parser.add_argument("--test_trg_file", type=str, default=None, help="name of target test file")
parser.add_argument("--test_trg_file_list", type=str, default=None, help="name of target test file")
parser.add_argument("--test_ref_file_list", type=str, default=None, help="name of target test file")
parser.add_argument("--test_file_idx_list", type=str, default=None, help="name of target test file")
parser.add_argument("--beam_size", type=int, default=None, help="beam size")
parser.add_argument("--max_len", type=int, default=300, help="maximum len considered on the target side")
parser.add_argument("--poly_norm_m", type=float, default=0, help="m in polynormial normalization")
parser.add_argument("--non_batch_translate", action="store_true", help="use non-batched translation")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--merge_bpe", action="store_true", help="")
parser.add_argument("--src_vocab_list", type=str, default=None, help="name of source vocab file")
parser.add_argument("--trg_vocab_list", type=str, default=None, help="name of target vocab file")
parser.add_argument("--n_train_sents", type=int, default=None, help="max number of training sentences to load")
parser.add_argument("--out_file_list", type=str, default="trans", help="output file for hypothesis")
parser.add_argument("--log_file", type=str, default="trans_log", help="output file for logs")
parser.add_argument("--debug", action="store_true", help="output file for hypothesis")

parser.add_argument("--nbest", action="store_true", help="whether to return the nbest list")
parser.add_argument("--ppl_only", action="store_true", help="whether to return the nbest list")
parser.add_argument("--trans_dev", action="store_true", help="whether to return the nbest list")
parser.add_argument("--out_prefix", type=str, default="", help="output file for logs")

def test(model, hparams_file, ppl_only, trans_dev, out_file_list, log_file, test_src_file_list, test_trg_file_list, test_ref_file_list, \
        test_file_idx_list, cuda, beam_size, max_len, poly_norm_m, batch_size, merge_bpe, out_prefix):
  hparams_file_name = hparams_file
  train_hparams = torch.load(hparams_file_name)
  hparams = TranslationHparams()
  for k, v in train_hparams.__dict__.items():
    setattr(hparams, k, v)
  
  if not ppl_only:
    out_file_list = out_file_list.split(",")
    print("writing translation to " + str(out_file_list))
  log_file = open(log_file, "w")
  
  hparams.test_src_file_list = test_src_file_list.split(",")
  hparams.test_trg_file_list = test_trg_file_list.split(",")
  hparams.test_ref_file_list = test_ref_file_list.split(",")
  hparams.test_file_idx_list = [int(i) for i in test_file_idx_list.split(",")]
  hparams.cuda=cuda
  hparams.beam_size=beam_size
  hparams.max_len=max_len
  hparams.merge_bpe=merge_bpe
  
  model.hparams.cuda = hparams.cuda
  data = RLDataUtil(hparams=hparams)
  filts = [model.hparams.pad_id, model.hparams.eos_id, model.hparams.bos_id]
  
  end_of_epoch = False
  num_sentences = 0
  
  with torch.no_grad():
    test_words = 0
    test_loss = 0
    test_acc = 0
    n_batches = 0
    total_ppl, total_bleu = 0, 0
    ppl_list, bleu_list = [], []
    for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, test_file_idx, x_rank in data.next_test(test_batch_size=8):
      # next batch
      y_count -= batch_size
      # word count
      test_words += y_count
  
      logits = model.forward(
        x, x_mask, x_len, x_pos_emb_idxs,
        y[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_char, y_char, file_idx=test_file_idx, step=0, x_rank=x_rank)
      logits = logits.view(-1, hparams.trg_vocab_size)
      labels = y[:,1:].contiguous().view(-1)
      val_loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=hparams.pad_id, reduction="none").sum()
      mask = (labels == hparams.pad_id)
      _, preds = torch.max(logits, dim=1)
      val_acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum()
     
      n_batches += batch_size
      test_loss += val_loss.item()
      test_acc += val_acc.item()
      if eof:
        val_ppl = np.exp(test_loss / test_words)
        print(" loss={0:<6.2f}".format(test_loss / test_words))
        log_file.write(" loss={0:<6.2f}\n".format(test_loss / test_words))
        print(" acc={0:<5.4f}".format(test_acc / test_words))
        log_file.write(" acc={0:<5.4f}\n".format(test_acc / test_words))
        print(" test_ppl={0:<.2f}".format(val_ppl))
        log_file.write(" test_ppl={0:<.2f}\n".format(val_ppl))
        test_words = 0
        test_loss = 0
        test_acc = 0
        n_batches = 0
        total_ppl += val_ppl
        ppl_list.append(val_ppl)
      if eop:
        break
  
    if trans_dev:
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      total_ppl, total_bleu = 0, 0
      ppl_list, bleu_list = [], []
      valid_bleu = None
      for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, test_file_idx, x_rank in data.next_dev(dev_batch_size=8):
        # next batch
        y_count -= batch_size
        # word count
        valid_words += y_count
  
        logits = model.forward(
          x, x_mask, x_len, x_pos_emb_idxs,
          y[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_char, y_char, file_idx=test_file_idx, step=0, x_rank=x_rank)
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
          print(" loss={0:<6.2f}".format(valid_loss / valid_words))
          log_file.write(" loss={0:<6.2f}\n".format(valid_loss / valid_words))
          print(" acc={0:<5.4f}".format(valid_acc / valid_words))
          log_file.write(" acc={0:<5.4f}\n".format(valid_acc / valid_words))
          print(" val_ppl={0:<.2f}".format(val_ppl))
          log_file.write(" val_ppl={0:<.2f}\n".format(val_ppl))
          valid_words = 0
          valid_loss = 0
          valid_acc = 0
          n_batches = 0
          total_ppl += val_ppl
          ppl_list.append(val_ppl)
        if eop:
          break
  
    if ppl_only: exit(0)
  
  with torch.no_grad():
    out_file = open(out_file_list[0], 'w', encoding='utf-8')
  
    test_idx = 0
    for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, test_file_idx, x_rank in data.next_test(test_batch_size=1):
      hs = model.translate(
              x, x_mask, beam_size=beam_size, max_len=max_len, poly_norm_m=poly_norm_m, x_train_char=[], y_train_char=[], file_idx=test_file_idx)
      for h in hs:
        h_best_words = map(lambda wi: data.trg_i2w[wi],
                         filter(lambda wi: wi not in filts, h))
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
        print("finished translating {}".format(out_file_list[test_idx]))
        ref_file = hparams.test_ref_file_list[test_idx]
        bleu_str = subprocess.getoutput(
          "./multi-bleu.perl {0} < {1}".format(ref_file, out_file_list[test_idx]))
        bleu_str = bleu_str.split('\n')[-1].strip()
        reg = re.compile("BLEU = ([^,]*).*")
        try:
          valid_bleu = float(reg.match(bleu_str).group(1))
        except:
          valid_bleu = 0.
        print(" test_bleu={0:<.2f}".format(valid_bleu))
        log_file.write(" test_bleu={0:<.2f}\n".format(valid_bleu))
  
        test_idx += 1
        if not eop:
          out_file = open(out_file_list[test_idx], "w", encoding="utf-8")
      if eop:
        break    
  
    if trans_dev:
      valid_hyp_file_list = [os.path.join(hparams.output_dir, "{}_dev{}.trans_final".format(out_prefix, i)) for i in hparams.dev_file_idx_list]
      out_file = open(valid_hyp_file_list[0], 'w', encoding='utf-8')
      dev_idx = 0
      for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, dev_file_index, x_rank in data.next_dev(dev_batch_size=1):
        hs = model.translate(
                x, x_mask, beam_size=beam_size, max_len=max_len, poly_norm_m=poly_norm_m, file_idx=[])
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
          ref_file = hparams.dev_ref_file_list[dev_idx]
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
          log_file.write(" val_bleu={0:<.2f}\n".format(valid_bleu))
          dev_idx += 1
          if not eop:
            out_file = open(valid_hyp_file_list[dev_idx], "w", encoding="utf-8")
        if eop:
          break

if __name__ == "__main__":
  args = parser.parse_args()
  
  model_file_name = args.model_file
  if not args.cuda:
    model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
  else:
    model = torch.load(model_file_name)
  model.eval()
  test(model, args.hparams_file, args.ppl_only, args.trans_dev, args.out_file_list, args.log_file, args.test_src_file_list, args.test_trg_file_list, args.test_ref_file_list, \
        args.test_file_idx_list, args.cuda, args.beam_size, args.max_len, args.poly_norm_m, args.batch_size, args.merge_bpe, args.out_prefix)

