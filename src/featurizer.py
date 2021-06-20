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

from collections import defaultdict, deque
import copy
import time


class Featurizer():
  def __init__(self, hparams, data_loader):
    self.hparams = hparams
    self.data_loader = data_loader
    self.num_feature = self.hparams.lan_size

  def get_state(self, src, src_len, trg):
    existed_src = (np.array(src_len) > 2).astype(int).sum(axis=0)
    if self.hparams.feature_type == "lan_dist":
      src_dist = existed_src * self.data_loader.lan_dist_vec / 100
      #data_feature = np.append(src_dist, [iter_percent, cur_dev_ppl]).reshape(1, -1)
      #data_feature = np.append(src_dist, [iter_percent]).reshape(1, -1)
      data_feature = src_dist.reshape(1, -1)
      data_feature = Variable(torch.FloatTensor(data_feature))
    elif self.hparams.feature_type == "zero_one":
      data_feature = existed_src.reshape(1, -1)
      data_feature = Variable(torch.FloatTensor(data_feature))
    elif self.hparams.feature_type == "one":
      data_feature = Variable(torch.ones((1, len(existed_src)))) 

    existed_src = Variable(torch.FloatTensor([existed_src.tolist()]))
    if self.hparams.cuda: 
      data_feature = data_feature.cuda()
      existed_src = existed_src.cuda()
    return [data_feature, existed_src]


