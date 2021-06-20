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


class Actor(nn.Module):
    def __init__(self, hparams, num_feature, lan_dist_vec):
        super(Actor, self).__init__()
        self.hparams = hparams
        hidden_size = hparams.d_hidden
        self.lan_dist_vec = Variable(
            torch.FloatTensor(lan_dist_vec.tolist()) / 10)
        self.w = nn.Linear(num_feature, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.decoder = nn.Linear(
            hidden_size, self.hparams.lan_size, bias=False)
        # init
        for p in self.decoder.parameters():
            init.uniform_(p, -self.hparams.actor_init_range,
                          self.hparams.actor_init_range)
        for p in self.w.parameters():
            init.uniform_(p, -self.hparams.actor_init_range,
                          self.hparams.actor_init_range)
            #init.uniform_(p, -0.1, 0.1)
        for p in self.w2.parameters():
            init.uniform_(p, -self.hparams.actor_init_range,
                          self.hparams.actor_init_range)
            #init.uniform_(p, -0.1, 0.1)

        if self.hparams.add_bias:
            self.bias = nn.Linear(1, self.hparams.lan_size, bias=False)
            self.bias.weight = torch.nn.Parameter(
                torch.FloatTensor([[
                    self.hparams.bias for _ in range(self.hparams.lan_size)
                ]]))
            self.bias.weight.requires_grad = False
        #self.decoder.bias = torch.nn.Parameter(torch.FloatTensor(lan_dist_vec))
        if self.hparams.cuda:
            self.lan_dist_vec = self.lan_dist_vec.cuda()
            self.w = self.w.cuda()
            self.w2 = self.w2.cuda()
            self.decoder = self.decoder.cuda()
            if self.hparams.add_bias:
                self.bias = self.bias.cuda()

    def forward(self, feature):
        #(model_feature, language_feature, data_feature) = feature
        feature, existed_src = feature
        batch_size = feature.size(0)

        if self.hparams.norm_feature:
            enc = self.w(feature / feature.sum(dim=-1).view(batch_size, -1))
        else:
            enc = self.w(feature)
        enc = torch.relu(enc)
        enc = self.w2(enc)
        enc = torch.relu(enc)
        if self.hparams.add_bias:
            bias = self.bias.weight * existed_src * self.lan_dist_vec
            logit = self.decoder(enc) + bias
        else:
            logit = self.decoder(enc)
        return logit


class HeuristicActor(nn.Module):
    def __init__(self, hparams, num_feature, lan_dist_vec):
        super(HeuristicActor, self).__init__()
        self.hparams = hparams
        hidden_size = hparams.d_hidden
        #self.lan_dist_vec = Variable(torch.FloatTensor(lan_dist_vec.tolist()) / 10)
        self.lan_dist_vec = Variable(
            torch.FloatTensor(lan_dist_vec.tolist()) *
            self.hparams.hs_actor_temp)

        self.bias = nn.Linear(1, self.hparams.lan_size, bias=False)
        self.bias.weight = torch.nn.Parameter(
            torch.FloatTensor(
                [[self.hparams.bias for _ in range(self.hparams.lan_size)]]))
        self.bias.weight.requires_grad = False
        if self.hparams.cuda:
            self.bias = self.bias.cuda()
            self.lan_dist_vec = self.lan_dist_vec.cuda()

    def forward(self, feature):
        #(model_feature, language_feature, data_feature) = feature
        feature, existed_src = feature

        bias_logit = self.bias.weight * existed_src * self.lan_dist_vec
        return bias_logit


class InitActor(nn.Module):
    def __init__(self, hparams, num_feature, lan_dist_vec):
        super(InitActor, self).__init__()
        self.hparams = hparams
        hidden_size = hparams.d_hidden
        lan_vector = [0 for _ in range(self.hparams.lan_size)]
        lan_vector[self.hparams.base_lan_id] = 100
        self.lan_dist_vec = Variable(torch.FloatTensor(lan_vector))
        if self.hparams.cuda:
            self.lan_dist_vec = self.lan_dist_vec.cuda()

    def forward(self, feature):
        #(model_feature, language_feature, data_feature) = feature
        feature, existed_src = feature

        logit = existed_src * self.lan_dist_vec
        return logit
