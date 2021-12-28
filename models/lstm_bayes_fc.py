# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
##### Bayesian version #####
from layers.linear_bayesian_layer import BayesianLinear
from blitz.utils import variational_estimator
from layers.attention import Attention, NoQueryAttention
from layers.squeeze_embedding import SqueezeEmbedding


@variational_estimator
class LSTM_BAYES_FC(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM_BAYES_FC, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = BayesianLinear(opt.hidden_dim, opt.polarities_dim, bias=True, freeze = False, 
        prior_sigma_1 = 2, prior_sigma_2 = 2, posterior_rho_init  = -1)
        self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear')

        # self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len.cpu())
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)
        out = self.dense(output)
        return out
