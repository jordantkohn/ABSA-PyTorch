# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
##### Bayesian version #####
from layers.lstm_bayesian_layer import BayesianLSTM
from layers.gru_bayesian_layer import BayesianGRU

from blitz.utils import variational_estimator
from layers.linear_bayesian_layer import BayesianLinear

from layers.attention import Attention, NoQueryAttention
from layers.squeeze_embedding import SqueezeEmbedding

@variational_estimator
class GRU_BAYES(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(GRU_BAYES, self).__init__()
        self.lstm = BayesianGRU(opt.embed_dim*2, opt.hidden_dim, bias=True, freeze = False,
                prior_sigma_1 = 0.7,
                prior_sigma_2 = 0.3,
                posterior_rho_init=-3,
                sharpen=True)
                #  prior_pi = 1,
                #  posterior_mu_init = 0,
                #  posterior_rho_init = -6.0,
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        # self.dense = BayesianLinear(opt.hidden_dim, opt.polarities_dim, bias=True, freeze = False, 
                          # prior_sigma_1 = 10, prior_sigma_2 = 10, posterior_rho_init  = 5 )
        self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear')

        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

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

        h, _ = self.lstm(x)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)
        out = self.dense(output)
        return out