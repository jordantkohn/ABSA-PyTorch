import torch
import torch.nn as nn
from layers.linear_bayesian_layer import BayesianLinear
from blitz.utils import variational_estimator

##### Bayesian version #####

@variational_estimator
class BERT_BAYES_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_BAYES_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = BayesianLinear(opt.bert_dim, opt.polarities_dim, bias=True, freeze = False,
                          # prior_sigma_1 = 0.1, prior_sigma_2 = 0.4, prior_pi = 1)
                          prior_sigma_1 = 2, prior_sigma_2 = 0.5, prior_pi = 1)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, return_dict=False)
        # x_len = torch.sum(text_bert_indices != 0, dim=-1)
        # pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits