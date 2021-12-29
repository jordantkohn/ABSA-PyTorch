import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer

from layers.dynamic_rnn import DynamicLSTM


class LSTM_BERT_EMBED(nn.Module):
    def __init__(self, bert, opt):
        # super(LSTM, self).__init__()
        super().__init__()
        # self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bert = bert
        # self.dropout = nn.Dropout(opt.dropout)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        self.lstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, return_dict=False)
        x_len = torch.sum(text_bert_indices != 0, dim=-1)
        out, (h, c) = self.lstm(pooled_output, x_len.cpu())
        out = self.dense(h[0])
        return out
