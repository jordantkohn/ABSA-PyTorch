# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import logging
import argparse
import math
import os
import sys
import random
import numpy

from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset, ABSACustom
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT
from models import LSTM_BAYES_FC, LSTM_BAYES_RNN
from models import BERT_BAYES_SPC
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        # self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        # self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        self.customset = ABSACustom(opt.dataset_file['custom'], tokenizer)

        # assert 0 <= opt.valset_ratio < 1
        # if opt.valset_ratio > 0:
        #     valset_len = int(len(self.trainset) * opt.valset_ratio)
        #     self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        # else:
        #     self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        # self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _custom_confidence(self, custom_data_loader, n_sample=25):
        n_correct, n_total = 0, 0
        n_correct_not_conf, n_total_not_conf = 0, 0
        n_correct_conf, n_total_conf = 0, 0
        t_targets_all, t_outputs_all = None, None
        texts, confs, preds, labels = [], [], [], []

        # switch model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for i_batch, t_batch in enumerate(custom_data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                texts.append(t_batch['text'])
                labels.append(t_targets)

                t_outputs_sample = torch.stack([self.model(t_inputs) for i in range(n_sample)])
                t_preds_sample = t_outputs_sample.argmax(axis=2)
                
                # calculate confidence score
                confidence = torch.Tensor([torch.bincount(i).max() for i in t_preds_sample.T]).T
                confidence = confidence/n_sample
                confs.append(confidence)

                # all predicttions
                t_preds = torch.stack([torch.argmax(torch.bincount(i)) for i in t_preds_sample.T])
                preds.append(t_preds)

                n_correct += (t_preds == t_targets).sum().item()
                n_total += len(t_preds)

                # visualization
                chart_file = '/content/drive/MyDrive/DLFinal Project/notebooks/ABSA-PyTorch-master/graphs/{}-{}-{}_{}.png'.format(
                  self.opt.model_name, self.opt.dataset, strftime("%y%m%d-%H%M", localtime()), i_batch)

                # this only works if batch size = 1
                confidence_by_class = torch.bincount(t_preds_sample.T[0]).cpu().numpy()
                
                plt.figure(figsize=(8,6))
                plt.bar(np.array(range(len(confidence_by_class)))-1, confidence_by_class)
                plt.xticks(np.array(range(len(confidence_by_class)))-1)
                plt.title('number of votes by class for custom example\n'+t_batch['text'][0] + \
                  '\n'+'label:' + str(int(t_targets.item())-1) + '    aspect:' + t_batch['aspect'][0])
                plt.xlabel('sentiment')
                plt.savefig(chart_file)

    

        # acc = n_correct / n_total
        # acc_conf = n_correct_conf / n_total_conf
        # percent_conf = n_total_conf / n_total

        return texts, confs, preds, labels


    def run(self):
        custom_data_loader = DataLoader(dataset=self.customset, batch_size=1, shuffle=False)

        self._reset_params()

        saved_model_path = 'state_dict/{}'.format(self.opt.model_statedict)
        self.model.load_state_dict(torch.load(saved_model_path))
        
        texts, confs, preds, labels = self._custom_confidence(custom_data_loader)
        for text, conf, pred, label in zip(texts, confs, preds, labels):
          logger.info('>> {} \n\tconfidence: {:.4f} \n\tpred: {} \n\tlabel: {} '.format(
            text[0], conf.item(), pred.item(), label.item()))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--model_name', default='bert_bayes_spc', type=str)
    # parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    parser.add_argument('--model_statedict', default='lstm_bayes_fc_restaurant_val_acc_0.7607', type=str)
    parser.add_argument('--custom_text', type=str)
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'lstm': LSTM,
        'lstm_bayes_fc': LSTM_BAYES_FC,
        'lstm_bayes_rnn': LSTM_BAYES_RNN,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        'bert_bayes_spc': BERT_BAYES_SPC,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        },
        'jordan_custom': {
            'custom': './datasets/custom/jordan_custom.txt'
        },
        'youssef_custom': {
            'custom': './datasets/custom/youssef_custom.txt'
        },
        'amro_custom': {
            'custom': './datasets/custom/amro_custom.txt'
        }
        
    }
    input_colses = {
        'lstm': ['text_indices'],
        'lstm_bayes_fc': ['text_indices', 'aspect_indices'],
        'lstm_bayes_rnn':['text_indices', 'aspect_indices'],
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
        'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
        'aoa': ['text_indices', 'aspect_indices'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'bert_bayes_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    # log_file = '/content/drive/MyDrive/DLFinal Project/notebooks/ABSA-PyTorch-master/logs/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    # logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
