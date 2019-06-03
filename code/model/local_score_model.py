# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import re
import datetime
import gensim
import math
import gc
import pickle

np.set_printoptions(threshold=np.NaN)
EMBEDDING_DIM = 300


class Local_Fai_score(nn.Module):
    def __init__(self, filter_num, filter_window, doc, context, title, embedding, lamda):
        super(Local_Fai_score, self).__init__()
        # self.embed = nn.Embedding(629937, EMBEDDING_DIM)
        self.embed = nn.Embedding(5053, EMBEDDING_DIM)
        self.dim_compared_vec = filter_num  # 卷积核个数
        self.num_words_to_use_conv = filter_window  # 卷积窗口大小
        self.sur_num_words_to_use_conv = 2
        self.lamda = lamda
        self.document_length = doc
        self.context_length = context * 2
        self.title_length = title
        self.embedding_len = embedding  # 词向量长度
        # self.word_embedding=nn.Embedding()
        self.surface_length = 10
        self.num_indicator_features = 623

        self.relu_layer = nn.ReLU(inplace=True)

        # self.layer_ms = nn.Sequential(
        # 	nn.Linear(self.embedding_len,self.dim_compared_vec),
        # 	nn.LeakyReLU()
        # 	)
        self.conv_ms = nn.Conv2d(
            in_channels=1,  # input_height
            out_channels=self.dim_compared_vec,  # n_filters
            kernel_size=[self.sur_num_words_to_use_conv, self.embedding_len]  # filter_size
        )
        self.avg_ms = nn.AvgPool1d(kernel_size=self.surface_length - self.sur_num_words_to_use_conv + 1)

        self.conv_mc = nn.Conv2d(
            in_channels=1,  # input_height
            out_channels=self.dim_compared_vec,  # n_filters
            kernel_size=[self.num_words_to_use_conv, self.embedding_len]  # filter_size
        )  # (1,150,6,1)
        self.avg_mc = nn.AvgPool1d(kernel_size=self.context_length - self.num_words_to_use_conv + 1)

        self.conv_md = nn.Conv2d(
            in_channels=1,  # input_height
            out_channels=self.dim_compared_vec,  # n_filters
            kernel_size=[self.num_words_to_use_conv, self.embedding_len]  # filter_size
        )
        self.avg_md = nn.AvgPool1d(kernel_size=self.document_length - self.num_words_to_use_conv + 1)

        self.conv_et = nn.Conv2d(
            in_channels=1,  # input_height
            out_channels=self.dim_compared_vec,  # n_filters
            kernel_size=[self.num_words_to_use_conv, self.embedding_len]  # filter_size
        )
        self.avg_et = nn.AvgPool1d(kernel_size=self.title_length - self.num_words_to_use_conv + 1)

        self.conv_eb = nn.Conv2d(
            in_channels=1,  # input_height
            out_channels=self.dim_compared_vec,  # n_filters
            kernel_size=[self.num_words_to_use_conv, self.embedding_len]  # filter_size
        )
        self.avg_eb = nn.AvgPool1d(kernel_size=self.document_length - self.num_words_to_use_conv + 1)

        self.softmax_layer = nn.Softmax(dim=1)

        self.layer_local = nn.Linear(6, 1, bias=True)
        self.layer_sensefeat = nn.Linear(self.num_indicator_features, 1, bias=True)
        self.layer_local_combine1 = nn.Linear(self.num_indicator_features+6, 1, bias=True)

        self.cos_layer = nn.CosineSimilarity(dim=1, eps=1e-6)

    def conv_opra(self, x, flag):
        # 0,1,2,3,4:mention,doc,context,title,body
        x = x.unsqueeze(0).unsqueeze(1)
        if flag == 0:
            x = self.avg_ms(self.relu_layer(self.conv_ms(x).squeeze(3))).squeeze(2)
        if flag == 1:
            x = self.avg_md(self.relu_layer(self.conv_md(x).squeeze(3))).squeeze(2)
        if flag == 2:
            x = self.avg_mc(self.relu_layer(self.conv_mc(x).squeeze(3))).squeeze(2)
        if flag == 3:
            x = self.avg_et(self.relu_layer(self.conv_et(x).squeeze(3))).squeeze(2)
        if flag == 4:
            x = self.avg_eb(self.relu_layer(self.conv_eb(x).squeeze(3))).squeeze(2)
        return x

    def sloppyMathLogSum(self, vals):
        m = float(vals.max().cpu().data)
        r = torch.log(torch.exp(vals - m).sum())
        r = r + m
        return r

    def local_score(self, mention_entity, m, n, mention_vec, context_vec, doc_vec, title_vec, body_vec, sfeatures):
        mention_vec = self.embed(mention_vec)
        doc_vec = self.embed(doc_vec)
        context_vec = self.embed(context_vec)
        body_vec = self.embed(body_vec)
        title_vec = self.embed(title_vec)
        for i in range(m):
            ms = self.conv_opra(mention_vec[i], 0)
            md = self.conv_opra(doc_vec, 1)
            mc = self.conv_opra(context_vec[i], 2)
            candi = 0
            candi_list = []
            for j in range(n):
                if int(mention_entity[i][j]) == 1:
                    et = self.conv_opra(title_vec[j], 3)
                    eb = self.conv_opra(body_vec[j], 4)
                    candi += 1
                    candi_list.append(j)
                    tt = sfeatures[str(i) + '|' + str(j)]
                    x = Variable(torch.Tensor(tt), requires_grad=False).cuda()  # (torch.Floattensor of size 623)
                    x = x.unsqueeze(0)
                    cos_st = self.cos_layer(ms, et)
                    cos_dt = self.cos_layer(md, et)
                    cos_ct = self.cos_layer(mc, et)
                    cos_sb = self.cos_layer(ms, eb)
                    cos_db = self.cos_layer(md, eb)
                    cos_cb = self.cos_layer(mc, eb)
                    C_score = torch.cat((cos_st, cos_dt, cos_ct, cos_sb, cos_db, cos_cb), 0).unsqueeze(0)
                    F_local = torch.cat((x, C_score), 1)
                    F_local = self.layer_local_combine1(F_local)
                    if candi == 1:
                        true_output = F_local
                    else:
                        true_output = torch.cat((true_output, F_local), 1)
            if len(true_output) == 1:
                true_output_softmax = true_output
                true_output_uniform = true_output
            else:
                true_output_softmax = (true_output - torch.mean(true_output)) / torch.std(true_output)
                true_output = (true_output - torch.mean(true_output)) / torch.std(true_output)
                true_output_uniform = (true_output + 1 - true_output.min()) / (true_output.max() - true_output.min())

            true_output_softmax = torch.exp(true_output_softmax - self.sloppyMathLogSum(true_output_softmax))
            true_output_uniform = true_output_uniform / true_output_uniform.sum()
            mask_2 = torch.zeros(candi, n)
            for can_ii in range(candi): mask_2[can_ii][candi_list[can_ii]] = 1
            true_output = torch.mm(true_output, Variable(mask_2).cuda())
            true_output_uniform = torch.mm(true_output_uniform, Variable(mask_2).cuda())
            true_output_softmax = torch.mm(true_output_softmax, Variable(mask_2).cuda())

            if i == 0:
                local_score = true_output
                local_score_softmax = true_output_softmax
                local_score_uniform = true_output_uniform
            else:
                local_score = torch.cat((local_score, true_output), 0)
                local_score_softmax = torch.cat((local_score_softmax, true_output_softmax), 0)
                local_score_uniform = torch.cat((local_score_uniform, true_output_uniform), 0)
        return local_score, local_score_softmax, local_score_uniform

    def forward(self, mention_entity, m, n, mention_vec, context_vec,
                doc_vec, title_vec, body_vec, sfeatures):
        local_score, local_score_softmax, local_score_uniform = self.local_score(mention_entity, m, n,
                                                                                                  mention_vec,
                                                                                                  context_vec, doc_vec,
                                                                                                  title_vec, body_vec,
                                                                                                  sfeatures)
        return local_score, local_score_softmax, local_score_uniform
