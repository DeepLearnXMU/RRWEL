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
import heapq

np.set_printoptions(threshold=np.NaN)
EMBEDDING_DIM = 300


class Fai_score(nn.Module):
    def __init__(self, filter_num, filter_window, doc, context, title, embedding, lamda):
        super(Fai_score, self).__init__()
        # self.embed = nn.Embedding(629937, EMBEDDING_DIM)
        self.embed = nn.Embedding(5053, EMBEDDING_DIM)
        # for p in self.parameters():p.requires_grad=False
        self.dim_compared_vec = filter_num  # 卷积核个数
        self.num_words_to_use_conv = filter_window  # 卷积窗口大小
        self.sur_num_words_to_use_conv = 2
        self.lamda = lamda
        self.document_length = doc
        self.context_length = context * 2
        self.title_length = title
        self.embedding_len = embedding  # 词向量长度
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
        self.layer_local_combine1 = nn.Linear(629, 1, bias=True)

        self.cos_layer = nn.CosineSimilarity(dim=1, eps=1e-6)

    def cos(self, x, y):
        cos = (x * y).sum() / (math.pow((x * x).sum(), 0.5) * math.pow((y * y).sum(), 0.5))
        return cos

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

    def global_softmax(self, x, e2e_mask, n):
        x = x.cpu().data
        # x = math.e**x
        # sum_x = x.cpu().data
        sum_x = torch.mm(x, e2e_mask)
        for i in range(n):
            sum_x[0][i] = 1 / sum_x[0][i]
        # sum_x=Variable(sum_x,requires_grad=False).cuda()
        x = x * sum_x
        x = Variable(x, requires_grad=False).cuda()
        return x

    def uniform_avg(self, x, n):
        for i in range(n):
            if abs(x[i].sum() - 0) < 1.0e-6: continue
            x[i] = x[i] / x[i].sum()

        return x

    def local_score(self, mention_entity, m, n, mention_vec, context_vec, doc_vec, title_vec, body_vec, sfeatures,
                    pos_embed_dict):
        e_embed = []
        m_embed = []
        pos_embed = []
        mention_vec = self.embed(mention_vec)
        doc_vec = self.embed(doc_vec)
        context_vec = self.embed(context_vec)
        body_vec = self.embed(body_vec)
        title_vec = self.embed(title_vec)

        for i in range(m):
            # 0,1,2,3,4:mention,doc,context,title,body
            ms = self.conv_opra(mention_vec[i], 0)
            md = self.conv_opra(doc_vec, 1)
            mc = self.conv_opra(context_vec[i], 2)
            candi = 0
            candi_list = []
            for j in range(n):
                if int(mention_entity[i][j]) == 1:
                    et = self.conv_opra(title_vec[j], 3)
                    eb = self.conv_opra(body_vec[j], 4)
                    pos_embed.append(pos_embed_dict[i])
                    candi += 1
                    candi_list.append(j)
                    tt = sfeatures[str(i) + '|' + str(j)]
                    x = Variable(torch.Tensor(tt), requires_grad=False).cuda()
                    x = x.unsqueeze(0)
                    f_score = self.layer_sensefeat(x)
                    cos_st = self.cos_layer(ms, et)
                    cos_dt = self.cos_layer(md, et)
                    cos_ct = self.cos_layer(mc, et)
                    cos_sb = self.cos_layer(ms, eb)
                    cos_db = self.cos_layer(md, eb)
                    cos_cb = self.cos_layer(mc, eb)
                    C_score = torch.cat((cos_st, cos_dt, cos_ct, cos_sb, cos_db, cos_cb), 0).unsqueeze(0)  # (1,6)
                    F_local = torch.cat((x, C_score), 1)
                    F_local = self.layer_local_combine1(F_local)
                    if candi == 1:
                        true_output = F_local
                    else:
                        true_output = torch.cat((true_output, F_local), 1)  # (1,n)

            if len(true_output) == 1:
                true_output_softmax = true_output
                true_output_temp = true_output
                true_output_uniform = true_output
            else:
                true_output_softmax = (true_output - torch.mean(true_output)) / torch.std(true_output)
                true_output_temp = (true_output - torch.mean(true_output)) / torch.std(true_output)
                true_output_uniform = (true_output + 1 - true_output.min()) / (true_output.max() - true_output.min())

            true_output_softmax = torch.exp(true_output_softmax - self.sloppyMathLogSum(true_output_softmax))
            true_output_softmax_s = self.softmax_layer(true_output_softmax)
            true_output_uniform = true_output_uniform / true_output_uniform.sum()

            mask_2 = torch.zeros(candi, n)
            for can_ii in range(candi): mask_2[can_ii][candi_list[can_ii]] = 1
            true_output = torch.mm(true_output, Variable(mask_2).cuda())
            true_output_temp = torch.mm(true_output_temp, Variable(mask_2).cuda())
            true_output_uniform = torch.mm(true_output_uniform, Variable(mask_2).cuda())
            true_output_softmax = torch.mm(true_output_softmax, Variable(mask_2).cuda())
            true_output_softmax_s = torch.mm(true_output_softmax_s, Variable(mask_2).cuda())

            if i == 0:
                local_score_temp = true_output_temp
                local_score_softmax = true_output_softmax
                local_score_uniform = true_output_uniform
                local_score_softmax_s = true_output_softmax_s
            else:
                local_score_temp = torch.cat((local_score_temp, true_output_temp), 0)
                local_score_softmax = torch.cat((local_score_softmax, true_output_softmax), 0)
                local_score_softmax_s = torch.cat((local_score_softmax_s, true_output_softmax), 0)
                local_score_uniform = torch.cat((local_score_uniform, true_output_uniform), 0)

        return local_score_temp, local_score_softmax, local_score_uniform, m_embed, e_embed, pos_embed, local_score_softmax_s

    def global_score(self, mention_entity, entity_entity, SR, m, n, local_score_norm, random_k,
                     lamda, flag, pos_embed, entity_embed_dict, fai_local_score_softmax_s):
        flag_entity = int(flag.split(":")[0])
        flag_sr = int(flag.split(":")[1])
        concat_embed = []
        entity_dis = []
        combine_dis = []

        for i in range(n):
            #concat_embed.append(torch.cat((entity_embed_dict[i], pos_embed[i].unsqueeze(0)), 1))
            concat_embed.append(entity_embed_dict[i]+pos_embed[i].unsqueeze(0))
        for i in range(n):
            entity_dis_tmp = []
            combine_dis_tmp = []
            for j in range(n):
                entity_dis_tmp.append(self.cos(entity_embed_dict[i], entity_embed_dict[j]))
                combine_dis_tmp.append(self.cos(concat_embed[i], concat_embed[j]))
            entity_dis.append(entity_dis_tmp)
            combine_dis.append(combine_dis_tmp)
        entity_dis = torch.Tensor(entity_dis)
        combine_dis = torch.Tensor(combine_dis)

        # 每个mention的前n个候选项之间相互传播
        candidate = []
        for i in range(m):
            t_local = local_score_norm[i].cpu().data.numpy().tolist()
            temp_max = list(map(t_local.index, heapq.nlargest(flag_entity, t_local)))
            candidate += temp_max

        e2e_mask = torch.ones(n, n)
        for i in range(n):
            for j in range(n):
                if (int(entity_entity[i][j]) == 0) or (j not in candidate): SR[i][j] = 0
                if (int(entity_entity[i][j]) == 1): e2e_mask[i][j] = 0
                if abs(SR[i][j] - 0.0) < 1.0e-6: continue
                SR[i][j] = SR[i][j] * 10
                SR[i][j] = math.e ** SR[i][j]
        if flag_sr == 1:
            for i in range(n):
                for j in range(n):
                    if abs(SR[i][j] - 0.0) < 1.0e-6:
                        entity_dis[i][j] = 0
            SR = SR + entity_dis

        if flag_sr == 2:
            for i in range(n):
                for j in range(n):
                    if abs(SR[i][j] - 0.0) < 1.0e-6:
                        combine_dis[i][j] = 0
            SR = SR + combine_dis
        if flag_sr == 3:
            for i in range(n):
                for j in range(n):
                    if abs(SR[i][j] - 0.0) < 1.0e-6:
                        combine_dis[i][j] = 0
            SR = combine_dis

        if flag_sr == 4:
            SR = torch.rand(n, n)
        SR = self.uniform_avg(SR, n)

        SR = Variable(SR, requires_grad=True).cuda()
        s = torch.ones(1, m)
        s = Variable(s, requires_grad=False).cuda()
        s = torch.mm(s, local_score_norm)
        fai_global_score = s
        for i in range(random_k):
            fai_global_score = (1 - lamda) * torch.mm(fai_global_score, SR) + lamda * s
        global_score = fai_global_score
        m2e = Variable(mention_entity).cuda()
        for iiii in range(m - 1):
            global_score = torch.cat((global_score, fai_global_score), 0)
        global_score = m2e * global_score
        fai_global_score = self.global_softmax(fai_global_score, e2e_mask, n)
        return s, fai_global_score, global_score

    def forward(self, mention_entity, entity_entity, SR, m, n, mention_vec, context_vec,
                doc_vec, title_vec, body_vec, sfeats, random_k, lamda, flag, pos_embed_dict,
                entity_embed_dict):
        fai_local_score, fai_local_score_softmax, fai_local_score_uniform, m_embed, e_embed, pos_embed, fai_local_score_softmax_s = self.local_score(
            mention_entity, m, n, mention_vec, context_vec, doc_vec, title_vec, body_vec, sfeats, pos_embed_dict)
        s, fai_global_score, global_score = self.global_score(mention_entity, entity_entity, SR, m, n,
                                                              fai_local_score_softmax, random_k, lamda, flag,
                                                              pos_embed, entity_embed_dict, fai_local_score_softmax_s)
        return s, fai_global_score, fai_local_score, fai_local_score_softmax, global_score
