# -*- encoding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from utils.scripts import *
import os
from tqdm import tqdm
from model.global_score_model import Fai_score
from model.local_score_model import Local_Fai_score
from utils.data_loader_f import *
from data_process import Vocabulary
import torch.nn as nn
import datetime
import math
from Loss.sparseLoss import SparseLossFunc
from Loss.GlobalLoss import GlobalLossFunc
import argparse

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--epoch', default=3)
    arg_parser.add_argument('--LR', default=0.01)
    arg_parser.add_argument('--window_context', default=5)
    arg_parser.add_argument('--window_doc', default=100)
    arg_parser.add_argument('--window_body', default=100)
    arg_parser.add_argument('--window_title', default=17)
    arg_parser.add_argument('--filter_num', default=64)
    arg_parser.add_argument('--filter_window', default=3)
    arg_parser.add_argument('--embedding', default=300)
    arg_parser.add_argument('--cuda_device', required=True, default=0)
    arg_parser.add_argument('--nohup', required=True, default="")
    arg_parser.add_argument('--batch', default=200)
    arg_parser.add_argument('--weight_decay', required=True, default=1e-5)
    arg_parser.add_argument('--local_model_loc', required=True)
    arg_parser.add_argument('--global_model_loc', required=True)
    arg_parser.add_argument('--random_k', required=True, default=3)
    arg_parser.add_argument('--lamda', required=True)
    arg_parser.add_argument('--gama', required=True)
    arg_parser.add_argument('--flag', required=True)
    arg_parser.add_argument('--embedding_finetune', default=1)
    arg_parser.add_argument('--data_root', default="../data")

    args = arg_parser.parse_args()

    torch.manual_seed(1)
    EPOCH = int(args.epoch)
    LR = float(args.LR)
    WEIGHT_DECAY = float(args.weight_decay)
    WINDOW_CONTEXT = int(args.window_context)
    WINDOW_DOC = int(args.window_doc)
    WINDOW_BODY = int(args.window_body)
    WINDOW_TITLE = int(args.window_title)
    FILTER_NUM = int(args.filter_num)
    FILTER_WINDOW = int(args.filter_window)
    EMBEDDING = int(args.embedding)
    LAMDA = float(args.lamda)
    GAMA = float(args.gama)
    BATCH = int(args.batch)
    MODEL_FLAG = args.flag
    FINETUNE = bool(int(args.embedding_finetune))
    LOCAL_MODEL_LOC = args.local_model_loc
    GLOBAL_MODEL_LOC = args.global_model_loc
    RANDOM_K = int(args.random_k)
    ROOT = str(args.data_root)
    torch.cuda.set_device(int(args.cuda_device))
    np.set_printoptions(threshold=np.NaN)

    print('Epoch num:              ' + str(EPOCH))
    print('Learning rate:          ' + str(LR))
    print('Weight decay:           ' + str(WEIGHT_DECAY))
    print('Context window:         ' + str(WINDOW_CONTEXT))
    print('Document window:        ' + str(WINDOW_DOC))
    print('Title window:           ' + str(WINDOW_TITLE))
    print('Body window:            ' + str(WINDOW_BODY))
    print('Filter number:          ' + str(FILTER_NUM))
    print('Filter window:          ' + str(FILTER_WINDOW))
    print('Embedding dim:          ' + str(EMBEDDING))
    print('Lambda:                 ' + str(LAMDA))
    print('Is finetune embedding:  ' + str(FINETUNE))
    print('Gama:                   ' + str(GAMA))
    print('Data root:              ' + str(ROOT))

    print("#######Data loading#######")
    data_loader_train = get_loader(ROOT, WINDOW_CONTEXT * 2, WINDOW_DOC, WINDOW_BODY, val=False,
                                   test=False, shuffle=True, num_workers=0)
    data_loader_val = get_loader(ROOT, WINDOW_CONTEXT * 2, WINDOW_DOC, WINDOW_BODY, val=True,
                                 test=False, shuffle=True, num_workers=0)
    data_loader_test = get_loader(ROOT, WINDOW_CONTEXT * 2, WINDOW_DOC, WINDOW_BODY, val=False,
                                  test=True, shuffle=True, num_workers=0)
    TrainFileNum = len(data_loader_train)
    print("Train data size:", len(data_loader_train))  # 337
    print("Dev data size:", len(data_loader_val))
    print("Test data size:", len(data_loader_test))
    ww_file = open(ROOT + "/dataset/wikipedia/wikipedia-name2bracket.tsv")
    ww_list = []
    for line in ww_file:
        ww_list.append(line.split("\t")[0])
    pos_embed_f = open(ROOT + "/pkl/pos_embed_128.pkl", 'rb')
    pos_embed_dict = pickle.load(pos_embed_f)
    doc_men, doc_totalmen = get_mentionNum(ROOT + "/dataset/doc_mentionNum.tsv")

    print("#######Model Initialization#######")
    cnn_score = Fai_score(FILTER_NUM, FILTER_WINDOW, WINDOW_DOC, WINDOW_CONTEXT, WINDOW_TITLE, EMBEDDING, LAMDA)
    cnn_score = cnn_score.cuda()
    pretrain = Local_Fai_score(FILTER_NUM, FILTER_WINDOW, WINDOW_DOC, WINDOW_CONTEXT, WINDOW_TITLE, EMBEDDING, LAMDA)
    pretrain.load_state_dict(torch.load(LOCAL_MODEL_LOC, map_location={'cuda:2': 'cuda:0'}))
    pretrained_dict = pretrain.state_dict()
    model_dict = cnn_score.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    model_dict.update(pretrained_dict)
    cnn_score.load_state_dict(model_dict)
    loss_function = GlobalLossFunc().cuda()
    optimizer = torch.optim.Adam(cnn_score.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(cnn_score)

    epoch_count = 0
    starttime = datetime.datetime.now()
    best_f1 = 0
    best_f2 = 0
    best_f6 = 0

    print("#######Training...#######")
    for epoch in range(EPOCH):
        epoch_count += 1
        print("****************epoch " + str(epoch_count) + "...****************")
        file_count = 0
        loss_sum = 0
        for k in tqdm(data_loader_train):
            cnn_score.train()
            file_count += 1
            y_label, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, filename, sfeats = extract_data_from_dataloader(
                k, FINETUNE)
            entity_embed_f = open(ROOT + "/pkl/entity_embed/" + filename + ".pkl", 'rb')
            entity_embed_dict = pickle.load(entity_embed_f)
            y_true = torch.Tensor(y_label.numpy())
            s, s_global, local_score, fai_local_score_softmax, global_score = cnn_score(mention_entity, entity_entity,
                                                                                        SR, m, n,
                                                                                        mention_vec,
                                                                                        context_vec, doc_vec, title_vec,
                                                                                        body_vec, sfeats, RANDOM_K,
                                                                                        LAMDA,
                                                                                        MODEL_FLAG, pos_embed_dict,
                                                                                        entity_embed_dict)

            y_true_index = []
            for y_t_i in range(m):
                for y_t_j in range(n):
                    if int(y_true[y_t_i][y_t_j]) == 1:
                        y_true_index.append(y_t_j)
            y_true_index = Variable(torch.LongTensor(y_true_index), requires_grad=False).cuda()
            loss = loss_function(s, s_global, local_score, global_score, y_true_index, GAMA, MODEL_FLAG)
            loss_sum += loss.cpu().data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # testing
            if file_count % 500 == 0 or file_count == TrainFileNum:
                print("#######Evaluate val#######")
                count_true = 0
                count_label = 0
                total_mentions = []
                actual_mentions = []
                actual_correct = []
                endtime = datetime.datetime.now()
                print("time:" + str((endtime - starttime).total_seconds()))
                cnn_score.eval()
                print("#######Evaluate val#######")
                count_test = 0
                for k_test in data_loader_val:
                    correct_temp = 0
                    count_test += 1
                    y_label, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, filename, sfeats = extract_data_from_dataloader(
                        k_test, FINETUNE)
                    y_true = torch.Tensor(y_label.numpy())
                    entity_embed_f = open(ROOT + "/pkl/entity_embed/" + filename + ".pkl", 'rb')
                    entity_embed_dict = pickle.load(entity_embed_f)
                    if m != int(doc_men[filename]):
                        print(str(m) + "|||" + str(doc_men[filename]))
                        print("erooooooor!!")
                    s, s_global, fai_score, fai_local_score_softmax, global_score = cnn_score(mention_entity,
                                                                                              entity_entity, SR, m, n,
                                                                                              mention_vec,
                                                                                              context_vec, doc_vec,
                                                                                              title_vec, body_vec,
                                                                                              sfeats, RANDOM_K,
                                                                                              LAMDA, MODEL_FLAG,
                                                                                              pos_embed_dict,
                                                                                              entity_embed_dict)

                    y_forecast_local = []
                    y_local = []
                    count_label += m
                    for i in range(m):
                        y_forecast_local.append(np.argmax(global_score[i].cpu().data.numpy()))
                    for i in range(m):
                        if int(y_label[i][int(y_forecast_local[i])]) == 1:
                            count_true += 1
                            correct_temp += 1
                    y_true = []

                    total_mentions.append(int(doc_totalmen[filename]))
                    actual_mentions.append(int(doc_men[filename]))
                    actual_correct.append(correct_temp)

                print("total_men:" + str(doc_totalmen[filename]) + "|||actual_men:" + str(
                    doc_men[filename]) + "|||correct:" + str(correct_temp))

                for i in range(m):
                    y_true_temp = []
                    for j in range(n):
                        if (int(y_label[i][j]) == 1):
                            y_true_temp.append(j)
                    y_true.append(y_true_temp)
                acc, eval_mi_prec, eval_ma_prec, eval_mi_rec, eval_ma_rec, eval_mi_f1, eval_ma_f1 = Fmeasure(count_true,
                                                                                                             count_label,
                                                                                                             actual_mentions,
                                                                                                             total_mentions,
                                                                                                             actual_correct)
                endtime = datetime.datetime.now()
                print("time:" + str((endtime - starttime).total_seconds()) + "|||epoch:" + str(
                    epoch_count) + "|||step:" + str(file_count) + "|||loss:" + str(float(loss_sum)) + "|||acc:" + str(
                    acc))
                print(
                        "eval_mi_prec:" + str(eval_mi_prec) + "|||eval_mi_rec:" + str(
                    eval_mi_rec) + "|||eval_mi_f1:" + str(
                    eval_mi_f1))
                print(
                        "eval_ma_prec:" + str(eval_ma_prec) + "|||eval_ma_rec:" + str(
                    eval_ma_rec) + "|||eval_ma_f1:" + str(
                    eval_ma_f1))
                """
                print("#######Evaluate test#######")
                for i in range(2, 8):
                    acc, eval_mi_prec, eval_ma_prec, eval_mi_rec, eval_ma_rec, eval_mi_f1, eval_ma_f1 = eval(ROOT,
                                                                                                             cnn_score,
                                                                                                             doc_men,
                                                                                                             doc_totalmen,
                                                                                                             data_loader_test,
                                                                                                             i, False,
                                                                                                             pos_embed_dict,
                                                                                                             RANDOM_K,
                                                                                                             LAMDA,
                                                                                                             MODEL_FLAG)
                    if i == 2 and eval_mi_f1 > best_f2:
                        model_f = str(eval_mi_f1)
                        model_f = model_f[model_f.index("."):model_f.index(".") + 4]
                        model_f = GLOBAL_MODEL_LOC + model_f + "_testb.pkl"
                        torch.save(cnn_score.state_dict(), model_f)
                        best_f2 = eval_mi_f1
                    if i == 6 and eval_mi_f1 > best_f6:
                        model_f = str(eval_mi_f1)
                        model_f = model_f[model_f.index("."):model_f.index(".") + 4]
                        model_f = GLOBAL_MODEL_LOC + model_f + "_test.pkl"
                        torch.save(cnn_score.state_dict(), model_f)
                        best_f6 = eval_mi_f1
                """







print('over')
