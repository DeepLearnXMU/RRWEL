# -*- encoding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from utils.scripts import *
import os
from tqdm import tqdm
from model.local_score_model import Local_Fai_score
from utils.data_loader_f import *
from data_process import Vocabulary
import torch.nn as nn
import datetime
import math
from Loss.selfloss import SelfLossFunc
import argparse
from torch.nn.init import kaiming_normal, uniform


#model initialization
def weight_init(m):
    with open('../data/pkl/vocab_embedding.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    if isinstance(m, nn.Embedding):
        m.weight.data.copy_(torch.from_numpy(embeddings))
    if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        torch.nn.init.xavier_normal(m.weight.data)
    elif isinstance(m, nn.Linear):
        # m.weight.data.uniform_(0,1)
        kaiming_normal(m.weight)
        m.bias.data.zero_()
        # torch.nn.init.xavier_normal(m.weight.data)
        # torch.nn.init.xavier_normal(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


# Output weight gradient
def weight_grad_stop(m):
    if isinstance(m, nn.Embedding):
        for i in m.parameters():
            i.requires_grad = False


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--epoch', default=30)
    arg_parser.add_argument('--LR', default=0.01)
    arg_parser.add_argument('--window_context', default=5)
    arg_parser.add_argument('--window_doc', default=100)
    arg_parser.add_argument('--window_body', default=100)
    arg_parser.add_argument('--window_title', default=17)
    arg_parser.add_argument('--filter_num', default=128)
    arg_parser.add_argument('--filter_window', default=5)
    arg_parser.add_argument('--embedding', default=300)
    arg_parser.add_argument('--lamda', default=0.01)
    arg_parser.add_argument('--cuda_device', required=True, default=0)
    arg_parser.add_argument('--nohup', required=True, default="")
    arg_parser.add_argument('--batch', default=200)
    arg_parser.add_argument('--weight_decay', required=True, default=1e-5)
    arg_parser.add_argument('--embedding_finetune', default=1)
    arg_parser.add_argument('--local_model_loc', required=True)
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
    BATCH = int(args.batch)
    FINETUNE = bool(int(args.embedding_finetune))
    LOCAL_MODEL_LOC = str(args.local_model_loc)
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
    print('Data root:              ' + str(ROOT))

    print("#######Data loading#######")
    data_loader_train = get_loader(ROOT, WINDOW_CONTEXT * 2, WINDOW_DOC, WINDOW_BODY,
                                   val=False,
                                   test=False, shuffle=True, num_workers=0)
    data_loader_val = get_loader(ROOT, WINDOW_CONTEXT * 2, WINDOW_DOC, WINDOW_BODY, val=True,
                                 test=False, shuffle=True, num_workers=0)
    data_loader_test = get_loader(ROOT, WINDOW_CONTEXT * 2, WINDOW_DOC, WINDOW_BODY, val=False,
                                  test=True, shuffle=True, num_workers=0)
    TrainFileNum = len(data_loader_train)
    print("Train data size:", len(data_loader_train))  # 337
    print("Dev data size:", len(data_loader_val))
    print("Test data size:", len(data_loader_test))
    doc_men, doc_totalmen = get_mentionNum(ROOT+"/dataset/doc_mentionNum.tsv")
    
    print("#######Model Initialization#######")
    cnn_score = Local_Fai_score(FILTER_NUM, FILTER_WINDOW, WINDOW_DOC, WINDOW_CONTEXT, WINDOW_TITLE, EMBEDDING, LAMDA)
    cnn_score = cnn_score.cuda()
    cnn_score.apply(weight_init)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_score.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(cnn_score)

    print("#######Training...#######")
    epoch_count = 0
    starttime = datetime.datetime.now()
    last_acc = 0
    word = []
    for epoch in range(EPOCH):
        epoch_count += 1
        print("****************epoch " + str(epoch_count) + "...****************")
        file_count = 0
        loss_sum = 0
        for k in tqdm(data_loader_train):
            file_count += 1
            y_label, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, filename, sfeats = extract_data_from_dataloader(
                k, FINETUNE)
            cnn_score.train()
            y_true = torch.Tensor(y_label.numpy())
            local_score, fai_local_score_softmax, fai_local_score_uniform = cnn_score(mention_entity, m, n, mention_vec,
                                                                                      context_vec, doc_vec, title_vec,
                                                                                      body_vec, sfeats)
            y_true_index = []
            for y_t_i in range(m):
                for y_t_j in range(n):
                    if int(y_true[y_t_i][y_t_j]) == 1:
                        y_true_index.append(y_t_j)
            # print(len(y_true_index))
            y_true_index = Variable(torch.LongTensor(y_true_index)).cuda()
            loss = loss_function(local_score, y_true_index)
            loss_sum += loss.cpu().data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # testing
            if file_count % BATCH == 0 or file_count == TrainFileNum:
                # print("eval")
                cnn_score.eval()
                count_true = 0
                count_label = 0
                total_mentions = []
                actual_mentions = []
                actual_correct = []
                endtime = datetime.datetime.now()
                print("time:" + str((endtime - starttime).total_seconds()))
                print("#######Computing score...#######")
                test_file_c = 0
                for k_test in data_loader_val:
                    correct_temp = 0
                    y_label, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, filename, sfeats = extract_data_from_dataloader(
                        k_test, FINETUNE)
                    if "testa " not in filename: continue
                    test_file_c += 1
                    y_true = torch.Tensor(y_label.numpy())
                    if m != int(doc_men[filename]):
                        print(str(m) + "|||" + str(doc_men[filename]))
                        print("erooooooor!!")
                    fai_local_score_temp, fai_local_score_softmax, fai_local_score_uniform = cnn_score(mention_entity,
                                                                                                       m, n, 
                                                                                                       mention_vec,
                                                                                                       context_vec,
                                                                                                       doc_vec,
                                                                                                       title_vec,
                                                                                                       body_vec, sfeats)
                                                                                                       
                    fai_score = fai_local_score_temp.cpu().data
                    y_forecast = []
                    y_local = []
                    count_label += m
                    for i in range(m):
                        y_forecast.append(np.argmax(fai_score[i].numpy()))
                    for i in range(m):
                        if int(y_label[i][int(y_forecast[i])]) == 1:
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
                if eval_mi_f1 > last_acc:
                    model_f = str(eval_mi_f1)
                    model_f = model_f[model_f.index("."):model_f.index(".") + 4]
                    model_f = LOCAL_MODEL_LOC + model_f + ".pkl"
                    torch.save(cnn_score.state_dict(), model_f)
                    last_acc = eval_mi_f1

                endtime = datetime.datetime.now()
                print("time:" + str((endtime - starttime).total_seconds()) + "|||epoch:" + str(
                    epoch_count) + "|||step:" + str(file_count) + "|||loss:" + str(float(loss_sum)) + "|||acc:" + str(
                    acc))
                print(
                    "eval_mi_prec:" + str(eval_mi_prec) + "|||eval_mi_rec:" + str(eval_mi_rec) + "|||eval_mi_f1:" + str(
                        eval_mi_f1))
                print(
                    "eval_ma_prec:" + str(eval_ma_prec) + "|||eval_ma_rec:" + str(eval_ma_rec) + "|||eval_ma_f1:" + str(
                        eval_ma_f1))
                count_true = 0
                count_label = 0
                total_mentions = []
                actual_mentions = []
                actual_correct = []
                endtime = datetime.datetime.now()
                print("time:" + str((endtime - starttime).total_seconds()))
                for i in range(2, 8):
                    eval(ROOT, cnn_score, doc_men, doc_totalmen, data_loader_test, i, True, None, 0, 0, 0)
print('over')
