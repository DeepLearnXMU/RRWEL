#-*- encoding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from utils.scripts import *
import os
from model.Pre_fai_score_xmg import Local_Fai_score
from model.score_global_train import Fai_score
from DataLoader.data_loader_f import *
from data_process import Vocabulary
import torch.nn as nn  
import datetime
import math
import argparse
import os
np.set_printoptions(threshold = np.NaN)

import fnmatch, os
def allFiles(root, patterns = '*', single_level = False, yield_folders = False):
    patterns = patterns.split(';')
    for path, subdirs, files in os.walk(root):
        if yield_folders:
           #add subdirs to the tail of files
           files.extend(subdirs)
        files.sort()
        for name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    yield os.path.join(path, name)
                    break
        #only deal one level of the dir
        if single_level:
            break

def get_testlist(in_f):
	#key::aquaint,ace2004,clueweb12,msnbc,wikipedia,aida_conll_train,aida_conll_testa,aida_conll_testb
	test_dict = {}
	with open(in_f) as f:
		for line in f:
			s = line.split("|||")
			if s[0] in test_dict:
				test_dict[s[0]].append(s[1])
			else:
				test_dict[s[0]]=[]
				test_dict[s[0]].append(s[1])
	return test_dict
	
	
def eval(cnn_score, doc_men, doc_totalmen, data_loader_test, dataflag, Ispretrain, pos_embed_dict, RANDOM_K, LAMDA, MODEL_FLAG):
	starttime = datetime.datetime.now()
	cnn_score.eval()		
	count_true=0
	count_true_global=0
	count_label=0
	total_mentions = []
	actual_mentions = []
	actual_correct = []
	files=[] 
	weight=[]
	word=[]
	print("#######Computing score...#######")
	count_test=0
	
	test_dict = get_testlist("/home/xmg/EL/EL/final_data/docname.txt")
	for k_test in data_loader_test:
		correct_temp = 0
		correct_temp_global = 0
		y_label, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, filename,sfeats=extract_data_from_dataloader(k_test,True)
		if dataflag==1:
			if "testa" not in filename:continue
		if dataflag==2:
			if "testb" not in filename:continue
		if dataflag==3:
			if filename not in test_dict["aquaint"]:continue
		if dataflag==4:
			if filename not in test_dict["ace2004"]:continue
		if dataflag==5:
			if filename not in test_dict["clueweb12"]:continue
		if dataflag==6:
			if filename not in test_dict["msnbc"]:continue
		if dataflag==7:
			if filename not in test_dict["ww"]:continue
		count_test+=1
		y_true = torch.Tensor(y_label.numpy())
		if m!=int(doc_men[filename]):
			print(str(m)+"|||"+str(doc_men[filename]))
			print("erooooooor!!") 
		#s,fai_global_score,fai_local_score=cnn_score(filename, word, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, sfeats, y_true, "False",     4,        0.4,  1         ,pos_embed_dict, "eval")
		#local_score, groupped_res_tmp, gold_res=cnn_score(filename, word, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec,True, sfeats, y_true)
		if Ispretrain:
			local_score, local_score_softmax, local_score_uniform=cnn_score(filename, word, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec,True, sfeats, y_true)
			fai_score = local_score
		else:
			entity_embed_f=open("/home/xmg/EL/EL/yoga_code/data/entity_embed/"+filename+".pkl", 'rb')
			entity_embed_dict = pickle.load(entity_embed_f)
			s, s_global, local_score, fai_local_score_softmax, global_score = cnn_score(filename, word, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, sfeats, y_true, False,  RANDOM_K, LAMDA, MODEL_FLAG, pos_embed_dict,"eval",entity_embed_dict)
			fai_score = global_score
		#local的acc计算
		
		y_forecast_local=[]
		y_local=[]
		count_label+=m
		for i in range(m):
			y_forecast_local.append(np.argmax(fai_score[i].cpu().data.numpy())) 
		for i in range(m):
			if int(y_label[i][int(y_forecast_local[i])])==1:
				count_true+=1
				correct_temp += 1
		y_true=[]
		
		total_mentions.append(int(doc_totalmen[filename]))
		actual_mentions.append(int(doc_men[filename]))
		actual_correct.append(correct_temp)
		#print("total_men:"+str(doc_totalmen[filename])+"|||actual_men:"+str(doc_men[filename])+"|||correct:"+str(correct_temp))
			
		for i in range(m):
			y_true_temp=[]
			for j in range(n):
				if(int(y_label[i][j])==1):
					y_true_temp.append(j)
			y_true.append(y_true_temp)
			
	acc=count_true*1.0/count_label
	ma_precs = [correct / float(actual_mentions[i]) for i, correct in enumerate(actual_correct)]
	ma_recs = [correct / float(total_mentions[i]) for i, correct in enumerate(actual_correct)]
	eval_mi_rec = sum(actual_correct) / float(sum(total_mentions))
	eval_ma_rec = sum(ma_recs) / float(len(ma_recs))
	eval_mi_prec = sum(actual_correct) / float(sum(actual_mentions))
	eval_ma_prec = sum(ma_precs) / float(len(ma_precs))
	eval_mi_f1 = 2 * eval_mi_rec * eval_mi_prec / (eval_mi_rec + eval_mi_prec)
	eval_ma_f1 = 2 * eval_ma_rec * eval_ma_prec / (eval_ma_rec + eval_ma_prec)
			
	endtime = datetime.datetime.now()
	#print("time:"+str((endtime - starttime).total_seconds())+"|||epoch:"+str(epoch_count)+"|||step:"+str(file_count)+"|||loss:"+str(float(loss_sum/BATCH))+"|||acc:"+str(acc)+"|||globalacc:"+str(acc_global)+"|||acc_train:"+str(acc_train)+"|||globalacc_train:"+str(acc_global_train))
	print("data_flag:\n1\ttesta\n2\ttestb\n3\taquaint\n4\tace2004\n5\tclueweb12\n6\tmsnbc\n7\twikipedia")
	print(count_test)
	print("data_flag:"+str(dataflag)+"|||time:"+str((endtime - starttime).total_seconds())+"|||acc:"+str(acc))
	print("eval_mi_prec:"+str(eval_mi_prec)+"|||eval_mi_rec:"+str(eval_mi_rec)+"|||eval_mi_f1:"+str(eval_mi_f1))
	print("eval_ma_prec:"+str(eval_ma_prec)+"|||eval_ma_rec:"+str(eval_ma_rec)+"|||eval_ma_f1:"+str(eval_ma_f1))
	
	

def main():
	arg_parser=argparse.ArgumentParser()
	arg_parser.add_argument('--window_context',default=5)
	arg_parser.add_argument('--window_doc',default=100)
	arg_parser.add_argument('--window_body',default=100)
	arg_parser.add_argument('--window_title',default=17)
	arg_parser.add_argument('--filter_num',default=64)
	arg_parser.add_argument('--filter_window', default=3)
	arg_parser.add_argument('--embedding',default=300)
	arg_parser.add_argument('--cuda_device',required=True, default = 0)
	arg_parser.add_argument('--nohup',required=True, default = "")
	arg_parser.add_argument('--model_loc', required=True)
	arg_parser.add_argument('--lamda', required=True)
	arg_parser.add_argument('--random_k', required=True)
	arg_parser.add_argument('--flag', required=True)
	arg_parser.add_argument('--pre', required=True)
	args=arg_parser.parse_args()

	torch.manual_seed(1)#随机游走种子
	WINDOW_CONTEXT = int(args.window_context) #上下文窗口
	WINDOW_DOC = int(args.window_doc) #文档窗口
	WINDOW_BODY = int(args.window_body) #wiki的body窗口，title大小为12
	WINDOW_TITLE = int(args.window_title) #wiki的title窗口
	FILTER_NUM = int(args.filter_num)
	FILTER_WINDOW = int(args.filter_window)
	EMBEDDING = int(args.embedding)
	RANDOM_K = int(args.random_k)
	LAMDA = float(args.lamda)
	MODEL_FLAG=str(args.flag)
	IsPRE = int(args.pre)
	IsPRE = True if IsPRE==1 else False

	MODEL_LOC=args.model_loc 
	torch.cuda.set_device(int(args.cuda_device))
	
	data_loader_train=get_loader("/home/xmg/EL/EL/final_data/", WINDOW_CONTEXT*2, WINDOW_DOC, WINDOW_BODY,val=False, test=False, shuffle=True, num_workers=0)
	data_loader_test=get_loader("/home/xmg/EL/EL/final_data/", WINDOW_CONTEXT*2, WINDOW_DOC, WINDOW_BODY,val=False, test=True, shuffle=True, num_workers=0)
	data_loader_val=get_loader("/home/xmg/EL/EL/final_data/", WINDOW_CONTEXT*2, WINDOW_DOC, WINDOW_BODY,val=True, test=False, shuffle=True, num_workers=0)
	doc_men, doc_totalmen = get_mentionNum("/home/xmg/EL/EL/final_data/doc_mentionNum.tsv")	
	if IsPRE:
		cnn_score=Local_Fai_score(FILTER_NUM, FILTER_WINDOW, WINDOW_DOC, WINDOW_CONTEXT, WINDOW_TITLE, EMBEDDING, LAMDA)
	else:
		cnn_score = Fai_score(FILTER_NUM, FILTER_WINDOW, WINDOW_DOC, WINDOW_CONTEXT, WINDOW_TITLE, EMBEDDING, LAMDA)
	cnn_score=cnn_score.cuda()
	cnn_score.load_state_dict(torch.load(MODEL_LOC))
	pos_embed_f=open("/home/xmg/EL/EL/yoga_code/data/pos_embed_64.pkl", 'rb')
	pos_embed_dict = pickle.load(pos_embed_f)
	
	for i in range(2,8):
		#eval(cnn_score, doc_men, doc_totalmen, data_loader_test, LAMDA, MODEL_FLAG,i)
		eval(cnn_score, doc_men, doc_totalmen, data_loader_test, i, IsPRE, pos_embed_dict,  RANDOM_K, LAMDA, MODEL_FLAG)
	#eval(cnn_score, doc_men, doc_totalmen, data_loader_test, 6, IsPRE, pos_embed_dict,  RANDOM_K, LAMDA, MODEL_FLAG)
	#eval(cnn_score, doc_men, doc_totalmen, data_loader_val, LAMDA, MODEL_FLAG,1)
	eval(cnn_score, doc_men, doc_totalmen, data_loader_val, 1, IsPRE, pos_embed_dict,  RANDOM_K, LAMDA, MODEL_FLAG)
	#eval(cnn_score, doc_men, doc_totalmen, data_loader_train, 9, IsPRE, pos_embed_dict,  RANDOM_K, LAMDA, MODEL_FLAG)
	#eval(cnn_score, doc_men, doc_totalmen, data_loader_train, LAMDA, MODEL_FLAG,0)
	
if __name__ == "__main__":
	main()