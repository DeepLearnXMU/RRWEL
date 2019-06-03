#-*- encoding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from scripts import *
import os
from yoga_code.model.Pre_fai_score_xmg import Fai_score
from yoga_code.DataLoader.data_loader_f import *
from data_process import Vocabulary
import torch.nn as nn  
import datetime
import math
import argparse
np.set_printoptions(threshold = np.NaN)
	
def eval(cnn_score, doc_men, doc_totalmen, ww_list, data_loader_test):
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
	
	for k_test in data_loader_test:
		correct_temp = 0
		correct_temp_global = 0
		count_test+=1
		y_label, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, filename,sfeats=extract_data_from_dataloader(k_test,files,weight,FINETUNE)
		if filename not in ww_list:continue
		print(filename)
		y_true = torch.Tensor(y_label.numpy())
		#if filename!="Larry_Worrell" and filename!="Julius_Scriver" and filename!="South_East_Lancashire_(UK_Parliament_constituency)":continue
		if m!=int(doc_men[filename]):
			print(str(m)+"|||"+str(doc_men[filename]))
			print("erooooooor!!")
		s,global_score,fai_score=cnn_score(filename, word, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, sfeats, y_true, IS_PRETRAIN, RANDOM_K, LAMDA,MODEL_FLAG)

		#local的acc计算
		y_forecast_local=[]
		y_forecast_global=[]
		y_local=[]
		count_label+=m
		for i in range(m):
			y_forecast_local.append(np.argmax(fai_score[i].cpu().data.numpy())) 
			y_forecast_global.append(np.argmax(global_score[i].cpu().data.numpy())) 
		for i in range(m):
			if int(y_label[i][int(y_forecast_local[i])])==1:
				count_true+=1
				correct_temp += 1
			if int(y_label[i][int(y_forecast_global[i])])==1:
				count_true_global+=1
				correct_temp_global += 1
		y_true=[]
				
		total_mentions.append(int(doc_totalmen[filename]))
		actual_mentions.append(int(doc_men[filename]))
		actual_correct.append(correct_temp)
		print("total_men:"+str(doc_totalmen[filename])+"|||actual_men:"+str(doc_men[filename])+"|||correct:"+str(correct_temp))
		print("total_men:"+str(doc_totalmen[filename])+"|||actual_men:"+str(doc_men[filename])+"|||correct_global:"+str(correct_temp_global))
				
		for i in range(m):
			y_true_temp=[]
			for j in range(n):
				if(int(y_label[i][j])==1):
					y_true_temp.append(j)
			y_true.append(y_true_temp)
						
		if n<50:
			print(fai_score.cpu().data.numpy())
			print(y_forecast_local)
			print(global_score.cpu().data.numpy())
			print(y_forecast_global)
			print(y_true)
					
	acc=count_true*1.0/count_label
	acc_global=count_true_global*1.0/count_label
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
	print("time:"+str((endtime - starttime).total_seconds())+"|||epoch:"+str(epoch_count)+"|||step:"+str(file_count)+"|||loss:"+str(float(loss_sum/BATCH))+"|||acc:"+str(acc)+"|||globalacc:"+str(acc_global))
	print("eval_mi_prec:"+str(eval_mi_prec)+"|||eval_mi_rec:"+str(eval_mi_rec)+"|||eval_mi_f1:"+str(eval_mi_f1))
	print("eval_ma_prec:"+str(eval_ma_prec)+"|||eval_ma_rec:"+str(eval_ma_rec)+"|||eval_ma_f1:"+str(eval_ma_f1))


def main():
	arg_parser=argparse.ArgumentParser()
	arg_parser.add_argument('--window_context',default=5)
	arg_parser.add_argument('--window_doc',default=100)
	arg_parser.add_argument('--window_body',default=100)
	arg_parser.add_argument('--window_title',default=24)
	arg_parser.add_argument('--filter_num',default=64)
	arg_parser.add_argument('--filter_window', default=3)
	arg_parser.add_argument('--embedding',default=300)
	arg_parser.add_argument('--cuda_device',required=True, default = 0)
	arg_parser.add_argument('--nohup',required=True, default = "")
	arg_parser.add_argument('--model_loc', required=True)
	arg_parser.add_argument('--lamda', default = 0.4)
	arg_parser.add_argument('--flag', default=1)
	args=arg_parser.parse_args()

	torch.manual_seed(1)#随机游走种子
	WINDOW_CONTEXT = int(args.window_context) #上下文窗口
	WINDOW_DOC = int(args.window_doc) #文档窗口
	WINDOW_BODY = int(args.window_body) #wiki的body窗口，title大小为12
	WINDOW_TITLE = int(args.window_title) #wiki的title窗口
	FILTER_NUM = int(args.filter_num)
	FILTER_WINDOW = int(args.filter_window)
	EMBEDDING = int(args.embedding)
	LAMDA = float(args.lamda)
	MODEL_FLAG=int(args.flag)

	MODEL_LOC=args.model_loc 
	torch.cuda.set_device(int(args.cuda_device))
	
	data_loader_test=get_loader("/home/xmg_cmw/EL/EL/final_data/", WINDOW_CONTEXT*2, WINDOW_DOC, WINDOW_BODY,val=False, test=True, shuffle=True, num_workers=0)
	doc_men, doc_totalmen = get_mentionNum("/home/xmg_cmw/EL/EL/final_data/doc_mentionNum.tsv")
	ww_file = open("/home/xmg_cmw/EL/EL/lius/NCEL-master/ncel_train_data/test/wikipedia/wikipedia-name2bracket.tsv")
	ww_list=[]
	for line in ww_file:
		ww_list.append(line.split("\t")[0])
	
	cnn_score=Fai_score(FILTER_NUM, FILTER_WINDOW, WINDOW_DOC, WINDOW_CONTEXT, WINDOW_TITLE, EMBEDDING, LAMDA)
	cnn_score=cnn_score.cuda()
	cnn_score.load_state_dict(torch.load(LOCAL_MODEL_LOC))
	
	eval(cnn_score, doc_men, doc_totalmen, ww_list, data_loader_test)
	
if __name__ == "__main__":
	main()