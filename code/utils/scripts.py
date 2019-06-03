import torch
from torch.autograd import Variable
import os
import math
import numpy as np
import nltk
import torch.nn as nn 
import re
import datetime
import pickle 
from tqdm import tqdm
from utils.data_loader_f import *
from model.local_score_model import Local_Fai_score
np.set_printoptions(threshold=np.NaN)

def Fmeasure(count_true, count_label, actual_mentions, total_mentions, actual_correct):
	acc = count_true * 1.0 / count_label
	ma_precs = [correct / float(actual_mentions[i]) for i, correct in enumerate(actual_correct)]
	ma_recs = [correct / float(total_mentions[i]) for i, correct in enumerate(actual_correct)]
	eval_mi_rec = sum(actual_correct) / float(sum(total_mentions))
	eval_ma_rec = sum(ma_recs) / float(len(ma_recs))
	eval_mi_prec = sum(actual_correct) / float(sum(actual_mentions))
	eval_ma_prec = sum(ma_precs) / float(len(ma_precs))
	eval_mi_f1 = 2 * eval_mi_rec * eval_mi_prec / (eval_mi_rec + eval_mi_prec)
	eval_ma_f1 = 2 * eval_ma_rec * eval_ma_prec / (eval_ma_rec + eval_ma_prec)
	return acc, eval_mi_prec, eval_ma_prec, eval_mi_rec, eval_ma_rec, eval_mi_f1, eval_ma_f1

def ZeroInitializer(param):
    shape = param.size()
    init = np.zeros(shape).astype(np.float32)
    param.data.set_(torch.from_numpy(init))

def extract_data_from_dataloader(k,finetune):
	mention=list(k[0][0])#type_k[0][0]:tuple,tuple-->list
	m=len(mention)
	y_label=k[0][1]
	filename=k[0][2]#type_k[0][3]:string
	ms_x=k[0][3]
	mention_vec=ms_x.cuda()#type_k[0][4]:list
	mc_x=k[0][4]
	context_vec=mc_x.cuda()
	md_x=k[0][5]
	doc_vec=md_x.cuda()
	et_x=k[0][6]
	title_vec=et_x.cuda()
	x=k[0][7]
	body_vec=x.cuda()
	mention_entity=k[0][8]#type_k[0][9]:floattensor m*n
	entity_entity=k[0][9]#type_k[0][10]:floattensor n*n
	SR=k[0][10]#type_k[0][10]:floattensor n*n
	n=len(entity_entity)
	s_features = k[0][-1]
	return y_label, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, filename, s_features

def sloppyMathLogSum(self,vals):
	m=vals.max()
	r=torch.log(torch.exp(vals-m).sum())
	r=r+m
	return r

def get_mentionNum(filename):
	doc_men={}
	doc_totalmen={}
	f = open(filename, encoding='utf-8')
	lines = f.readlines()
	for line in lines:
		s=line.split("|||")
		doc_men[s[0]]=s[1]
		doc_totalmen[s[0]]=s[-1]
		
	return doc_men, doc_totalmen

def get_testlist(in_f):
	#key::aquaint,ace2004,clueweb12,msnbc,wikipedia,aida_conll_train,aida_conll_testa,aida_conll_testb
	test_dict = {}
	with open(in_f, "r", encoding="utf-8") as f:
		for line in f:
			s = line.split("|||")
			if s[0] in test_dict:
				test_dict[s[0]].append(s[1])
			else:
				test_dict[s[0]]=[]
				test_dict[s[0]].append(s[1])
	return test_dict
	
def eval(ROOT, cnn_score, doc_men, doc_totalmen, data_loader_test, dataflag, Ispretrain, pos_embed_dict, RANDOM_K, LAMDA, MODEL_FLAG):
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
	test_dict = get_testlist(ROOT+"/dataset/docname.txt")
	mention_num = 0
	wrong_num = 0
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
		mention_num += m
		for i in range(m):
			trueflag = 0
			for j in range(n):
				if int(y_label[i][j])==1:trueflag=1
			if trueflag==0:
				wrong_num+=1
		if m!=int(doc_men[filename]):
			print(str(m)+"|||"+str(doc_men[filename]))
			print("erooooooor!!") 
			
		if Ispretrain:
			local_score, local_score_softmax, local_score_uniform = cnn_score(mention_entity,m, n, mention_vec, context_vec, doc_vec, title_vec, body_vec, sfeats)
			fai_score = local_score_softmax
		else:
			entity_embed_f=open(ROOT+"/pkl/entity_embed/"+filename+".pkl", 'rb')
			entity_embed_dict = pickle.load(entity_embed_f)

			s, s_global, local_score, fai_local_score_softmax, global_score = cnn_score(mention_entity, entity_entity, SR, m, n, mention_vec, context_vec, doc_vec, title_vec, body_vec, sfeats, RANDOM_K, LAMDA, MODEL_FLAG, pos_embed_dict, entity_embed_dict)
			fai_score = global_score

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

		for i in range(m):
			y_true_temp=[]
			for j in range(n):
				if(int(y_label[i][j])==1):
					y_true_temp.append(j)
			y_true.append(y_true_temp)

	acc, eval_mi_prec, eval_ma_prec, eval_mi_rec, eval_ma_rec, eval_mi_f1, eval_ma_f1 = Fmeasure(count_true,
																								 count_label,
																								 actual_mentions,
																								 total_mentions,
																								 actual_correct)
	endtime = datetime.datetime.now()
	print("data_flag:\n1\ttesta\n2\ttestb\n3\taquaint\n4\tace2004\n5\tclueweb12\n6\tmsnbc\n7\twikipedia")
	print(count_test)
	print("data_flag:"+str(dataflag)+"|||time:"+str((endtime - starttime).total_seconds())+"|||acc:"+str(acc))
	print("data_flag:"+str(dataflag)+"eval_mi_prec:"+str(eval_mi_prec)+"|||eval_mi_rec:"+str(eval_mi_rec)+"|||eval_mi_f1:"+str(eval_mi_f1))
	print("data_flag:"+str(dataflag)+"eval_ma_prec:"+str(eval_ma_prec)+"|||eval_ma_rec:"+str(eval_ma_rec)+"|||eval_ma_f1:"+str(eval_ma_f1))
	print(mention_num)
	print(wrong_num)
	
	return 	acc, eval_mi_prec, eval_ma_prec, eval_mi_rec, eval_ma_rec, eval_mi_f1, eval_ma_f1
	