#coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import re
import datetime
import math
import gc

np.set_printoptions(threshold=np.NaN)

class Fai_score(nn.Module):
	def __init__(self, filter_num, filter_window, doc, context, title, embedding, lamda):
		super(Fai_score, self).__init__()
		self.dim_compared_vec=filter_num #卷积核个数
		self.num_words_to_use_conv = filter_window #卷积窗口大小
		self.lamda = lamda
		self.document_length = doc
		self.context_length = context*2
		self.title_length = title
		self.embedding_len = embedding #词向量长度
		self.surface_length=10

		self.relu_layer=nn.ReLU()
		
		# self.layer_ms = nn.Sequential(
		# 	nn.Linear(self.embedding_len,self.dim_compared_vec),
		# 	nn.LeakyReLU()
		# 	)
		self.conv_ms = nn.Conv2d(
				in_channels=1, #input_height
				out_channels=self.dim_compared_vec, #n_filters
				kernel_size=[self.num_words_to_use_conv,self.embedding_len] #filter_size
			)
		self.avg_ms = nn.AvgPool1d(kernel_size=self.surface_length-self.num_words_to_use_conv+1)
		
		self.conv_mc = nn.Conv2d(
				in_channels=1, #input_height
				out_channels=self.dim_compared_vec, #n_filters
				kernel_size=[self.num_words_to_use_conv,self.embedding_len] #filter_size
			)#(1,150,6,1)
		self.avg_mc = nn.AvgPool1d(kernel_size=self.context_length-self.num_words_to_use_conv+1)
		
		self.conv_md = nn.Conv2d(
				in_channels=1, #input_height
				out_channels=self.dim_compared_vec, #n_filters
				kernel_size=[self.num_words_to_use_conv,self.embedding_len] #filter_size
			)
		self.avg_md = nn.AvgPool1d(kernel_size=self.document_length-self.num_words_to_use_conv+1)
		
		self.conv_et = nn.Conv2d(
				in_channels=1, #input_height
				out_channels=self.dim_compared_vec, #n_filters
				kernel_size=[self.num_words_to_use_conv,self.embedding_len] #filter_size
			)
		self.avg_et = nn.AvgPool1d(kernel_size=self.title_length-self.num_words_to_use_conv+1)
		
		self.conv_eb = nn.Conv2d(
				in_channels=1, #input_height
				out_channels=self.dim_compared_vec, #n_filters
				kernel_size=[self.num_words_to_use_conv,self.embedding_len] #filter_size
			)
		self.avg_eb = nn.AvgPool1d(kernel_size=self.document_length-self.num_words_to_use_conv+1)
		
		self.softmax_layer = nn.Softmax(dim=1)
		
		self.layer_local = nn.Linear(6,1,bias=True)
			
	def cos(self, x, y):
		cos = (x*y).sum() / ( ( (x*x).sum().pow(0.5) ) * ( (y*y).sum().pow(0.5)) )
		cos=cos/2+0.5
		#distance = ((x-y)*(x-y)).sum().pow(0.5)
		return cos
		
	def masked_softmax(self, x, mask):
		x = x/0.1
		# print('y:'+str(y))
		x = math.e**x
		x = x * mask
		x = x/x.sum()
		return x

	def embedding_layer(self,mention_vec, context_vec, doc_vec, title_vec, body_vec):
		ms=[]
		mc=[]
		et=[]
		eb=[]
		for i in range(len(mention_vec)):
			x=mention_vec[i].unsqueeze(0)
			x=x.unsqueeze(1)
			x=self.conv_ms(x)
			x=self.relu_layer(x).squeeze(3)
			x=self.avg_ms(x).squeeze(2)
			ms.append(x)
			# x=self.layer_ms(mention_vec[i])
			# ms.append(x)
		doc_vec=doc_vec.unsqueeze(0)#cnn输入需要是四维的，第一维表示batch
		doc_vec=doc_vec.unsqueeze(1)#第二维表示channel
		md=self.conv_md(doc_vec)#(1,1,100,300)-->(1,150,96,1)
		md=self.relu_layer(md)#(1,150,96,1)-->(1,150,96,1)
		md=md.squeeze(3)#(1,150,96,1)-->(1,150,96)
		md=self.avg_md(md)#(1,150,96)-->(1,150,1)
		md=md.squeeze(2)#(1,150,1)-->(1,150)
		for i in range(len(context_vec)):
			x=context_vec[i].unsqueeze(0)
			x=x.unsqueeze(1)
			x=self.conv_mc(x)
			x=self.relu_layer(x).squeeze(3)
			x=self.avg_mc(x).squeeze(2)
			mc.append(x)
		for i in range(len(title_vec)):
			x=title_vec[i].unsqueeze(0)
			x=x.unsqueeze(1)
			x=self.conv_et(x)
			x=self.relu_layer(x).squeeze(3)
			x=self.avg_et(x).squeeze(2)
			et.append(x)
		for i in range(len(body_vec)):
			x=body_vec[i].unsqueeze(0)
			x=x.unsqueeze(1)
			x=self.conv_eb(x)
			x=self.relu_layer(x).squeeze(3)
			x=self.avg_eb(x).squeeze(2)
			eb.append(x)
		return ms,md,mc,et,eb

	def local_score(self, mention_entity, m, n, ms, md, mc, et, eb):
		print('mention:'+str(m)+'  entity:'+str(n))
		for i in range(m):
			mask=torch.Tensor(mention_entity[i].numpy())
			mask=Variable(mask,requires_grad=False).cuda()
			for j in range(n):
				if int(mention_entity[i][j])==1:
					cos_st=self.cos(ms[i],et[j])
					cos_dt=self.cos(md,et[j])
					cos_ct=self.cos(mc[i],et[j])
					cos_sb=self.cos(ms[i],eb[j])
					cos_db=self.cos(md,eb[j])
					cos_cb=self.cos(mc[i],eb[j])
					F_local=torch.cat((cos_st,cos_dt,cos_ct,cos_sb,cos_db,cos_cb),0)
					F_local=F_local.unsqueeze(0)#(1,6)
				else:
					cos_zero=Variable(torch.Tensor([0])).cuda()
					F_local=torch.cat((cos_zero,cos_zero,cos_zero,cos_zero,cos_zero,cos_zero),0)
					F_local=F_local.unsqueeze(0)
				if j==0:
					fai_local_score_cos=F_local
					#fai_local_score=F_local
				else:
					fai_local_score_cos=torch.cat((fai_local_score_cos,F_local),0)
					#fai_local_score=torch.cat((fai_local_score,F_local),0)

			fai_local_score_cos=self.layer_local(fai_local_score_cos)#n*6-->n*1
			#fai_local_score_cos=self.relu_layer(fai_local_score_cos)
			fai_local_score_cos=fai_local_score_cos.squeeze(1)
			fai_local_score_cos=fai_local_score_cos * mask
			fai_local_score_normalize=self.masked_softmax(fai_local_score_cos, mask)
			fai_local_score_normalize=fai_local_score_normalize.unsqueeze(0)
			fai_local_score_cos=fai_local_score_cos.unsqueeze(0)
			if i==0:
				fai_local_score_norm=fai_local_score_normalize
				fai_local_score=fai_local_score_cos
			else:
				fai_local_score_norm=torch.cat((fai_local_score_norm,fai_local_score_normalize),0)
				fai_local_score=torch.cat((fai_local_score,fai_local_score_cos),0)

		#print(fai_local_score.cpu().data.numpy())
		return fai_local_score,fai_local_score_norm

	
	def forward(self, weight, word, mention_entity, entity_entity, SR, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec,IS_PRETRAIN):
		ms,md,mc,et,eb=self.embedding_layer(mention_vec, context_vec, doc_vec, title_vec, body_vec)
		fai_local_score,fai_local_score_norm= self.local_score(mention_entity, m, n, ms, md, mc, et, eb)
		if IS_PRETRAIN:
			return fai_local_score,fai_local_score_norm