#-*- encoding: utf-8 -*-
import torch
import torch.nn as nn  
import torch.nn.functional as F
from torch.autograd import Variable

class GlobalLossFunc(nn.Module):
	def __init__(self):
		super(GlobalLossFunc, self).__init__()
		return
	
	def forward(self, s, s_global, local_score, global_score, y_true, GAMA, flag):
		flag = int(flag.split(":")[2])
		loss1 = F.cross_entropy(local_score, y_true)
		#l1_dis = nn.PairwiseDistance(p=1)
		#loss2 = l1_dis(s, s_global)
		loss2 = F.pairwise_distance(s,s_global)
		loss3 = F.cross_entropy(global_score, y_true)
		#loss2=F.kl_div(s,fai_global_score)
		"""
		print("s")
		print(s.cpu().data.numpy())
		print("fai_global_score")
		print(fai_global_score.cpu().data.numpy())
		tt = fai_global_score * torch.reciprocal(s)
		print("fai_global_score * torch.reciprocal(s)")
		print(tt.cpu().data.numpy())
		loss2 = fai_global_score * torch.log(tt)
		print("loss")
		print(loss2.cpu().data.numpy())
		loss2 = loss2.sum()
		"""
		if flag==1:
			loss=torch.add((1-GAMA)*loss1,GAMA*loss2)
		if flag==2:
			loss=torch.add(loss1,GAMA*loss3)
		if flag==3:
			loss = loss3 
		return loss
