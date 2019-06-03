#-*- encoding: utf-8 -*-
import torch
import torch.nn as nn  
import torch.nn.functional as F
from torch.autograd import Variable

class SelfLossFunc(nn.Module):
	def __init__(self):
		super(SelfLossFunc, self).__init__()
		return
	
	def forward(self, predict, gold, local, mention_entity, m, n):
		#predict是预测的结果，gold是标准答案
		#P_e是神经网络entity->entity的参数，SR是统计的
		loss1 = F.multilabel_soft_margin_loss(predict, gold, size_average=True)
		dist=F.pairwise_distance(local,predict)
		m_e=[]
		for i in range(m):
			temp=[]
			for j in range(n):
				temp.append(int(mention_entity[i][j]))
			m_e.append(temp)
		m_e1 = Variable(torch.Tensor(m_e)).cuda()
		normal=m_e1*dist
		loss2 = normal.sum()/(len(normal))
		loss=torch.add(loss1,loss2)
		return loss

'''

https://www.zhihu.com/question/66988664/answer/247952270


get the params :
http://www.pytorchtutorial.com/pytorch-note5-save-and-restore-models/
'''