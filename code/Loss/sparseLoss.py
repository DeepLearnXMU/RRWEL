#-*- encoding: utf-8 -*-
import torch
import torch.nn as nn  
import torch.nn.functional as F
from torch.autograd import Variable

class SparseLossFunc(nn.Module):
	def __init__(self):
		super(SparseLossFunc, self).__init__()
		return
	
	def forward(self, groupped_res_tmp, gold_res):
		loss=(groupped_res_tmp-gold_res).sum()
		return loss
