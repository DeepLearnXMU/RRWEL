import math
import torch
import pickle

def get_pos_embed(pos, dim):
	pos_embed = [math.cos(pos/math.pow(10000,(i-1)/dim)) if i%2==1 else math.sin(pos/math.pow(10000,i/dim)) for i in range(dim)]
	pos_embed = torch.Tensor(pos_embed)
	return pos_embed
	
if __name__ == '__main__':
	f = open("/home/xmg/EL/EL/yoga_code/data/pos_embed_128.pkl","wb")
	a={}
	for i in range(1000):
		a[i]=get_pos_embed(i, 128)
	pickle.dump(a, f)
	f.close()
	f=open("/home/xmg/EL/EL/yoga_code/data/pos_embed_128.pkl", 'rb')
	data1 = pickle.load(f)
	print(data1[1])
	print(data1[2])
	print(data1[900])
		