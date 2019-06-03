#coding=utf-8
import _pickle as pickle
import os
import string
# import gensim
import nltk
import numpy
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils import data
from torch.autograd import Variable
import re
import gc
import datetime

numpy.set_printoptions(threshold=numpy.NaN)

def del_string_punc(s):
    return ''.join([x for x in list(s) if x not in string.punctuation])

#数据集类
class DataSet(data.Dataset):
    def __init__(self,root,windows,doc_num,body_num,val,test):
        '''
        主要目标： 获取所有文件名，##并根据训练，验证，测试划分数据##
        '''
        self.root = root
        self.windows=windows
        self.doc_num=doc_num
        self.body_num=body_num
        self.EMBEDDING_SIZE=300
        self.train=None
        self.test=test
        if self.test:
            docname='pkl/test_docs.pkl'
            el_file='el_test.xml'
            self.folder=self.root+'/dataset/Doc/test_doc/'
            self.midfold='pkl'
            self.sr_fold=self.root+'/pkl/sr/test/'
        else:
            if val:
                docname='pkl/val_docs.pkl'
                el_file='el_val.xml'
                self.folder=self.root+'/dataset/Doc/testa_doc/'
                self.midfold='pkl'
                self.sr_fold=self.root+'/pkl/sr/val/'
            else:
                self.train=True
                docname='pkl/train_docs.pkl'
                el_file='el_train.xml'
                self.folder=self.root+'/dataset/Doc/train_doc/'
                self.midfold='pkl'
                self.sr_fold=self.root+'/pkl/sr/train/'


        with open(os.path.join(self.root, docname),'rb') as f:
            docname_list=pickle.load(f)
        self.docname_list=docname_list

        with open(os.path.join(self.root,self.midfold, 'entity_list.pkl'),'rb') as f:
            entity_list=pickle.load(f)
        self.entity_list=entity_list

        with open(os.path.join(self.root, self.midfold,'d2m.pkl'),'rb') as f:
            d2m=pickle.load(f)
        self.d2m=d2m

        with open(os.path.join(self.root,self.midfold, 'm2e.pkl'),'rb') as f:
            m2e=pickle.load(f)
        self.m2e=m2e

        with open(os.path.join(self.root, self.midfold,'e2e.pkl'),'rb') as f:
            e2e=pickle.load(f)
        self.e2e=e2e

        with open(os.path.join(self.root, self.midfold, 'e2TD.pkl'),'rb') as f:
            entity2TD=pickle.load(f)
        self.entity2TD=entity2TD

        with open(os.path.join(self.root, self.midfold, 'title2id.pkl'),'rb') as f:
            title2id=pickle.load(f)
        self.id2title={value:key for key,value in title2id.items()}

        with open(os.path.join(self.root, self.midfold, 'vocab.pkl'),'rb') as f:
            vocab=pickle.load(f)
        self.vocab=vocab

        with open(os.path.join(self.root, self.midfold, 'unk.pkl'),'rb') as f:
            unk=pickle.load(f)
        self.unk=unk
        self.nptype=self.unk.dtype
        self.min_vau=100
        
        if os.path.exists(os.path.join(self.root,'entityT_max_len.txt')):
            with open(os.path.join(self.root,'entityT_max_len.txt'),'r',encoding='utf-8') as fr:
                entityT_max_len=int(fr.readlines()[0].strip())
        else:
            print('process data error:there is not a file of entity len')
            input()
        self.entityT_max_len=entityT_max_len
            
    def get_doc_entity_num(self,docment):
        mentions,offsets,gold_entities_ids,m_ind,m_start=zip(*self.d2m[docment])
        mentions2entities=[self.m2e[x+'|'+m_ind[i]+'|'+docment] for i,x in enumerate(mentions)]
        related_entities=[]
        for x in mentions2entities:
            for value in x:
                if value not in related_entities:
                    related_entities.append(value)
        return len(related_entities)

    #像序列一样可以直接用下标获取，切片
    def __getitem__(self, index):
        """
        return:
        文档中的mention string列表（1维）
        文档中相关的实体集string列表（1维）
        标准实体答案（2维 tensor）  
        文档名称string   
        mention对应的词向量列表（3维）   
        mention对应的上下文向量列表（3维）   
        文档向量（2维）   
        entity title向量（3维）   
        entity body向量（3维）   
        mention到related entity的01对应矩阵（2维 tensor）   
        文档中mention相关的实体集到实体集的01关联矩阵（2维 tensor）   
        """
        folder=self.folder   
        docment=self.docname_list[index]
        vocab=self.vocab
        mentions,offsets,gold_entities_ids,m_ind,m_start=zip(*self.d2m[docment])
        tmp_mentions=[]
        tmp_offsets=[]
        tmp_gold_entities_ids=[]
        tmp_m_ind=[]
        tmp_m_start=[]
        sp={}
        with open(os.path.join(self.root,'features',docment),'rb') as fr_sp:
            sp=pickle.load(fr_sp)

        for i,m in enumerate(mentions) :
            es=self.m2e[m+'|'+m_ind[i]+'|'+docment]
            flag=True
            for v in es:
                if m+'|||'+m_start[i]+'|||'+v not in sp:
                    flag=False
                    break
            if flag:
                tmp_mentions.append(m)
                tmp_offsets.append(offsets[i])
                tmp_gold_entities_ids.append(gold_entities_ids[i])
                tmp_m_ind.append(m_ind[i])
                tmp_m_start.append(m_start[i])

        #print(len(tmp_mentions))
        mentions,offsets,gold_entities_ids,m_ind,m_start=tmp_mentions,tmp_offsets,tmp_gold_entities_ids,tmp_m_ind,tmp_m_start

        mentions2entities=[self.m2e[x+'|'+m_ind[i]+'|'+docment] for i,x in enumerate(mentions)]
        
        e2e01_idx=[]
        idx_0=0
        idx_1=idx_0
        related_entities=[]
        for ind,x in enumerate(mentions2entities):
            for value in x:
                related_entities.append(value+'|~!@# $ %^&*|'+str(m_ind[ind]))
                idx_1+=1
            e2e01_idx.append((idx_0,idx_1))
            idx_0=idx_1

        sparse_features={}

        
        for i,m in enumerate(mentions) :
            es=self.m2e[m+'|'+m_ind[i]+'|'+docment]
            for v in es:
                if m+'|||'+m_start[i]+'|||'+v in sp:
                    temp_sf=sp[m+'|||'+m_start[i]+'|||'+v]            
                    j=related_entities.index(v+'|~!@# $ %^&*|'+str(m_ind[i]))
                    sparse_features[str(i)+'|'+str(j)]=temp_sf
                    # del temp_sf
                else:
                    # break
                    print('features error:'+docment+' '+m+' : '+m_ind[i]+' ; '+v)
                    break
    
        # print('==end get sparse feature==')
        # nowtime=datetime.datetime.now()
        # print(nowtime)
        mentions_strings=mentions#字符串列表
        gold_entities=[]
        for line in gold_entities_ids:
            line=line.split('//')#多个标准答案
            line=[e for e in line if e.strip()!='']
            gold_entities.append(line)
        gold_entities_ind=[[0 for i in range(len(related_entities))] for j in range(len(mentions))]
        for i,entity_l in enumerate(gold_entities):
            for e in entity_l:
                ne=e+'|~!@# $ %^&*|'+str(m_ind[i])
                if ne not in related_entities:
                    print('file erroe')
                    print(docment)
                    print(ne)
                    
                gold_entities_ind[i][related_entities.index(ne)]=1
        mentions_offset=offsets

        #获取mention的向量
        #print("#######Surface vector...#######")
        mentions_surface=[]
        for m in mentions_strings:
            # print(m)
            tokens = nltk.tokenize.word_tokenize(m.lower())
            word_ind=[vocab(x) for x in tokens]
            mentions_surface.append(word_ind)

        # with open(os.path.join(self.root, 'AIDA-YAGO2-dataset.tsv'),'r',encoding='utf-8') as fr:
        #     cont=fr.readlines()

        #print("#######Context vector...#######")
        mentions_context=[]
        windows=self.windows
        for ind,offset in enumerate(mentions_offset):
            mentions_contex=[x.lower() for x in nltk.tokenize.word_tokenize(offset.lower())  if x not in string.punctuation]
            if len(mentions_contex)<self.windows:
                mentions_contex.extend(['<pad>']*((self.windows)-len(mentions_contex)))
            # mentions_context.append(mentions_contex)
            else:
                mentions_contex[:]=mentions_contex[:self.windows]
            mentions_context.append([vocab(del_string_punc(x)) for x in mentions_contex])
        #print("#######Document vector...#######")
        with open(os.path.join(folder,docment),'r',encoding='utf-8') as fr:
            cont=fr.read().strip() #已经分过词
        doc_words=[x.lower() for x in cont.split(' ') if x.strip()!=''][:self.doc_num]
        docment_vec=[vocab(x) for x in doc_words]
        if len(doc_words)<self.doc_num:
            docment_vec.extend([0]*(self.doc_num-len(doc_words)))

        new_mentions=mentions
        new_related_entities=related_entities
        '''mention 有多个词的时候(虽然好像只有1个)'''
        new_mentions_surface=[]
        for i,m in enumerate(mentions_surface):
            new_mentions_surface.append([])
            for id in m:
                new_mentions_surface[i].append(int(id))
            if len(new_mentions_surface[i])<self.windows:
                new_mentions_surface[i].extend([0]*((self.windows)-len(new_mentions_surface[i])))
            else:
                new_mentions_surface[i]=new_mentions_surface[i][:self.windows]
        
        new_mentions_context=mentions_context
        new_docment_vec=docment_vec

        new_docment_vec=Variable(torch.LongTensor(new_docment_vec))
        new_mentions_surface=Variable(torch.LongTensor(new_mentions_surface))
        new_mentions_context=Variable(torch.LongTensor(new_mentions_context))

        #print("#######Entity vector...#######")
        entities_Tvec=[]
        entities_Bvec=[]
        
        for e in new_related_entities:
            entity_id=e.split('|~!@# $ %^&*|')[0]
            title,body=self.entity2TD.get(entity_id,[self.id2title[entity_id],'']) 
            title_tokens = nltk.tokenize.word_tokenize(title.lower())[:self.windows]
            # print(title_tokens)
            body_tokens=nltk.tokenize.word_tokenize(body.lower())[:self.body_num]
            title_vec=[]
            for x in title_tokens:
                title_vec.append(vocab(x))
            body_vec=[]
            for x in body_tokens:
                body_vec.append(vocab(x))
            if len(title_vec)<self.entityT_max_len:
                title_vec.extend([0 for i in range((self.entityT_max_len-len(title_vec)))])
            # print(len(title_vec))
            if len(body_vec)<self.body_num:
                body_vec.extend([0 for i in range((self.body_num-len(body_vec)))])
            title_vec=title_vec[:self.entityT_max_len]
            body_vec=body_vec[:self.body_num]
            entities_Tvec.append(title_vec)
            entities_Bvec.append(body_vec)
        new_entities_Tvec=Variable(torch.LongTensor(entities_Tvec))
        new_entities_Bvec=Variable(torch.LongTensor(entities_Bvec))

        new_mentions2entities=[[0 for i in range(len(new_related_entities))] for j in range(len(new_mentions))]
        for i,m in enumerate(mentions2entities):
            for x in m:
                new_mentions2entities[i][new_related_entities.index(x+'|~!@# $ %^&*|'+str(m_ind[i]))]=1
        
        new_entities2entities=[[1 for i in range(len(new_related_entities))] for j in range(len(new_related_entities))]

        for ind,m in enumerate(mentions):
            # cur_m2e=mentions2entities[ind]
            ind_a,ind_b=e2e01_idx[ind]
            len_ab=ind_b-ind_a
            for xxx_ind in range(ind_a,ind_b):
                for jjj_ind in range(ind_a,ind_b):
                    new_entities2entities[xxx_ind][jjj_ind]=0
        
        for i,e in enumerate(new_entities2entities):
            new_entities2entities[i][i]=0
        
        gold_entities_ind=torch.FloatTensor(gold_entities_ind)

        new_mentions2entities=torch.Tensor(new_mentions2entities)
        new_entities2entities=torch.Tensor(new_entities2entities)

        with open(self.sr_fold+str(docment),'rb') as fr_pkl:
            entity_sr=pickle.load(fr_pkl)
        new_entity_sr=torch.Tensor(entity_sr)
        return new_mentions,gold_entities_ind,docment,new_mentions_surface,new_mentions_context,new_docment_vec,new_entities_Tvec,new_entities_Bvec,new_mentions2entities,new_entities2entities,new_entity_sr,sparse_features

    def __len__(self):
        return len(self.docname_list)

def collate_fn(data):
    return data

##获取数据
def get_loader(root,windows,doc_num,body_num,val,test, shuffle, num_workers):
    data = DataSet(root,windows,doc_num,body_num,val,test)

    # Data loader for gold_el dataset
    # test表示是否是测试，和shuffle互异
    # 返回  [一个文档中的mention列表（1维），标准实体列表（1维），文档中mention相关的实体集（1维），文档名称（一个string），mention对应的词向量列表（2维），mention对应的上下文向量列表（2维），文档向量（1维），mention对应的相关实体列表（2维），文档中mention相关的实体集的相关实体列表（2维）] for every iteration.

    data_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=1,# 批大小
                                               # 若dataset中的样本数不能被batch_size整除的话，最后剩余多少就使用多少
                                              shuffle=shuffle,# 是否随机打乱顺序
                                              num_workers=num_workers,# 多线程读取数据的线程数
                                              collate_fn=collate_fn
                                              )
    return data_loader