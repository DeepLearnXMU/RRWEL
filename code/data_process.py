#coding=utf-8
import pickle
import gensim
import nltk
import math
import numpy

def pkl_list():
    docname_list=[]
    mention_list=[]
    entity_list=[]
    d2m={}
    m2e={}
    e2e={}
    entity2TD={}
    with open('gold_kb_2.xml','r',encoding='utf-8') as fr_kb,open('gold_el.tsv','r',encoding='utf-8') as fr_el:
        '''获取entity_list e2e e2TD entityT_max_len'''
        l=0
        for line in fr_kb:
            line=line.strip()
            line_split=line.split('|||')
            entity=line_split[0]
            related_entities_set=set(line_split[3].split('//'))#相关实体去重去重
            related_entities=list(related_entities_set)
            # related_entities.sort()#相关实体排序，可有可无
            new_related_entities=[e for e in related_entities if (e!=entity and e.strip()!='')]
            related_entities=new_related_entities
            title=line_split[2]
            if entity not in entity_list:
                entity_list.append(entity)
                e2e[entity]=related_entities
                entity2TD[entity]=[title,line_split[4]]
            else:#应该不会进入该分支
                # e2e[entity].update(related_entities)
                print('error')
                input()
            
            e_l=len(nltk.tokenize.word_tokenize(title.lower()))
            if e_l>l:
                l=e_l
        entityT_max_len=l
        if entityT_max_len<1:
            print('error')
            input()
        with open('entityT_max_len.txt','w',encoding='utf-8') as fw:
            fw.write(str(entityT_max_len)+'\n')
        print("entity title max len is : "+str(entityT_max_len))

        '''获取mention_list d2m m2e'''
        for line in fr_el:
            line=line.strip()
            line_split=line.split('|||')
            mention=line_split[0]
            entity=line_split[1].split('//')#多个标准答案的列表
            docname=line_split[2]

            if entity=='':
                print('mention to entity error??')
            if mention not in mention_list:
                mention_list.append(mention)
                m2e[mention]=set(entity)
            else:
                m2e[mention].update(set(entity))
            '''d2m:document name->[mention offset groundtruth_entity_id]s'''
            if docname not in docname_list:
                docname_list.append(docname)
                d2m[docname]=[[mention,line_split[3],line_split[1],len(mention.split(' '))]]
            else:
                d2m[docname].append([mention,line_split[3],line_split[1],len(mention.split(' '))])
        new_m2e={}
        for x in m2e:
            new_m2e[x]=list(m2e[x])
        m2e=new_m2e

    #切分测试集训练集,运行完还需要人工处理一步

    test_data=[]
    val_data=[]
    train_data=[]
    with open('el_train.xml','w',encoding='utf-8') as fw_train,open('el_test.xml','w',encoding='utf-8') as fw_test,open('el_val.xml','w',encoding='utf-8') as fw_val,open('gold_el.tsv','r',encoding='utf-8') as fr_el:
        print('数据切分')
        for line in fr_el:
            line=line.strip()
            line_split=line.split('|||')
            docname=line_split[2]
            if 'testa ' in docname:
                val_data.append(line)
            elif 'testb ' in docname:
                test_data.append(line)
            else:
                train_data.append(line)
        
        for line in test_data:
            fw_test.write(line+'\n')
        for line in train_data:
            fw_train.write(line+'\n')
        for line in val_data:
            fw_val.write(line+'\n')
            
    train_doc_list=[]
    with open('el_train.xml','r',encoding='utf-8') as fr:
        for line in fr:
            line=line.strip()
            line_split=line.split('|||')
            docname=line_split[2]
            if docname not in train_doc_list:
                train_doc_list.append(docname)
    test_doc_list=[]
    with open('el_test.xml','r',encoding='utf-8') as fr:
        for line in fr:
            line=line.strip()
            line_split=line.split('|||')
            docname=line_split[2]
            if docname not in test_doc_list:
                test_doc_list.append(docname)
    val_doc_list=[]
    with open('el_val.xml','r',encoding='utf-8') as fr:
        for line in fr:
            line=line.strip()
            line_split=line.split('|||')
            docname=line_split[2]
            if docname not in val_doc_list:
                val_doc_list.append(docname)
    
    with open('pkl/train_docs.pkl','wb') as f:
        pickle.dump(train_doc_list, f)
    with open('pkl/test_docs.pkl','wb') as f:
        pickle.dump(test_doc_list, f)
    with open('pkl/val_docs.pkl','wb') as f:
        pickle.dump(val_doc_list, f)
        
    with open('pkl/docname_list.pkl','wb') as f:
        pickle.dump(docname_list, f)
    with open('pkl/mention_list.pkl','wb') as f:
        pickle.dump(mention_list, f)
    with open('pkl/entity_list.pkl','wb') as f:
        pickle.dump(entity_list, f)

    with open('pkl/d2m.pkl','wb') as f:
        pickle.dump(d2m, f)
    with open('pkl/m2e.pkl','wb') as f:
        pickle.dump(m2e, f)
    with open('pkl/e2e.pkl','wb') as f:
        pickle.dump(e2e, f)
    with open('pkl/e2td.pkl','wb') as f:
        pickle.dump(entity2TD, f)

class Vocabulary(object):
    ##两个字典
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    ##添加单词
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    ##返回单词对应的id
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def get_word(self,i):
        return self.idx2word[i]
        
    ##返回词库大小
    def __len__(self):
        return len(self.word2idx)

def build_vocab():
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/xmg/EL/data/enwiki-vecs.bin',binary=True) 
    words = model.vocab

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    
    with open('pkl/vocab.pkl','wb') as f:
        pickle.dump(vocab, f)
    
    selfnptype=model['happy'].dtype
    unk=numpy.random.random(size=300).astype(selfnptype)*2-1
    with open('pkl/unk.pkl','wb') as f:
        pickle.dump(unk, f)
    
if __name__=='__main__':
    pkl_list()
    build_vocab()
    