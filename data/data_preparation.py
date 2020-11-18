
# coding: utf-8

# In[ ]

#data_size: the total number of data you want to put in the model( for example: twitter15 is 742)
#retweet_user_size: the length of retweet propagation you want to utilize( in the paper we use 40 retweet users)
data_size=742
retweet_user_size=40

##read the Data 
import numpy as np
import os
path=r""
files= os.listdir(path)
a=[]
k=[]
for file in range(0,data_size):
    f = open(r"{}.txt".format(file))
    results=[]
    k.append(file)    
    next(f)
    for line in f.readlines():
        results.append(list(map(float,line.split(','))))
    results=np.array(results)
    results=results.tolist()
    a.append(results)  
    
##let all the rewteet user size of news be the same
import random
from sklearn import preprocessing
data_all=[]
for i in range(0,data_size):
    if len(a[i])>=rewteet_user_size:
        k=a[i][0:rewteet_user_size]
        
        data_all.append(k)
    else:
        a[i]=np.asarray(a[i])
        q=a[i][np.random.choice(a[i].shape[0],rewteet_user_size,replace=True),:]
        q=q.tolist()
        a[i]=a[i].tolist()
        k=a[i].extend(q)
        k=a[i][0:rewteet_user_size]
        
        data_all.append(k)
        
##use the user profile to calculate their cosine similarity for building the graph
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
path=r"  "
files= os.listdir(path)
a=[]
cos=[]
for j in data_all:
    adj=[]
    data=pd.read_csv(data_all[j])
    
    for i in range(0,retweet_user_size):
        a=data.iloc[i,:]
        a=np.array(a)
        a=a.reshape(1,-1)
        similar=[]
        
        for k in range(0,retweet_user_size):
            b=data.iloc[k,:]
            b=np.array(b)
            b=b.reshape(1,-1)
            cosine=cosine_similarity(a,b)
            similar.append(np.round(cosine,2))
            
        similar=np.array(similar)
        similar=similar.flatten()
        adj.append(similar)
    cos.append(adj)

 
#encode the news content
import pandas as pd
import numpy as np
with open(r".txt", 'r') as f: # read all the news content 
    next(f)
    contents = f.readlines()

#vocab_size: how many different words in the news content  
from keras.preprocessing.text import one_hot
vocab_size=
encoded_docs=[one_hot(d,vocab_size) for d in contents]

from keras.preprocessing.sequence import pad_sequences
padded_docs=pad_sequences(encoded_docs,maxlen= ,padding="post") #let all the word embedding be the same length

