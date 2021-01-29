import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings
warnings.filterwarnings("ignore")

#load training data
traindf=pd.read_csv('output\\freq_bi_tweets_train.csv',encoding='utf-8',sep=",")

#load dev data
devdf=pd.read_csv('output\\freq_bi_tweets_dev.csv',encoding='utf-8',sep=",")

#load test data
testdf=pd.read_csv('output\\freq_bi_tweets_test.csv',encoding='utf-8',sep=",")


#initiate lists of emotions and word to vector distribution
em=["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"]
#load libraries
import gensim 
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import warnings
warnings.filterwarnings("ignore")

#transform tweets of specific emotion class to tokenized single paragraph
def tokenize_tweets(e):
    tmp = ' '.join(traindf[traindf[e]==1]["clean"])
    return nltk.word_tokenize(tmp)

#iterate over emotion classes to append tokenized tweets emotion in single list
data=[]
for e in em:
    data.append(tokenize_tweets(e))
    
#tag each emotion tweets
documents = [TaggedDocument(x,[i]) for i,x in enumerate(data)]
#call doc2vec 
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=10, epochs=40)
#build vocabulary from tagged documents
model.build_vocab(documents)
#train doc2vec model
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

#define cosine function
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
#compute feature vector of tweets
#measure cosine similarity between input tweet and each emotion tweets
for i,e in enumerate(em):
    emotion_vector=model.infer_vector(data[i])
    traindf["doc2vec_"+str(e)] = traindf["clean"].apply(lambda x: cosine(model.infer_vector(nltk.word_tokenize(x)),emotion_vector))

for i,e in enumerate(em):
    emotion_vector=model.infer_vector(data[i])
    devdf["doc2vec_"+str(e)] = devdf["clean"].apply(lambda x: cosine(model.infer_vector(nltk.word_tokenize(x)),emotion_vector))
    


for i,e in enumerate(em):
    emotion_vector=model.infer_vector(data[i])
    testdf["doc2vec_"+str(e)] = testdf["clean"].apply(lambda x: cosine(model.infer_vector(nltk.word_tokenize(x)),emotion_vector))

#save feature values training to csv
traindf.to_csv("output//doc2vec_freq_bi_tweets_train.csv",index=False,encoding="utf-8")
#save feature values dev to csv
devdf.to_csv("output//doc2vec_freq_bi_tweets_dev.csv",index=False,encoding="utf-8")
#save feature values test to csv
testdf.to_csv("output//doc2vec_freq_bi_tweets_test.csv",index=False,encoding="utf-8")
