
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
#load tweets and features extracted
traindf=pd.read_csv('output//lex_doc2vec_freq_bi_tweets_train.csv',encoding="utf-8")
devdf=pd.read_csv('output//lex_doc2vec_freq_bi_tweets_dev.csv',encoding="utf-8")
testdf=pd.read_csv('output//lex_doc2vec_freq_bi_tweets_test.csv',encoding="utf-8")
lexdf=pd.read_csv('dataset//NRC-Hashtag-Emotion-Lexicon-v0.2.txt',sep="\t",names=['emotion','word','intensity'])
lexdf.head(10)
lexdf["word"]=lexdf["word"].apply(lambda x: str(x).replace('#',''))
lexdf.head(10)
sns.countplot(x="emotion",data=lexdf)
#emotions list of lexicon
em=["anger","anticipation", "disgust", "fear", "joy", "sadness", "surprise","trust"]
def hash_lexicon(hashlist,e):
    wrds=list(map(str, hashlist.strip().lower().replace('\'','').strip('[]').split(',')))
    em=0
    for wrd in wrds:        
        if lexdf[lexdf["word"]==wrd.strip()]["emotion"].all()==e:
            em = em +1
    em= em / len(wrds)
    em = round(em,4)
    return em
#compute average of tokens occurrence in each emotion class based on lexicon
for e in em:
    traindf["hlex_"+str(e)]=traindf["hashtags"].apply(lambda x : hash_lexicon(x,e))
    
    devdf["hlex_"+str(e)]=devdf["hashtags"].apply(lambda x : hash_lexicon(x,e))
    
    testdf["hlex_"+str(e)]=testdf["hashtags"].apply(lambda x : hash_lexicon(x,e))
#save lexicon features to csv
traindf.to_csv("output//hlex_lex_doc2vec_freq_bi_tweets_train.csv",index=False,encoding="utf-8")
devdf.to_csv("output//hlex_lex_doc2vec_freq_bi_tweets_dev.csv",index=False,encoding="utf-8")
testdf.to_csv("output//hlex_lex_doc2vec_freq_bi_tweets_test.csv",index=False,encoding="utf-8")
