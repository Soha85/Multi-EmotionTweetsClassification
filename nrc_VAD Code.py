import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os

#load tweets and features extracted
traindf=pd.read_csv('output//hlex_lex_doc2vec_freq_bi_tweets_train.csv',encoding="utf-8")
devdf=pd.read_csv('output//hlex_lex_doc2vec_freq_bi_tweets_dev.csv',encoding="utf-8")
testdf=pd.read_csv('output//hlex_lex_doc2vec_freq_bi_tweets_test.csv',encoding="utf-8")
traindf.head(10)
lexdf=pd.read_csv('dataset//NRC-VAD-Lexicon.txt',encoding='utf-8',sep="\t")
lexdf.head(10)

#compute average of tokens occurrence valence, arousal, and dominance
traindf["V"]=traindf["clean"].apply(lambda x : sum([lexdf[lexdf["Word"]==wrd]["Valence"].iloc[0] for wrd in x.split(" ") if len(lexdf[lexdf["Word"]==wrd])>0 ])/len(x.split(" ")))
traindf["D"]=traindf["clean"].apply(lambda x : sum([lexdf[lexdf["Word"]==wrd]["Dominance"].iloc[0] for wrd in x.split(" ") if len(lexdf[lexdf["Word"]==wrd])>0 ])/len(x.split(" ")))
traindf["A"]=traindf["clean"].apply(lambda x : sum([lexdf[lexdf["Word"]==wrd]["Arousal"].iloc[0] for wrd in x.split(" ") if len(lexdf[lexdf["Word"]==wrd])>0 ])/len(x.split(" ")))

devdf["V"]=devdf["clean"].apply(lambda x : sum([lexdf[lexdf["Word"]==wrd]["Valence"].iloc[0] for wrd in x.split(" ") if len(lexdf[lexdf["Word"]==wrd])>0 ])/len(x.split(" ")))
devdf["D"]=devdf["clean"].apply(lambda x : sum([lexdf[lexdf["Word"]==wrd]["Dominance"].iloc[0] for wrd in x.split(" ") if len(lexdf[lexdf["Word"]==wrd])>0 ])/len(x.split(" ")))
devdf["A"]=devdf["clean"].apply(lambda x : sum([lexdf[lexdf["Word"]==wrd]["Arousal"].iloc[0] for wrd in x.split(" ") if len(lexdf[lexdf["Word"]==wrd])>0 ])/len(x.split(" ")))

testdf["V"]=testdf["clean"].apply(lambda x : sum([lexdf[lexdf["Word"]==wrd]["Valence"].iloc[0] for wrd in x.split(" ") if len(lexdf[lexdf["Word"]==wrd])>0 ])/len(x.split(" ")))
testdf["D"]=testdf["clean"].apply(lambda x : sum([lexdf[lexdf["Word"]==wrd]["Dominance"].iloc[0] for wrd in x.split(" ") if len(lexdf[lexdf["Word"]==wrd])>0 ])/len(x.split(" ")))
testdf["A"]=testdf["clean"].apply(lambda x : sum([lexdf[lexdf["Word"]==wrd]["Arousal"].iloc[0] for wrd in x.split(" ") if len(lexdf[lexdf["Word"]==wrd])>0 ])/len(x.split(" ")))
traindf.head(10)

#save lexicon features to csv
traindf.to_csv("output//vad_hlex_lex_doc2vec_freq_bi_tweets_train.csv",index=False,encoding="utf-8")
devdf.to_csv("output//vad_hlex_lex_doc2vec_freq_bi_tweets_dev.csv",index=False,encoding="utf-8")
testdf.to_csv("output//vad_hlex_lex_doc2vec_freq_bi_tweets_test.csv",index=False,encoding="utf-8")
