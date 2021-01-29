
import numpy as np 
import pandas as pd 
import os
#load NRC emotion lexicon
emotion_lexicon=pd.read_csv('dataset//NRC-Emotion-Lexicon.csv')
emotion_lexicon.columns.values
#emotions list of lexicon
em=["Positive", "Negative", "Anger","Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise","Trust"]
#exclude english words only
emotion_lexicon=emotion_lexicon[["English (en)","Positive", "Negative", "Anger",
       "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise","Trust"]]
#load tweets and features extracted
traindf=pd.read_csv('output//doc2vec_freq_bi_tweets_train.csv',encoding="utf-8")
devdf=pd.read_csv('output//doc2vec_freq_bi_tweets_dev.csv',encoding="utf-8")
testdf=pd.read_csv('output//doc2vec_freq_bi_tweets_test.csv',encoding="utf-8")
#view columns name
traindf.columns.values
#compute average of tokens occurrence in each emotion class based on lexicon
for e in em:
    traindf["lex_"+str(e)]=traindf["clean"].apply(lambda x : sum(
        [emotion_lexicon[emotion_lexicon["English (en)"]==wrd][str(e)].values[0] for wrd in x.split(" ") if len(emotion_lexicon[emotion_lexicon["English (en)"]==wrd])>0])
                                              /len(x.split(" ")))
    devdf["lex_"+str(e)]=devdf["clean"].apply(lambda x : sum(
        [emotion_lexicon[emotion_lexicon["English (en)"]==wrd][str(e)].values[0] for wrd in x.split(" ") if len(emotion_lexicon[emotion_lexicon["English (en)"]==wrd])>0])
                                              /len(x.split(" ")))
    testdf["lex_"+str(e)]=testdf["clean"].apply(lambda x : sum(
        [emotion_lexicon[emotion_lexicon["English (en)"]==wrd][str(e)].values[0] for wrd in x.split(" ") if len(emotion_lexicon[emotion_lexicon["English (en)"]==wrd])>0])
                                              /len(x.split(" ")))
traindf.head(10)
#save lexicon features to csv
traindf.to_csv("output//lex_doc2vec_freq_bi_tweets_train.csv",index=False,encoding="utf-8")
devdf.to_csv("output//lex_doc2vec_freq_bi_tweets_dev.csv",index=False,encoding="utf-8")
testdf.to_csv("output//lex_doc2vec_freq_bi_tweets_test.csv",index=False,encoding="utf-8")
