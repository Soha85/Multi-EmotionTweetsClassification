# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:02:59 2021

@author: 20101
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


#load training data
traindf=pd.read_csv('output/cleaned_training_tweets.csv',encoding='utf-8',sep=",")
#load testing data
testdf=pd.read_csv('output/cleaned_testing_tweets.csv',encoding='utf-8',sep=",")
#load develop data
devdf=pd.read_csv('output/cleaned_develop_tweets.csv',encoding='utf-8',sep=",")

#import libraries
import re
import nltk
from nltk import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#initiate lists of emotions and frequency distribution
em=["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"]
uni_lst,bi_lst=[],[]

#compute unigram frequency distribution
def uni_freq(x):
    tmp=' '.join(traindf[traindf[x]==1]["clean"])
    tmp=re.sub('\n','',tmp)
    return FreqDist(nltk.word_tokenize(tmp))

#compute bigram frequency distribution
def bi_freq(x):
    tmp=' '.join(traindf[traindf[x]==1]["clean"])
    tmp=re.sub('\n','',tmp)
    tmp_bi=nltk.bigrams(nltk.word_tokenize(tmp))
    return FreqDist(tmp_bi)

#iterate over emotions 
for x in em:
    uni_lst.append(uni_freq(x))
    bi_lst.append(bi_freq(x))
    
    #function of display word cloud
def display_wrd_cloud(lst,i):
    wc = WordCloud(width=800, height=400, max_words=100).generate_from_frequencies(lst)
    plt.subplot(6,2,i)
    plt.imshow(wc, interpolation="bilinear")
    plt.title(em[i-1])
    plt.axis("off")

plt.subplots(6, 2,figsize=(15,25))

#WordCloud of emotions
for i,x in enumerate(em):
    display_wrd_cloud(uni_lst[i],i+1)

plt.subplot(6,2,12)
plt.axis("off")

plt.show()

#define fuction to plot maximum 20 tokens
def plt_freq(rows,cols,lst,i):
    plt.subplot(rows,cols,i)
    wrd,lbl=zip(*lst)
    try:
        wrd=[x+","+y for (x,y) in wrd]
        plt.barh(wrd,lbl)
    except:
        plt.barh(wrd,lbl)
    plt.title(em[i-1])

#plot highest common 20 tokens in each emotion
rows=6
cols=2
plt.subplots(rows, cols,figsize=(15,25))
for i,x in enumerate(em):
    plt_freq(rows,cols,uni_lst[i].most_common(20),i+1)

plt.subplot(rows,cols,12)
plt.axis("off")
plt.show()

#plot highest common 10 bi-grams in each emotion
rows,cols=11,1
plt.subplots(rows, cols,figsize=(15,30))
for i,x in enumerate(em):
    plt_freq(rows,cols,bi_lst[i].most_common(10),i+1)

plt.show()

#intersections between each unigram frequency two emotions
intersections=dict({})
for i,val1 in enumerate(uni_lst):
    for j,val2 in enumerate(uni_lst):
        if (j>i):
            lbl=em[i]+" & "+em[j]
            intersect=len(set(val1.keys()).intersection(set(val2.keys())))
            print (lbl,":",intersect)
            intersections[lbl]=intersect
            
#plot bar graph to visualize intersections between each two emotions
plt.figure(figsize=(15,15))
plt.barh(list(intersections.keys()),list(intersections.values()))

#intersections between each bigram frequency two emotions
bi_intersections=dict({})
for i,val1 in enumerate(bi_lst):
    for j,val2 in enumerate(bi_lst):
        if (j>i):
            lbl=em[i]+" & "+em[j]
            intersect=len(set(val1.keys()).intersection(set(val2.keys())))
            print (lbl,":",intersect)
            bi_intersections[lbl]=intersect
            
#plot bar graph to visualize intersections between each two emotions
plt.figure(figsize=(15,15))
plt.barh(list(bi_intersections.keys()),list(bi_intersections.values()))


#compute average frequency distribution of tweet to each emotion for both unigram and bigram
for i,e in enumerate(em):
    traindf["freq_"+e]=traindf["clean"].apply(lambda x: sum([uni_lst[i].get(wrd)/len(uni_lst[i].keys()) if uni_lst[i].get(wrd)!=None else 0 for wrd in nltk.word_tokenize(x)]))
    traindf["bi_"+e]=traindf["clean"].apply(lambda x: sum([bi_lst[i].get(tpl)/len(bi_lst[i].keys()) if bi_lst[i].get(tpl)!=None else 0 for tpl in nltk.bigrams(nltk.word_tokenize(x))]))
#compute average frequency distribution of tweet to each emotion for both unigram and bigram
for i,e in enumerate(em):
    testdf["freq_"+e]=testdf["clean"].apply(lambda x: sum([uni_lst[i].get(wrd)/len(uni_lst[i].keys()) if uni_lst[i].get(wrd)!=None else 0 for wrd in nltk.word_tokenize(x)]))
    testdf["bi_"+e]=testdf["clean"].apply(lambda x: sum([bi_lst[i].get(tpl)/len(bi_lst[i].keys()) if bi_lst[i].get(tpl)!=None else 0 for tpl in nltk.bigrams(nltk.word_tokenize(x))]))
#compute average frequency distribution of tweet to each emotion for both unigram and bigram
for i,e in enumerate(em):
    devdf["freq_"+e]=devdf["clean"].apply(lambda x: sum([uni_lst[i].get(wrd)/len(uni_lst[i].keys()) if uni_lst[i].get(wrd)!=None else 0 for wrd in nltk.word_tokenize(x)]))
    devdf["bi_"+e]=devdf["clean"].apply(lambda x: sum([bi_lst[i].get(tpl)/len(bi_lst[i].keys()) if bi_lst[i].get(tpl)!=None else 0 for tpl in nltk.bigrams(nltk.word_tokenize(x))]))
   

#save feature values to csv
traindf.to_csv("output/freq_bi_tweets_train.csv",index=False,encoding="utf-8")
testdf.to_csv("output/freq_bi_tweets_test.csv",index=False,encoding="utf-8")
devdf.to_csv("output/freq_bi_tweets_dev.csv",index=False,encoding="utf-8")
