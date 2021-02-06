import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
#import gensim library
import gensim
import nltk
from gensim.models import LdaModel,LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LsiModel
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#load preprocessed tweets
traindf=pd.read_csv('output/vad_hlex_lex_doc2vec_freq_bi_tweets_train.csv')
testdf=pd.read_csv('output/vad_hlex_lex_doc2vec_freq_bi_tweets_test.csv')
devdf=pd.read_csv('output/vad_hlex_lex_doc2vec_freq_bi_tweets_dev.csv')
#emotion classes
em=['anger', 'anticipation', 'disgust', 'fear','joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise','trust']
stopwrds = pd.read_csv('dataset/stopwords.txt',names=["wrds"])
tweets_preprocessed = []
for e in em:
    st = ' '.join(traindf[traindf[e]==1]["clean"])
    st = nltk.word_tokenize(st)
    lst = [x for x in st if x not in stopwrds["wrds"].values.tolist()]
    tweets_preprocessed.extend([lst])

# Set training parameters.
num_topics = 1
chunksize = 2000
passes = 50
iterations = 50
eval_every = 100
for i,e in enumerate(em):
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    gensim_dictionary = gensim.corpora.Dictionary([tweets_preprocessed[i]])
            
    # Bag-of-words representation of the documents.
    gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in [tweets_preprocessed[i]]]
    
    #Make a index to word dictionary.
    temp = gensim_dictionary[0]  # This is only to "load" the dictionary.
    id2word = gensim_dictionary.id2token

    lda_model = LdaModel(corpus=gensim_corpus,id2word=id2word,iterations=iterations,num_topics=num_topics,passes=passes,eval_every=eval_every,minimum_probability =0
                         ,per_word_topics=True,random_state=123)
    print(e,lda_model.print_topics())

    traindf["coh_lda_"+str(e)]=traindf["clean"].apply(lambda x:CoherenceModel(model=lda_model, texts=[nltk.word_tokenize(x)],dictionary=gensim_dictionary, coherence='c_v')
                                        .get_coherence())
    devdf["coh_lda_"+str(e)]=devdf["clean"].apply(lambda x: CoherenceModel(model=lda_model, texts=[nltk.word_tokenize(x)],dictionary=gensim_dictionary, coherence='c_v')
                                        .get_coherence())
    testdf["coh_lda_"+str(e)]=testdf["clean"].apply(lambda x:CoherenceModel(model=lda_model, texts=[nltk.word_tokenize(x)],dictionary=gensim_dictionary, coherence='c_v')
                                        .get_coherence())

traindf.head(10)
traindf.to_csv("output/coh_lex_doc_freq_tweets_train.csv",index=False)
testdf.to_csv("output/coh_lex_doc_freq_tweets_test.csv",index=False)
devdf.to_csv("output/coh_lex_doc_freq_tweets_dev.csv",index=False)
