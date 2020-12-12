# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
"""
LDA-ANALYSIS MODULE

Author: Maryam Rahafrooz
Created Aug 29 2020

This is a module to:
1) label the tweets that include hashtag with the hashtag modularity class,
   then aggregate the tweets of each class
2) apply LDA Mallet and compute c_v coherence for various number of topics 
   for number of topics starting from start_num to limit_num using atep_num
3) visualize the coherence values 
4) Visualize the LDA Mallet topics for the optimal model
"""
#reference: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

import pandas as pd
import numpy as np
import re
import os
from pprint import pprint
import pickle
import preprocessing

from wordcloud import WordCloud

import gensim
from gensim import corpora
import gensim.downloader as api
from gensim.models.wrappers import LdaMallet
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim as gensimvis



def find_topic_num(col_class, limit_num, start_num, step_num):
    """
    1) label the tweets that include hashtag with the hashtag modularity class,
    then aggregate the tweets of each class => texts
    2) apply LDA Mallet and compute c_v coherence for various number of topics 
       for number of topics starting from start_num to limit_num using atep_num
    3) visualize the coherence values 
    
    
    Parameters:
    col_class: class label column 
    limit_num:maximum numberfor topics
    start_num: minimun numbr of topics
    step_num: steps to increase topic numbers
    
    """
    col_text='tokenized'#tokenized or spacy_lemmatized
    #aggregate the hashdocs texts of col_text for hashtags belong to same class
    labeled_hashdocs=pd.read_csv('data files/labeled_hashdocs.csv')
    agg_labeled_tweets=pd.DataFrame(labeled_hashdocs.astype(str).groupby(col_class)[col_text].unique())
    agg_labeled_tweets=agg_labeled_tweets.reset_index()
    pattern = re.compile(r'\w+')
    agg_labeled_tweets[col_text]=agg_labeled_tweets[col_text].apply(lambda x: pattern.findall(str(x)))
    dfh_unpoprmvd=pd.read_csv('data files/dfh_unpoprmvd.csv')
    print( r'a total of', format(dfh_unpoprmvd.shape[0],','),' tweets are labeled into',
          format(labeled_hashdocs[col_class].nunique(),','),'specified hashtag clusters.')


    #create corpus: texts=data_lemmatized
    data_lemmatized  = [token for token in agg_labeled_tweets[col_text]]

    # Create Dictionary  
    id2word = corpora.Dictionary(data_lemmatized)

    # Term Document Frequency (in BOW format)
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]


    #Build LDA Mallet: find optimal number of topics using c_v
    # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    os.environ.update({'MALLET_HOME':r'C:/new_mallet/mallet-2.0.8/'})
    mallet_path = 'C:/new_mallet/mallet-2.0.8/bin/mallet' # path to mallet-2.0.8/bin/mallet, update it

    coherence_values = []
    model_list = []
    for num_topics in range(start_num, limit_num, step_num):
        model = LdaMallet(mallet_path,
                          corpus=corpus,
                          num_topics=num_topics,
                          id2word=id2word,
                          random_seed=1,
                          optimize_interval=5,
                          iterations=1500)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, 
                                        texts=data_lemmatized,
                                        dictionary=id2word,
                                        coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())


    # Show graph
    x = range(start_num, limit_num, step_num)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.figure(figsize=(20,10))
    plt.show()

    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 3))
    
    return corpus, id2word, model_list
        

def viz_optimal_malletlda(optimal_model , corpus, id2word):
    """
    Visualize the LDA Mallet topics for the optimal model:
    1) show terms per topic dataframe
    2) wordcloud of the topics
    3) topic visualization with pyLDAvis
    
    """
        
    # activate the following line to display topics
#     pprint(optimal_model.show_topics(formatted=False))
    
    
    topics=[[(term, round(wt, 3)) 
             for term, wt in optimal_model.show_topic(n, topn=100)] 
            for n in range(0, optimal_model.num_topics)]
    #pprint( topics) # display 20 top words rounded in 3 digits
    
    
    #dataframe of topics and top 10 words
    topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics],
                         columns = ['Terms per Topic'], 
                         index=['Topic'+str(t) for t in range(1, optimal_model.num_topics+1)] )
    
    display(topics_df)
    
    # visualize the topics as wordclouds
    #reference:https://medium.com/swlh/topic-modeling-lda-mallet-implementation-in-python-part-2-602ffb38d396

    # initiate wordcloud object
    wc = WordCloud(background_color="white", 
                   colormap="Dark2",
                   max_font_size=150,
                   random_state=42)
    #no_normalize_plurals='False',
    #no_normalize_plurals:whether to remove trailing ‘s’ from words
    #relative_scaling default is auto=0.5 :scaling of words by frequency (0 - 1)

    # set the figure size
    plt.figure(figsize=(6,17), dpi=180)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)


    # Create subplots for each topic
    for i in range(optimal_model.num_topics):
        plt.plot()

        wc.generate(text=topics_df["Terms per Topic"][i])
        
        plt.subplot(12, 3, i+1 )
        plt.imshow(wc,  interpolation='bilinear')
        plt.axis("off")
        plt.title(topics_df.index[i])
    
    plt.tight_layout()
    plt.show()
#     print()# line
    
    
    # visualization with pyLDAvis

    #convert optimal_model (which is a Mallet model) to a gensim model

    def convertldaMalletToldaGen(mallet_model):
        model_gensim = LdaModel(
            id2word=mallet_model.id2word,
            num_topics=mallet_model.num_topics,
            alpha=mallet_model.alpha) 
        model_gensim.state.sstats[...] = mallet_model.wordtopics
        model_gensim.sync_state()
        return model_gensim

    optimal_model_gensim = convertldaMalletToldaGen(optimal_model)
    vis_data = gensimvis.prepare(optimal_model_gensim, corpus, id2word, sort_topics=False)
    return vis_data

