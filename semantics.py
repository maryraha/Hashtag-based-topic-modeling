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
SEMANTICS MODULE

Author: MaryRaha
Created June 29 2020

1) Create hashtag documents from a dataset of hashtaged tweets
2) Create tfidf_matrix
3) Use cosine and soft cosine similarity measurementsand to find similar hashtag (docs) 

"""
import pandas as pd
import numpy as np
import swifter
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction import stop_words
from sklearn import metrics
from sklearn.metrics.pairwise import linear_kernel #for cosine similarities
import gensim
from gensim.matutils import softcossim 
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity




def hashtag_docs(dfh_reduced, nodes):
    """ 
    create and save hashtag documents by combining the tokenized tweets 
    for each hashtag    
    """ 
    hashdocs=dfh_reduced[['hashtags', 'tokenized','preprocessed','spacy_lemmatized']].copy()
    
    hashdocs=hashdocs.drop_duplicates()# remove duplicate rows of hashtag-list of tokens
    #this will remove case of tweets with same body and cases that a hashtag has appeared more than one time in a tweet
    
    
    #aggregate the tweets for each hashtag, count shows the number of tweets with each hashtag,
    #aggregate the tokenized and preprocessed tex bodies of the tweets with each hashtag
    hashdocs=hashdocs.assign(count=1).groupby(['hashtags']).agg({'count':'sum', 
                                                                 'tokenized':lambda x : ','.join(set(x)),
                                                                 'preprocessed':lambda x : ','.join(set(x)),
                                                                 'spacy_lemmatized':lambda x : ','.join(set(x))})
   
    col_name= 'text_h_in' #options are 'text_h_in ,'text_h_out' , 'text_neh'
    #convert list of tokens and tweets to string
    def string_prep(df1, col_name):
#         df1[col_name]=df1[col_name].apply(lambda x: re.sub('\s+', ' ', x)) #remove double spaces
        df1[col_name]=df1[col_name].apply(lambda x: "".join(x))
        df1[col_name]=df1[col_name].apply(lambda x: x.replace('][',','))
        df1[col_name]=df1[col_name].apply(lambda x: x.replace('[',''))
        df1[col_name]=df1[col_name].apply(lambda x: x.replace(']',''))
        df1[col_name]=df1[col_name].apply(lambda x: x.replace("'",''))
        return df1
    hashdocs=string_prep(hashdocs,'tokenized')
    hashdocs=string_prep(hashdocs,'preprocessed')
    
    def spacy_string_prep(df1, col_name):
        df1[col_name]=df1[col_name].apply(lambda x: x.replace('[',''))
        df1[col_name]=df1[col_name].apply(lambda x: x.replace(']',''))
        df1[col_name]=df1[col_name].apply(lambda x: x.replace("'",''))
        df1[col_name]=df1[col_name].apply(lambda x: x.split(","))    
        return df1
    hashdocs=spacy_string_prep(hashdocs,'spacy_lemmatized')

    
    hashdocs=hashdocs.reset_index()
    #Insert hashtag Ids from nodes.csv
    hashdocs.rename(columns={'hashtags':'Name'}, inplace=True)
    hashdocs=pd.merge(hashdocs,nodes[['Name','Id']] , on='Name', how='left')
    hashdocs.to_csv('data files/hashdocs.csv', index=False)
    hashdocs['tokenized']=hashdocs['tokenized'].apply(lambda x: re.sub('\s+', ' ', x)) #remove double spaces
    print('hashdocs.csv file is successfully created!')
    return hashdocs 





def DTM_TFIDF(hashdocs , col_name2 , ngram_range_v , max_df_v, max_features_v , token_pattern_v):
    """
    TF-IDF normaliation using TfidfVectorizer
    create a "list" of all aggregated hashtag documents and save it as corpus 
    then apply TF-IDF normalization and row-wise euclidean normalization all by TfidfVectorizer.
    
    - use the defined tokenizer       
    - use a scipy.sparse matrix to store the features and efficiently handle sparse matrices.
    - max_df to filters out too frequent terms. e.g max_df=0.8 filters out the terms that happen in 80% of the documents 
    - max_features:If not None, build a vocabulary that only consider the top max_features ordered by term frequency across
      the corpus.
    
    Note: As we are working with aggregated tweets here min_df is not a good choice.
    min_df=2 filteres out terms that happen in 2 or less than 2 documents. 
    
    Note:column of the tfidf_ matrix is based on the alphabetic order of the terms
  
    """
    #define Tfidf Vectorizer:
    tfidf = TfidfVectorizer(
        #tokenizer=tokenize,# as col_name2='tokenized' the input text is already toenized 
        analyzer='word', # features made of words
        ngram_range=ngram_range_v,
        use_idf=True,# enable inverse-document-frequency reweighting
#         smooth_idf=True,  # prevents zero division for unseen words
        max_df=max_df_v, 
        max_features= max_features_v,
        stop_words='english',
        #min_df=1,  # min count for relevant vocabulary
        #strip_accents='unicode',  # replace all accented unicode char 
        # by their corresponding  ASCII char 
        token_pattern= token_pattern_v,  
        lowercase=False,
        sublinear_tf=True)
    
    #send in all the aggregated tweets:
    # creates a list of aggregated tweets for each hashtag that are in form of comma-separated tokens
    #e.g. ['aggregated hashtag doc', 'aggregated hashtag doc2',...]
    corpus = hashdocs[col_name2].values.tolist()
    labels = hashdocs['Name'].values.tolist()
    tfidf_matrix = tfidf.fit_transform(corpus)#tfidfs wighted doc-term matrix
    feature_names = tfidf.get_feature_names() #list of the terms or column headers of the tfidf matrix 
    #tfidf_matrix=tfidf_matrix.astype('float16') # to change the tfidf_matrix from float64 to float16
    print ('The TFIDF matrix is successfully created!\nThe Document-Term Matrix (DTM) with TF-IDF weights has',
           format(tfidf_matrix.shape[0],','),'hashtag documents (in rows), and',format(tfidf_matrix.shape[1],','),
           'terms (in columns).')
    #return feature_names, corpus, labels, tfidf_matrix
    return tfidf_matrix , feature_names, labels, corpus


 
    
    
def hashdocs_tfidf(ngram_range_v, max_df_v, max_features_v , token_pattern_v ):
    """   
    This function do the following tasks:
    1) create and save hashtag documents (by combining the tweet texts in the column specifed by col_name2 for each hashtag)
    2) create tfidf_matrix for the hashtag documents 
    """
    col_name2='tokenized' # options are 'tokenized', 'preprocessed','spacy_lemmatized'
    
    dfh_reduced= pd.read_csv('data files/dfh_unpoprmvd.csv')
    nodes= pd.read_csv('data files/nodes.csv')
    
    #create hashtag document considering hashtag_in_tokenized
    hashdocs=hashtag_docs(dfh_reduced, nodes)
    tfidf_matrix , feature_names, labels, corpus= DTM_TFIDF(hashdocs , col_name2 , ngram_range_v, max_df_v, max_features_v , token_pattern_v)

    return tfidf_matrix , feature_names, labels, corpus  





def similarity_edges(sim_matrix, similarity_threshold, similarity_edges_file):
    """    
    input:
        similarity matrix
        similarity_threshold is the cosine similarity cut off point specified by the user 
  
    output:
        create and save all edges created based on hashtag similarities in the path specified     
    """
    hashdocs=pd.read_csv('data files/hashdocs.csv')    
    nodes=pd.read_csv('data files/nodes.csv')    

    #create a similarity_table with all possible undirected edges in rows excluding self loops 
    # the number of rows= ((total number of hashtags)^2-(total number of hashtags))/2
    cl_Ids= hashdocs['Id']
    row_Ids= hashdocs['Id']
    # to address cosine similarities by hashtag ids
    sim_matrix=pd.DataFrame(sim_matrix, columns=cl_Ids, index=row_Ids)

    all_pairs=[(i,j) 
               for i in cl_Ids
               for j in row_Ids if (i>j)] 
    # similarity matrix is a symmetric matrix, we consider only the upper triangular 
    #because the graph generated by edge.csv is an undirected graph and only has one of the a-b or b-a links between the edges
    #depending on the edge construction pattern and by filtering that we can find semantic-based new links.


    similarity_table= pd.DataFrame(data=all_pairs , columns=('hashtags_x','hashtags_y'))
    similarity_table['cosine_sim']=similarity_table.swifter.apply(lambda x: sim_matrix[x.hashtags_x][x.hashtags_y] , axis=1)
    if similarity_table.shape[0]==((nodes.shape[0] * nodes.shape[0])-nodes.shape[0])/2:
        print('the cosine similarity table is successfully created!\n')
    else:
        print('a problem has happened!')


    #apply the similarity threshold    
    similarity_links=similarity_table[similarity_table.cosine_sim>= similarity_threshold]
    print('a total of', format(similarity_links.shape[0],','),
        'hashtag connections are created based on semantic similarity of the hashtag docs.\n ')



    similarity_links=similarity_links.rename(columns={'hashtags_x':'Source','hashtags_y':'Target', 'cosine_sim':'Weight'}).copy()
    similarity_links['Type']='Undirected'       
    # semantically related hashtags get large weights relative to their cosine similarity score
    similarity_links.to_csv(similarity_edges_file, index=False)
       


    

# -


