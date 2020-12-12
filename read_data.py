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

# + code_folding=[]
"""
READ_DATA MODULE

Author: MaryRaha
Created June 29 2020

This is a module to:
1) read the tweets extracted from TCAT-DMI, 
2) optimize the memory usage,
3) extract English tweets
4) explore the characteristics of the data
5) create and save data1.csv and data2.csv
  - data1: dataset of unique english tweets (no retweet)
  - data2: data2: dataset of unique English tweets 
    in which quote tweets and replies are enriched by aggregating them with the text of their original tweet. 
    
"""
import pandas as pd
import numpy as np
import re
from contractions import CONTRACTION_MAP  #file contractions.py is added to the repository
import swifter
import nltk
import os
from tabulate import tabulate
from data_opt import optimized_data # to optimize the memory usage
from datetime import datetime
import preprocessing # preprocessing module
import matplotlib.pyplot as plt
# %matplotlib inline


    
def en_tweets(data): 
    """Extract English Tweets
    
    it uses lang column (which is the the language detected by twitter API) to filter out non english tweets
    output: 
        english tweets (tweets with lang=en)
        total_tweets: total number of tweets  
        total_en_tweets: total number of english tweets      
    """
    total_tweets= data.shape[0]
    data=data[data.lang =='en']
    total_en_tweets=data.shape[0]
    #print(f'This data set has {total_tweets} tweets in total which includes {total_en_tweets} english tweets. \n')
    data=data.drop('lang', axis=1)
    return data, total_tweets , total_en_tweets



def explore_data(data, total_tweets , total_en_tweets):
    """
    create retweet column with 0 and 1 values
    fill the empty columns with 0
    create a table to show different characteristics of the dataset
    
    """    
    def find_RT(row): #it detects tweets that start with RT
        if row.startswith('RT'):
            return 1
        return 0
    data['retweet']=data['text'].apply(lambda row: find_RT(row)) # a new column as retweet is created to specify if a tweet starts with RT 
    
    data.loc[:,'in_reply_to_status_id'] = data.loc[:,'in_reply_to_status_id'].fillna(0)
    data.loc[:,'quoted_status_id'] = data.loc[:,'quoted_status_id'].fillna(0)
    data.loc[:,'truncated'] = data.loc[:,'truncated'].fillna(0)
    data.loc[:,'lat'] = data.loc[:,'lat'].fillna(0)
    
    headers = ["Characteristics", "Value"] 
#     table = [
#         ["Number of columns", len(data.columns)],
#         ['Tweets', format(total_tweets,',')],
#         ['English tweets', format(total_en_tweets,',')],
#         ['--Retweets (RT)', format(len(data[data.text.str.startswith('RT')]),',')],
#         ['--Unique tweets (not RT)', format(total_en_tweets-len(data[data.text.str.startswith('RT')]),',')],
#         ['--Replies', format(data.shape[0]-len(data.loc[data['in_reply_to_status_id']==0]),',')],
#         ['--Quoting tweets', format(data.shape[0]-len(data.loc[data['quoted_status_id']==0]),',')],
#         ['--RT & quoting', format(len(data.loc[(data['retweet'] == 1) & (data['quoted_status_id'] != 0)]),',')],
#         ["--RT & reply", format(len(data.loc[(data['retweet'] == 1) & (data['in_reply_to_status_id'] != 0)]),',')], 
#         ['--RT & not quoting', format(len(data.loc[(data['retweet'] == 1) & (data['quoted_status_id'] == 0)]),',')],
#         ['--not RT & quoting', format(len(data.loc[(data['retweet'] == 0) & (data['quoted_status_id'] != 0)]),',')],
#         ['--not RT & not quoting & reply', format(len(data.loc[(data['retweet'] == 0) &(data['quoted_status_id'] == 0) & (data['in_reply_to_status_id'] !=0)]),",")],
#         ['--not RT & quoting & not reply', format(len(data.loc[(data['retweet'] == 0) & (data['quoted_status_id'] != 0) &(data['in_reply_to_status_id'] ==0)]),',')],
#         ["--not RT & not quoting & not reply", format(len(data.loc[(data['retweet']== 0) & (data['quoted_status_id']== 0) & (data['in_reply_to_status_id'] ==0)]),',')], 
#         ['--not RT & quoting & reply', format(len(data.loc[(data['retweet'] == 0) & (data['quoted_status_id'] != 0) & (data['in_reply_to_status_id'] !=0)]),",")],
#         ['--RT & quoting & reply', format(data.shape[0]-len(data.loc[(data['retweet'] == 0) & (data['quoted_status_id'] == 0) & (data['in_reply_to_status_id'] ==0)]),',')],
#         ["--Truncated tweets", format(data.loc[data['truncated']==1 ].shape[0],',')], 
#         ['--Tweets with a link referring to a sensitive content', format(data.loc[data['possibly_sensitive']== 1].shape[0],',')],
#         ['--Tweets that their content is removed due to copyright', format(data.loc[data['withheld_copyright']== 1].shape[0],',')],
#         ['--Tweets with user specified location text', format(data.shape[0]-len(data.loc[data['location'].astype(str)=='nan']),",")],
#         ['--GeoTagged tweets', format(data.shape[0]-len(data.loc[data['lat']==0]),',')],
#         ["--Tweets issued by users with verified accounts", format(len(data.loc[data['from_user_verified']==1]),',')]
#     ]
    
    table = [
        ['Tweets', format(total_tweets,',')],
        ['English Tweets', format(total_en_tweets,',')],
        ['Unique English Tweets (not retweet)', format(total_en_tweets-len(data[data.text.str.startswith('RT')]),',')]
    ]
    print(tabulate(table, headers, tablefmt="fancy_grid"))
    print()
    return data 



def remove_retweets(data): 
    """
    removes retweets and saves the result dataset in data1.csv
    """
    data1=data[~data.text.str.startswith('RT')][['id','created_at','text','quoted_status_id','retweet', 
                                                 'in_reply_to_status_id','hashtags','mentions']] 
    #data1: dataset with no retweet (unique english tweets)   
    return data1



def enrich_tweets( data, data1): 
    """enrich replies and quotes that are not started with RT and saves the result dataset in data2.csv
    relies and quotes texts are enriched by aggregating them with the text of the tweets that they are replying or quoting
    
    it creates and combines three datasets:
    df0_q: dataset of enriched quote tweets that are not RT 
    df0_r: dataset of enriched replies
    df0_qr: dataset of enriched replies that are quoted (to exclude tweets that belong to the first two dataset)
    """
    #df0_q: dataset of enriched quote tweets that are not RT
    df01=data.loc[(data['quoted_status_id']!=0 )& (data['retweet']==0),
                  ['id','created_at','text','quoted_status_id','retweet', 'in_reply_to_status_id',
                   'hashtags','mentions']]
    
    df01.rename(columns={'id': 'id0','quoted_status_id':'id', 'hashtags':'hashtags0', 
                         
                         'mentions':'mentions0'} , inplace=True)
    #vlookup df01 and the original data on index columns 
    df01['id']=df01['id'].astype('int64')# since quoted_status_id is float64 and id is int64. for lookup they must have the same type
    df0_q=pd.merge(data1[['id','text','hashtags','mentions']], df01, on=['id'] , how='right' )
    df0_q['text_x'] = df0_q['text_x'].replace(np.nan, '', regex=True) 
    df0_q['text'] = df0_q[['text_x', 'text_y']].astype('str').apply(lambda x: ' '.join(x) , axis=1) # column'text' is the enriched text      
   
    #aggregate the mentions and hashtags as well
    df0_q['hashtags']= df0_q[['hashtags','hashtags0']].apply(lambda x: x.str.cat(sep='; '),
                                                             axis=1).replace(r'^\s*$', np.nan, regex=True)
    
    df0_q['mentions']= df0_q[['mentions','mentions0']].apply(lambda x: x.str.cat(sep='; '),
                                                             axis=1).replace(r'^\s*$', np.nan, regex=True)
    
    df0_q.rename(columns={'id': 'quoted_status_id','id0':'id'}, inplace=True)
    df0_q = df0_q.drop(['text_x','text_y','hashtags0','mentions0'], axis=1)    
    # df0_r: dataset of enriched replies 
    df02=data.loc[ (data['in_reply_to_status_id']!=0),['id','created_at','text','quoted_status_id',
                                                       'retweet', 'in_reply_to_status_id','hashtags','mentions']]
    
    df02.rename(columns={'id': 'id0','in_reply_to_status_id':'id', 'hashtags':'hashtags0',
                         'mentions':'mentions0'} , inplace=True)
    #vlookup df01 and the original data on index columns 
    df02['id']=df02['id'].astype('int64')# since quoted_status_id is float64 and id is int64. for lookup they must have the same type
    df0_r=pd.merge(data1[['id','text','hashtags','mentions']], df02, on=['id'] , how='right' )
    df0_r['text_x'] = df0_r['text_x'].replace(np.nan, '', regex=True)
    df0_r['text'] = df0_r[['text_x', 'text_y']].astype('str').apply(lambda x: ' '.join(x), axis=1) # column'text' is the enriched text
    
    #aggregate the mentions and hashtags as well
    df0_r['hashtags']= df0_r[['hashtags','hashtags0']].apply(lambda x: x.str.cat(sep='; '),
                                                             axis=1).replace(r'^\s*$', np.nan, regex=True)
    df0_r['mentions']= df0_r[['mentions','mentions0']].apply(lambda x: x.str.cat(sep='; '),
                                                             axis=1).replace(r'^\s*$', np.nan, regex=True)
    df0_r.rename(columns={'id': 'in_reply_to_status_id','id0':'id'}, inplace=True)
    df0_r = df0_r.drop(['text_x','text_y','hashtags0','mentions0'], axis=1)    
    #df0_qr: dataset of enriched replies that are quoted
    df0_qr=pd.merge(df0_q, df0_r, on=['id'] , how='inner' )
    df0_qr['text_x'] = df0_qr['text_x'].replace(np.nan, '', regex=True)
    df0_qr['text']= df0_qr[['text_x', 'text_y']].astype('str').apply(lambda x: ' '.join(x), axis=1) # column'text' is the enriched text
    
    df0_qr['hashtags']= df0_qr[['hashtags_x','hashtags_y']].apply(lambda x: x.str.cat(sep='; '),
                                                                  axis=1).replace(r'^\s*$', np.nan, regex=True)
    
    df0_qr['mentions']= df0_qr[['mentions_x','mentions_y']].apply(lambda x: x.str.cat(sep='; '), 
                                                                  axis=1).replace(r'^\s*$', np.nan, regex=True)
    
    df0_qr = df0_qr.drop(['text_x','text_y','hashtags_x','mentions_y', 'hashtags_y', 
                          'mentions_x','quoted_status_id_y','retweet_x','in_reply_to_status_id_x'], axis=1)
    
    df0_qr.rename(columns={'retweet_y':'retweet' , 'quoted_status_id_x':'quoted_status_id',
                           'in_reply_to_status_id_y':'in_reply_to_status_id'}, inplace=True)
    # combine df0_r, df0_q, df0_qr = the dataset of enriched replies and quoted tweets
    df0_rr=df0_r[~df0_r.id.isin(df0_qr.id)]
    df0_qq=df0_q[~df0_q.id.isin(df0_qr.id)]
    df0_concat=pd.concat([df0_rr,df0_qq,df0_qr])
    if len(df0_concat)== (len(data1.loc[(data1['quoted_status_id'] != 0)])+
                          len(data1.loc[(data1['quoted_status_id'] == 0) & (data1['in_reply_to_status_id'] !=0)])):
        print('data1.csv file is successfully created and saved!')
        print('data2.csv file is successfully created and saved!\n')
    else:
        print('A problem has happened!')
    
    # data2: dataset with no retweets in which replies and quoted tweets are enriched with their referenced tweets
    df0_concat0 = df0_concat.set_index('id')
    data2 = data1.set_index('id')
    data2.update(df0_concat0)
    data2.reset_index(inplace=True)
    
    data1= data1[['created_at','text','hashtags','mentions']].copy() # because later we only need there two columns
    #extract date from timestamp
    data1["created_at"] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S').strftime('%m/%d/%Y') for d in data1['created_at']]
    data1.to_csv('data files/data1.csv', index=False)
    
    data2= data2[['created_at','text','hashtags','mentions']].copy() # because later we only need there two columns
    #extract date from timestamp
    data2["created_at"] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S').strftime('%m/%d/%Y') for d in data2['created_at']]
    data2.to_csv('data files/data2.csv', index=False)
    
    return data2

def tweet_char_plot(data1, data2):
    """
    1) counting number of characters per tweet for data1 and data2
    2) show them on a plot
    
    """ 
    print('Compare character length of the tweets in Data1 and Data2: first remove link, images, and mentions from the tweet texts: ')
    # first preprocess the text of the tweets to remove link , mentions,.. 
    data1['text_proc'] = data1['text'].swifter.apply(lambda x: preprocessing.text_preprocessing(x))
    data1['textlen']=data1['text_proc'].apply(lambda x: len(x.split()))
    data1['ID'] = data1.index 
    d1len=pd.DataFrame(data1[['textlen','ID']].groupby('textlen')['ID'].unique())
    d1len['Count']=data1[['textlen','ID']].groupby('textlen').count()# count frequency of each textlen
    d1len['textlen']=d1len.index
    
    data2['text_proc'] = data2['text'].swifter.apply(lambda x: preprocessing.text_preprocessing(x))
    data2['textlen']=data2['text_proc'].apply(lambda x: len(x.split()))
    data2['ID'] = data2.index 
    d2len=pd.DataFrame(data1[['textlen','ID']].groupby('textlen')['ID'].unique())
    d2len['Count']=data2[['textlen','ID']].groupby('textlen').count()# count frequency of each textlen
    d2len['textlen']=d2len.index
    
    plt.subplots( figsize=(10, 5)) 
    plt.plot(d2len['textlen'],d2len['Count'], color='red', label='Data2')
    plt.plot(d1len['textlen'],d1len['Count'], color='blue', label='Data1')
    plt.xlabel('number of words per tweet') 
    plt.ylabel('tweet frequency')
    plt.title('word distribution per tweet for the two datasets')
    plt.legend(loc='upper right')
    plt.show()
    
    #boxplot (for both data1 and data2)
    newdata2 = d2len['textlen'].repeat(d2len['Count'])
    newdata1 = d1len['textlen'].repeat(d1len['Count'])
#     data=[newdata1,newdata2]
#     fig = plt.figure(figsize =(10,5)) 
#     ax = fig.add_subplot(111) 
#     # Creating axes instance 
#     bp = ax.boxplot(data, patch_artist = True, notch ='True', vert = 0 ,
#                     meanline=True, showmeans=True) 
#     colors = ['#00FF00', '#FFFF00'] 
#     for patch, color in zip(bp['boxes'], colors): 
#         patch.set_facecolor(color) 
#     # changing color and linewidth of whiskers 
#     for whisker in bp['whiskers']: 
#         whisker.set(color ='#8B008B', 
#                     linewidth = 1.5, 
#                     linestyle =":") 
#     # x-axis labels 
#     ax.set_yticklabels(['data_1', 'data_2']) 
#     # Adding title  
#     plt.title("Boxplot of the number of words per tweet for the two datasets") 
#     plt.show(bp)
      
    #boxplot
#     fig1, ax1 = plt.subplots(figsize=(10, 5))
# #     ax1.set_title('Basic Plot')
# #     ax1.set_xticklabels(['Data'])
#     ax1.boxplot(newdata2, vert = 0, showmeans=True, meanline=True)
#     plt.show()

def create( datafile_name, use_cols , category_cols):
    
    # read and optimize the data
    data= optimized_data(datafile_name, use_cols, category_cols)

    #extract English tweets
    data, total_tweets , total_en_tweets= en_tweets(data)

    # explore the data
    data= explore_data(data, total_tweets , total_en_tweets)

    #create and save data1.csv
    data1= remove_retweets(data)

    #create and save data2.csv
    data2= enrich_tweets(data, data1)

    tweet_char_plot(data1, data2)

# -


