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
HASHTAG ANALYSIS MODULE

Author: Maryam Rahafrooz
Created July 5 2020

This is a module for analyzing the Twitter hashtags.

1) Extract the tweets with hashtags
2) create the dataset of hashtags, and listthe unique hashtags
3) Remove Unpopular hashtags
4) Draw hashtag-related plots to explore the distribution of hashtags, the number of hashtags per tweets,
and the relationship between the number of hashtags and the existence of a URL in tweets

"""

import numpy as np
import pandas as pd
from urlextract import URLExtract # to extract url from texts
from tabulate import tabulate
import matplotlib.pyplot as plt



def tweets_with_hashtags(df ):
    """ extract tweets with hashtags from the dataset of all tweets
    input: df is the dataframe of all tweets
           path to save dataset of all tweets
           
    output: dfh is the dataframe of all tweets that include hashtags
            total_hashtweets is the count of tweets in this dataset that include at least one hashtag
    
    dataset of all tweets with hashtags will be saved in path specified as dfh.csv
    """
    col_name = 'text_h_in'
    col = [2] # the column of df that includes hashtags
    dfh = df[df.columns[col]].dropna()  #remove rows with NaN hashtags
    dfh['hashtags'] = dfh['hashtags'].str.split(';')
    dfh[[ col_name , 'tokenized','preprocessed','spacy_lemmatized']]=df[[col_name , 'tokenized', 'preprocessed','spacy_lemmatized']]
    total_hashtweets = dfh.shape[0] # number of tweets with hashtags in this datase
    # save the dataset of all tweets with hashtags before filtering out unpopular hashtages
    dfh.to_csv('data files/dfh.csv', index=False)
    
    # number of hashtags: h_humber , h_ne_number
    if (col_name=='text_h_in'or col_name=='text_h_out'): 
        print('This preprocessed dataset has a total of', format(df['h_number'].sum(),','),'(repetetive) hashtags.')
    else:
        print('This preprocessed dataset has a total of', format(df['h_number'].sum(),','),'(repetetive) hashtags out of which ',
              format(df['h_ne_number'].sum(),','),'hashtags are non-end hashtags.\n',)

    return dfh, total_hashtweets 



def explode(dfh, lst_cols, fill_value='', preserve_index=True):
    """ 
    explode all hashtags of each tweet on rows while preserving the TweetID
    input: dfh (dataframe of tweets with hashtags) 
           lst_cols (column of hashtags that want to explode) 
    """
    if (lst_cols is not None and len(lst_cols) > 0
            and not isinstance(lst_cols,
                               (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    idx_cols = dfh.columns.difference(lst_cols)  # all columns except lst_cols
    lens = dfh[lst_cols[0]].str.len()  # lengths of lists
    idx = np.repeat(dfh.index.values, lens)  # preserving original indexes
    res = (pd.DataFrame(
        {col: np.repeat(dfh[col].values, lens)
         for col in idx_cols}, index=idx).assign(
             **{
                 col: np.concatenate(dfh.loc[lens > 0, col].values)
                 for col in lst_cols
             }))
    if (lens == 0).any():  # append rows with empty lists
        res = (res.append(dfh.loc[lens == 0, idx_cols],
                          sort=False).fillna(fill_value))
        res = res.sort_index()  # revert the original index order
    if not preserve_index:  # reset index if requested
        res = res.reset_index(drop=True)
    return res



def hashtag_edit(hashtag):
    """ normalize hashtags 
    input: a hashtag

    1. lower casing the hashtags to find unique hashtags
       Twitter hashtags are not case sensitive (uppercasing only helps with readability of the hashtags)
   
    2. remove leading or ending spaces from hashtags
       as some hashtags that are extracted by TCAT have leading or ending spaces
    """
    hashtag = hashtag.lower()
    hashtag = hashtag.strip()  
    return hashtag



def hashtag_dataframe(df, n ):
    """create a hashtag dataframe and save it in the specified path,
    then filter out hashtags with less than n count in the corpus of the tweets
      
    input: df is the dataframe of all tweets
           path to save dataset of all tweets 
           n is the hashtag count to filter out unpopular hashtags
    note: if n=0 then no hashtag is filtered out
           
    output: dataset of all tweets with hashtags is saved in dfh.csv 
            total number of tweets with hashtags is saved in total_hashtweets
            total number of hashtags is saved in total_hashtags
            dfh is the dataset of all hashtags with their tweet id and tweet message in each row
            dfh_reduced is the dataset of all hashtags (with at least n count) with their tweet id and tweet message in each row
            dfhu is the dataset of all unique hashtags with list of tweets they occurred in
            dfhu_reduced is the dataset of all unique hashtags with at least n count
            
            dfh.csv: all tweets with hashtags
            dfh_unpoprmvd.csv: hashtag-tweet each row for all tweets with hashtags after removing unpopular hashtags
            
    """
    dfh , total_hashtweets= tweets_with_hashtags(df ) # create dataset of all tweets with hashtags
    dfh = explode(dfh, ['hashtags'], preserve_index=True) #explode hashtags in rows
    dfh['hashtags'] = dfh['hashtags'].apply(lambda x: hashtag_edit(x)).astype(str)
    dfh['TweetID'] = dfh.index # create TweetID
    dfh.reset_index(drop=True, inplace=True)
    total_hashtags=dfh.shape[0]# total number of (repetetive) hashtags in the corpus
    dfhu=pd.DataFrame(dfh[['hashtags','TweetID']].groupby('hashtags')['TweetID'].unique()) #find unique hashtags and define dfhu
    dfhu['Count']=dfh[['hashtags','TweetID']].groupby('hashtags').count()# count frequency of each hashtag
    total_uniqhashtags=dfhu.shape[0]# total number of unique hashtags
    dfhu_reduced=dfhu[dfhu.Count>=n]# to keepfilter out #s with less than n count
    dfh_reduced = dfh[dfh.hashtags.isin(dfhu_reduced.index)]#to extract tweets related to hashtags with at least n frequency
    dfh_reduced.to_csv( 'data files/dfh_unpoprmvd.csv', index=False)
    total_hashtags_reduced=dfh_reduced.shape[0]# total number of (repetetive) hashtags after removing npopular hashtags
    total_uniqhashtags_reduced=dfhu_reduced.shape[0]# total number of (repetetive) hashtags after removing npopular hashtags    
    return dfhu_reduced, dfh_reduced, total_hashtweets, total_hashtags , total_uniqhashtags ,total_hashtags_reduced, total_uniqhashtags_reduced 

   

    
def graph_nodes(dfhu_reduced):
    """create nodes.csv file for graph generation
    input: dfh_reduced which is the dataset of all unique hashtags with at least n count 
    note: nodes is created based on the unpoprmvd file
    """
    nodes=dfhu_reduced.sort_values(by=['Count'], ascending=False)
    nodes=nodes.reset_index()
    nodes['Id']=(nodes.index)+1 # to create id for the nodes
    nodes['Label']=nodes['hashtags'].astype(str)
    nodes['Name']=nodes['hashtags'].astype(str)
    nodes = nodes[['Id','Name', 'Label', 'Count']] # to import nodes.csv into gephi it should have this format
    print("The nodes.csv file is successfully created! It includes", format(nodes.shape[0],','), "nodes.")
    nodes.to_csv('data files/nodes.csv', index=False)
    return nodes




def graph_edges(dfh_reduced, nodes ):
    """creates edges.csv file for graph generation 
    and uses noise_threshold to remove noisy edges    
    output: edges.csv and edge_noise_removed     
    note: edges is created based on the unpoprmvd file
    """
    ed0 = dfh_reduced[['hashtags','TweetID']].merge(dfh_reduced[['hashtags','TweetID']], on='TweetID', how='left')
    edge = ed0[~ed0[['hashtags_x', 'hashtags_y','TweetID']].apply(frozenset, axis=1).duplicated()] # to make it undirected, so that don't add both fentanyl-diazpam and diazpam-fentanyl for one TweetID
    edge = edge[edge['hashtags_x']!=edge['hashtags_y']]# remove self loops    
    
    #Vlookup to create the edges.csv in the needed format and save it 
    edge.rename(columns={'hashtags_x':'Name', 'hashtags_y':'Label'}, inplace=True)
    edge=pd.merge(edge,nodes[['Name','Id']] , on='Name', how='left')
    edge.rename(columns={'Id':'Source'}, inplace=True)
    edge=pd.merge(edge,nodes[['Label','Id']] , on='Label', how='left')
    edge.rename(columns={'Id':'Target'}, inplace=True)
    edges=edge[['Source','Target']].copy()
    edges['Type'] = 'Undirected'
    edges['Weight'] = 1
    edges.to_csv('data files/edges.csv', index=False)
    print("The edges.csv file is successfully created! The total number of hashtag co-ocurrences in this dataset is", 
          format(edges.shape[0],','),'.\n')    
    return edges #,edge_noise_removed




def hashtag_plots(df, nodes):
    """Hashtag-url relationship
    
    to explore the relationship between the number of hashtags and
    existance of url in the tweets
    
    input:df which the dataset of cleaned tweets
          node which is dataset of all unique hashtags representing hashtag graph nodes
    output:
    """
    #figure-1: hashtag distribution
    fig, ax=plt.subplots(figsize=(10, 5))
    ax.plot(nodes['Id'], nodes['Count'], marker='*', color='blue')
    ax.set(xlabel='Hashtag Id', ylabel='Hashtag frequency',
           title='Fig-1: Distribution of hashtags for this dataset (logarithmic scale)')
    ax.set_yscale('log') #y axis in logarithmic scale 
    ax.set_xscale('log') #x axis in logarithmic scale
    #fig.savefig("test.png")
    plt.show()     
    
    extractor = URLExtract() #to extract urls from tweet texts
#     df['urls'] =pd.read_csv(os.path.join('python files',dataset))['text'].apply(lambda x: extractor.find_urls(x)) 
    df['urls'] = df['text'].apply(lambda x: extractor.find_urls(x))
    
    #create df_hash1 
    df_hash1=df[['hashtags', 'urls', 'h_number']].copy() #h_number is number of hashtags in each tweet
    df_hash1['hashtags']= df_hash1['hashtags'].str.split(';')
    df_hash1['hashtags'] = df_hash1['hashtags'].apply(lambda d: d if isinstance(d, list) else []) #to replace NaN with empty list
    #df_urlhash['hash_count']=df_urlhash['hashtags'].apply(len)
    df_hash1['url_count']=df_hash1['urls'].apply(len)
    df_hash1['url']=df_hash1['url_count'].astype(bool).astype(int) #to specify if tweet has at least one url
    df_hash1['TweetID'] = df_hash1.index
    df_hash1.reset_index(drop=True, inplace=True)
    
    # group tweets based on the number of hashtags
    df_hash2=pd.DataFrame(df_hash1.groupby('h_number')['TweetID'].unique())
    df_hash2['number_of_tweets']= df_hash2['TweetID'].apply(lambda x: len(x))
    df_hash2['hash_count'] = df_hash2.index
    df_hash2.reset_index(drop=True, inplace=True)
    
    # compute number of totall urls
    for i in range(df_hash2.shape[0]):
        df_hash2.loc[i,'url_total_count']=df_hash1[df_hash1['TweetID'].isin(df_hash2.loc[i,'TweetID'])]['url_count'].sum()
        #for each group of tweets in hash2 dataset we sum up the number of corresponding url_count in hash dataset
        df_hash2.loc[i,'url_total_withurl']=df_hash1[df_hash1['TweetID'].isin(df_hash2.loc[i,'TweetID'])]['url'].sum()
        df_hash2.loc[i,'url_percent_withurl']= df_hash2.loc[i,'url_total_withurl']/df_hash2.loc[i,'number_of_tweets']
  
    #figure-2: Number of hashtags per tweet    
    fig, ax=plt.subplots(figsize=(10, 5))
    ax.plot(df_hash2['hash_count'],df_hash2['number_of_tweets'], marker='*', color='purple')
    ax.set(xlabel='Hashtag frequency per tweet', ylabel='log number of tweets',
                title='Fig-2: Number of hashtags per tweet')
    ax.set_yscale('log') #y axis in logarithmic scale
    #fig.savefig("test.png")
    plt.show()
    
#     fig, axes=plt.subplots(nrows=1, ncols=2)
#     axes[0].plot(nodes['Id'],nodes['Count'], marker='*', color='blue')
#     axes[0].set(xlabel='Hashtag Id', ylabel='Hashtag frequency',
#                 title='Fig-1: Distribution of hashtags for this dataset (logarithmic scale)')
#     axes[1].plot(df_hash2['hash_count'],df_hash2['number_of_tweets'], marker='*', color='purple')
#     axes[1].set(xlabel='Hashtag frequency per tweet', ylabel='log number of tweets',
#                 title='Fig-2: Number of hashtags per tweet (logarithmic scale)')
#     axes[1].set_yscale('log')
# #     axes[1].set_xscale('log')
#     #fig.savefig("test.png")
#     plt.show()
    
    #figure-3: url-hashtag relationship 
    fig, ax=plt.subplots(figsize=(10, 5))
    ax.plot(df_hash2['hash_count'],df_hash2['url_percent_withurl'], marker='*', color='red')
    ax.set(xlabel='Hashtag frequency per tweet', ylabel='percentage of tweets with at list one url',
                title='Fig-3: The relationship between the number of hashtags and the existence of a URL in tweets')
    # ax.set_yscale('log')
    # ax.set_xscale('log') 
    #fig.savefig("test.png")
    plt.show()


        
def report (df , total_hashtweets , total_hashtags , total_uniqhashtags ,total_hashtags_reduced,
            total_uniqhashtags_reduced , edges, nodes):#, edge_noise_removed , edges 
    """ reports the characteristics of the cleaned tweets
    and also top 20 popular hashtags
    """
    headers = ["Characteristics of the cleaned data", "Value"]
    table = [["cleaned tweets", format(df.shape[0],',')],
             ['Total hashtags (repetitive) in cleaned tweets', format(total_hashtags,',')],
             ['-- Unique hashtags', format(total_uniqhashtags,',')],
             ['Total hashtags (repetitive) after removing unpopular hashtags', format(total_hashtags_reduced,',')],
             ['-- Unique hashtags after removing unpopular hashtags', format(total_uniqhashtags_reduced,',')],
             ["Tweets containing no hashtags", format(df.shape[0]-total_hashtweets,',')], 
             ["Tweets containing at least one hashtag", format(total_hashtweets,',')],
             ['-- One hashtag per tweet', format(len(df.loc[df['h_number']==1]),',')],
             ['-- Two hashtags per tweet', format(len(df.loc[df['h_number']==2]),',')],
             ['-- More than two hashtags per tweet', format(len(df.loc[df['h_number']>=3]),",")],
             ['Maximum number of hashtags used in a tweet', format(df['h_number'].max(),',')],
             ["Total hashtag co-occurrences after removing unpopular hashtags ", format(edges.shape[0],',')]] # co-occurrence in different tweets, 
    print(tabulate(table, headers, tablefmt="fancy_grid"))
    # top 20 hashtags
    print(tabulate(nodes[['Id','Label']].loc[0:19].values, headers=['No.','Hashtag'], 
                   tablefmt='fancy_grid'))
    

   
    
    
    
def hashtag_explore( n):
    """
    input: path which is the location of cleaned dataset (df)
           n is the hashtag count to filter out unpopular hashtags
           
    this function creates:
    1) dfh.csv (dataset of hashtag-tweeets)
    2) dfhu.csv (all the unique hashtags with list of tweet ids they occured in)
    3) dfh_unpoprmvd.csv (removed unpopular hashtags with less than n counts in the corpus)
    4) nodes.csv (after removing unpopular hashtags) 
    5) edges.csv
    6) edges_noise_removed.csv (removed noise from edges.csv)
    7) fig-1,2: hashtsg distribution (based on nodes.csv which means after removing unpopular hashtags)
    8) fig 3,4: number of hashtags per tweet (based on the full set of tweets which means before removing unpopular hashtags)
    9) fig 5: relationship bewteen the number of hashtags and existance of a url in tweets (based on full set of tweets)
    10) table of characteristics of the cleaned data both for after removing unpopular hashtags and after noise removal
    11) table of top 20 hashtags (based on nodes.csv which means after removing unpopular hashtags)
    """
    df = pd.read_csv('data files/df.csv')  
#     df[col_name]=df[col_name].apply(lambda x: x.replace("'", ''))
    dfhu_reduced, dfh_reduced, total_hashtweets, total_hashtags , total_uniqhashtags ,total_hashtags_reduced, total_uniqhashtags_reduced = hashtag_dataframe(df, n )
    nodes = graph_nodes(dfhu_reduced )
    edges= graph_edges(dfh_reduced, nodes)
    hashtag_plots(df, nodes )
    report(df , total_hashtweets , total_hashtags , total_uniqhashtags ,total_hashtags_reduced,
            total_uniqhashtags_reduced ,edges , nodes)

    
    

    
    
