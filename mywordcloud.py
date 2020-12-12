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
CREATE WORDCLOUD

This is a module to create a wordcloud representation. 

input: 
 df1: dataset to consider
 col_name: text of this column of the df1

"""
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# %matplotlib inline

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(15, 10))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

def string_prep(df1, col_name):
    df1[col_name]=df1[col_name].apply(lambda x: x.replace(';',','))
    df1[col_name]=df1[col_name].apply(lambda x: x.replace('[',','))
    df1[col_name]=df1[col_name].apply(lambda x: x.replace(']',''))
    df1[col_name]=df1[col_name].apply(lambda x: x.replace("'",''))
    df1[col_name]=df1[col_name].apply(lambda x: x.replace(",",''))
    df1[col_name]=df1[col_name].apply(lambda x: x.split(' '))
    df1=df1.explode(col_name)
    df1[col_name]=df1[col_name].apply(lambda x: x.lower())
    return df1

def create_wc(col_name):
    """
    options for col_name are 'preprocessed' and 'tokenized'
    tokenized if want to  work with tokens
    preprocessed if want to work with clean sentences before tokenizing/ lemmatizing/ stop word removal

    """
    df=pd.read_csv('data files/df.csv')# read df file
    df1=df[['created_at',col_name]].copy()    
    #need to do string_prep as its type is not list
    df1=string_prep(df1,col_name)       
    text=df1[col_name].str.cat(sep=' ')    
    wordcloud = WordCloud(width = 3000,
                          height = 2000,
                          random_state= 1,
                          normalize_plurals= False,
                          collocations=False,
                          background_color='white',
                          color_func=lambda *args, **kwargs: "black",
                          colormap='Set2').generate(text)
    #collocations=True will include the bigrams 
    plot_cloud(wordcloud)

# -


