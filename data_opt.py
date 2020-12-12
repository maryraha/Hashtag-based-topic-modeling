#!/usr/bin/env python
# coding: utf-8
# %%
'''
DATA_OPT MODULE
This module takes the csv file of the collected tweets from TCAT-DMI and optimizes the memory usage
by changing datatypes and removing unnecessary columns  

'''

import numpy as np 
import pandas as pd
import os
from typing import List


def optimize_floats(df) :
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df):
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    for col in ints:
        IsInt = False
        mx = df[col].max()
        mn = df[col].min()
            
        # Integer does not support NA, therefore, NA needs to be filled
        if not np.isfinite(df[col]).all(): 
            df[col].fillna(0,inplace=True) 
                
        # test if column can be converted to an integer
        asint = df[col].fillna(0).astype(np.int64)
        result = (df[col] - asint)
        result = result.sum()
        if result > -0.01 and result < 0.01:
            IsInt = True
            
        # Make Integer/unsigned Integer datatypes
        if IsInt:
            if mn >= 0:
                if mx < 255:
                    df[col] = df[col].astype(np.uint8)
                elif mx < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif mx < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64) 
    return df


def optimize_objects(df, category_cls: List[str]):
    for col in df.select_dtypes(include=['object']):
        if col not in category_cls:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
        else:
            df[col] = df[col].astype('category')
    return df


def optimize(df, category_cls: List[str]):
    return optimize_floats(optimize_ints(optimize_objects(df, category_cls)))

def optimized_data(data_path, use_cols, category_cls: List[str] = []):
#     data_path=os.path.join(tmppath, data_path)
    start_mem_usg= os.path.getsize(data_path)/ 1024**2
    df= pd.read_csv(data_path, usecols=use_cols)
    optimize(df, category_cls)
    print("___Read the data and optimize the memory usage___")
    print("Original dataframe is :",start_mem_usg," MB")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print( "Optimized dataframe is: ",mem_usg," MB")
    print("Memory usage has optimized",100*(1-mem_usg/start_mem_usg),"% \n")
    return df


