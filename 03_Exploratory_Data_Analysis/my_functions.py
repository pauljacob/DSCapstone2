#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

pd.set_option('display.max_colwidth', None)
pd.options.display.max_info_columns = 999

#return the first 5 and last 5 rows of this dataframe
def p(df_):
    if df_.shape[0] > 6:
        print(df_.shape)
        return pd.concat([df_.head(), df_.tail()])
    else:
        return df_

def rcp(file_, pd_=None):
    if pd_ == None:
        return pd.read_csv(os.path.join('..', 'processed_data', file_))
    else:
        return pd.read_csv(os.path.join('..', 'processed_data', file_), parse_dates=pd_)
    
def rcr(file_, pd_=None):
    if pd_ == None:
        return pd.read_csv(os.path.join('..', 'raw_data', file_))
    else:
        return pd.read_csv(os.path.join('..', 'raw_data', file_), parse_dates=pd_)
    
#sort dataframe by column
def s(df_, column_):
    return df_.sort_values(column_)

#reset index and sort dataframe by column
def sr(df_, column_):
    df_ = df_.sort_values(column_)
    return df_.reset_index(drop=True)

#print length of list
def pl(list_):
    print(len(list_))
    return list_

#print length of list
def pdc(dict_):
    print(len(dict_))
    return dict_

# In[ ]:





# In[6]:


def adder(a, b):
    '''add a and b. string after'''
    return a + b


# In[7]:


#print(adder.__doc__)


# In[ ]:

















'''
take dataframe with two suffixes and return the stacked transformation
'''

def wide2tall(df_, unique_identifiers_list_, suffixes_):
    
    column_names_list = list(df_.columns)
    
    column_names_first_list = [k for k in column_names_list if k.endswith(suffixes_[0])]
    
    column_names_second_list = [k for k in column_names_list if k.endswith(suffixes_[1])]
    
    column_names_first_second_list = column_names_first_list + column_names_second_list
    column_names_not_first_second_list = [k for k in column_names_list if not k in column_names_first_second_list]
    
    
    #add unique identifier to first_list and second_list if those lists do not contain the unique identifier


    #add_to_first_list = []
    dont_add_to_first_list = []
    
    #add_to_second_list = []
    dont_add_to_second_list = []
    
    merge_on_me_list = []

    for i in range(len(unique_identifiers_list_)):

        #if unique_identifier is in list, then don't add it. otherwise add it.
        for j in range(len(column_names_first_list)):
            if unique_identifiers_list_[i] in column_names_first_list[j]:
                dont_add_to_first_list = dont_add_to_first_list + [unique_identifiers_list_[i]]            

        for j in range(len(column_names_second_list)):
            if unique_identifiers_list_[i] in column_names_second_list[j]:
                dont_add_to_second_list = dont_add_to_second_list + [unique_identifiers_list_[i]]
        
        for j in range(len(column_names_not_first_second_list)):
            if unique_identifiers_list_[i] in column_names_not_first_second_list[j]:
                merge_on_me_list = merge_on_me_list + [unique_identifiers_list_[i]]
    
    add_to_first_list = [k for k in unique_identifiers_list_ if not k in dont_add_to_first_list]
    
    add_to_second_list = [k for k in unique_identifiers_list_ if not k in dont_add_to_second_list]
    
    column_names_first_list_added = add_to_first_list + column_names_first_list
    
    column_names_second_list_added = add_to_second_list + column_names_second_list
    
    #print(column_names_first_list_added)
    
    #print(column_names_second_list_added)
    
    
    
    df_first_ = df_.loc[:,column_names_first_list_added]
    
    df_second_ = df_.loc[:,column_names_second_list_added]
    
    
    

    alphanumeric = ''
    suffix_cat = ''
    for i in range(len(suffixes_)):
        
        for character in suffixes_[i]:
            if character.isalnum():
                alphanumeric += character
        
        suffix_cat = suffix_cat + alphanumeric + '_'
        alphanumeric = ''
        
        
    #suffix_cat = suffix_cat[0:len(suffix_cat) - 1]
    
    df_first_.loc[:, suffix_cat] = 0
    
    df_second_.loc[:, suffix_cat] = 1
    
    
    
    column_names_list_stripped = [k.split(suffixes_[0])[0] for k in column_names_first_list_added]
    
    column_names_first_dict = dict(zip(column_names_first_list_added, column_names_list_stripped))
    
    column_names_second_dict = dict(zip(column_names_second_list_added, column_names_list_stripped))
    
    
    df_first_.rename(columns=column_names_first_dict, inplace=True)


    df_second_.rename(columns=column_names_second_dict, inplace=True)
    
    
    df_fs = pd.concat([df_first_, df_second_])

    df_nfs = df_.loc[:, column_names_not_first_second_list]
    
    #print()
    #print(column_names_not_first_second_list)
    
    df_all = pd.merge(df_fs, df_nfs, on=merge_on_me_list, how='inner')

    
    
    return df_all
    
    
    

#go from tall to wide

def tall2wide2(df_, not_first_second_list_, indicator_column_, keep_indicator_=False, first_second_column_suffix_=None):

    
    if first_second_column_suffix_ == None:
        first_second_column_suffix_ = ['_a', '_b']
    
    column_names_list_ = list(df_.columns)
    
    column_names_first_second_list_ = [k for k in column_names_list_ if not k in not_first_second_list_]
    
    df_first_ = df_.loc[(df_.loc[:, indicator_column_] == 0), :]
    
    df_second_ = df_.loc[(df_.loc[:, indicator_column_] == 1), :]
    
    if keep_indicator_ == False:
        df_first_.drop(columns=indicator_column_, inplace=True)
        df_second_.drop(columns=indicator_column_, inplace=True)
    
    column_names_first_second_list_first_ = [k + first_second_column_suffix_[0]
                                            for k in column_names_first_second_list_]
    
    column_names_first_second_list_second_ = [k + first_second_column_suffix_[1]
                                             for k in column_names_first_second_list_]
    
    
    column_names_first_second_dict_first_ = dict(zip(column_names_first_second_list_,
                                                     column_names_first_second_list_first_))
    
    column_names_first_second_dict_second_ = dict(zip(column_names_first_second_list_,
                                                     column_names_first_second_list_second_))
    
    df_first_.rename(columns=column_names_first_second_dict_first_, inplace=True)
    
    df_second_.rename(columns=column_names_first_second_dict_second_, inplace=True)
    
    return pd.merge(df_first_, df_second_, on=not_first_second_list_)
    

    
    
    
    
    
    
    
    
    
#take difference of numeric columns with the same prefix, store to stat_cma#_diff

#drop stat_cma# columns

def get_numeric_diff(df_, groupby_list_, not_difference_list_, suffix_list_):
    
    column_names_numeric_list_ = list(df_.select_dtypes('number').columns)
    
    column_names_numeric_list_cleaned_ = [k for k in column_names_numeric_list_
                                          if (not k in not_difference_list_)]
    
    column_names_first_second_list_ = [k.split(suffix_list_[0])[0]
                                       for k in column_names_numeric_list_cleaned_
                                       if k.endswith(suffix_list_[0])]
    
    column_names_first_second_list_rgl_ = [k for k in column_names_first_second_list_
                                           if not k in groupby_list_]

    column_names_first_second_list_rgl_first_ = [k + suffix_list_[0] for k in column_names_first_second_list_rgl_]
    
    column_names_first_second_list_rgl_second_ = [k + suffix_list_[1] for k in column_names_first_second_list_rgl_]
    
    column_names_first_second_list_rgl_diff_ = [k + '_diff' for k in column_names_first_second_list_rgl_]
    
    #return len(column_names_first_second_list_rgl_first_), len(column_names_first_second_list_rgl_second_), len(column_names_first_second_list_rgl_diff_)
    
    
    df_[column_names_first_second_list_rgl_diff_] = df_[column_names_first_second_list_rgl_first_].fillna(0).values - df_[column_names_first_second_list_rgl_second_].fillna(0)
    
    column_names_drop_ = column_names_first_second_list_rgl_first_ + column_names_first_second_list_rgl_second_
    
    df_.drop(columns = column_names_drop_, inplace=True)    
    
    #return df_[column_names_first_second_list_rgl_first_]
    
    #return df_[column_names_first_second_list_rgl_second_]
    
    return df_


    
    
def wide_get_cma_num_diff3(df_, groupby_list_, not_difference_list_, suffix_list_, window_size_max_min_):
    
    
    #get suffix category column name: suffix_cat
    alphanumeric = ''
    suffix_cat = ''
    for i in range(len(suffix_list_)):
        
        for character in suffix_list_[i]:
            if character.isalnum():
                alphanumeric += character
        
        suffix_cat = suffix_cat + alphanumeric + '_'
        alphanumeric = ''

    
    column_names_list_ = list(df_.columns)
    
    column_names_first_second_list_ = [k for k in column_names_list_
                                       if not k in not_difference_list_]
    
    column_names_first_second_list_first_ = [k for k in column_names_first_second_list_
                                            if k.endswith(suffix_list_[0])]
    
    column_names_first_second_list_first_stripped_ = [k.split(suffix_list_[0])[0]
                                                     for k in column_names_first_second_list_first_]
    
    
    df_ = wide2tall(df_, not_difference_list_, suffix_list_)
    

    
    column_names_number_list_ = list(df_.select_dtypes('number'))

    column_names_number_list_ = [k for k in column_names_number_list_ if not k in not_difference_list_]
    
    column_names_number_list_ = [k for k in column_names_number_list_ if not k in suffix_cat]
    
    gbl_column_names_number_list_ = groupby_list_ + [k for k in column_names_number_list_
                                                          if not k in groupby_list_] 
    
    df_numeric_gbl_ = df_[gbl_column_names_number_list_]
    
    
    df_gbl_cma_ = df_numeric_gbl_.groupby(
                                            groupby_list_).apply(lambda x:
                                            x.rolling(window_size_max_min_[0],
                                            window_size_max_min_[1]).mean().shift()
                                         )[column_names_number_list_]
    
    pre_rename_cma_drop_list_ = [k for k in column_names_number_list_ if k in groupby_list_]
    
    df_gbl_cma_.drop(columns=pre_rename_cma_drop_list_, inplace=True)
    
     
    column_names_number_list_cma_num_ = [k + '_cma' + str(window_size_max_min_[0]) for k in column_names_number_list_]
    
    column_names_number_dict_cma_num_ = dict(zip(column_names_number_list_, column_names_number_list_cma_num_))
        
    df_gbl_cma_.rename(columns=column_names_number_dict_cma_num_, inplace=True)
    
    df_gbl_cma_ = df_gbl_cma_.reset_index()
    
    df_gbl_cma_.rename(columns={'level_2': 'index'}, inplace=True)

    df_ = df_.reset_index()
    
    column_names_merged_list = ['index'] + groupby_list_
    
   
    column_names_number_list_rgbl_ = [k for k in column_names_number_list_ if not k in groupby_list_]
    
    df_dropped_ = df_.drop(columns=column_names_number_list_rgbl_)
        
    df_gbl_cma_merged_ = pd.merge(df_dropped_, df_gbl_cma_, on=column_names_merged_list)
    
    
    
    df_gbl_cma_merged_.drop(columns='index', inplace=True)
    
    
    df_wide_ = tall2wide2(df_gbl_cma_merged_, not_difference_list_, suffix_cat)

    df_wide_ = sr(df_wide_, 'game_date')
    
    
    
    #take difference of numeric columns with the same prefix, store to stat_cma#_diff
    
    return get_numeric_diff(df_wide_, groupby_list_, not_difference_list_, suffix_list_)
    
    #merge back to columns not taken a difference on
    
