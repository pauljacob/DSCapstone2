#get libraries
import pandas as pd
import os
import numpy as np
import itertools
#from functools import reduce


#get visualization libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns



#ML preprocessing
from sklearn.preprocessing import StandardScaler

#get ML functions
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


from sklearn import __version__ as sklearn_version
import datetime


#get ML metric functions
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, log_loss, auc, brier_score_loss
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report, brier_score_loss
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import log_loss








####################
import os
import pandas as pd
import numpy as np


#data wrangling
from functools import reduce


#get visualization libraries
import matplotlib.pyplot as plt

#data preprocessing

#get ML functions
from sklearn.model_selection import train_test_split


#ML
import pickle
from sklearn.model_selection import learning_curve


#save object
# import zipfile
# import shutil


def initialize_custom_notebook_settings():
    '''
    initialize the jupyter notebook display width'''
        
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:99.9% !important; }</style>"))

    
    pd.options.display.max_columns = 999
    pd.options.display.max_rows = 999

    pd.set_option('display.max_colwidth', None)
    pd.options.display.max_info_columns = 999

    



'''
Convenience functions: read, sort, print, and save data frame, dictionary, or model.
'''
def p(df):
    '''
    Return the first 5 and last 5 rows of this DataFrame.'''
    if df.shape[0] > 6:
        print(df.shape)
        return pd.concat([df.head(), df.tail()])
    else:
        return df

def rcp(filename, parse_dates=None, index_col=None):
    '''
    Read a file from the data/processed folder.'''
    if parse_dates == None:
        return pd.read_csv(os.path.join('..', 'data', 'processed', filename), index_col=index_col)
    else:
        return pd.read_csv(os.path.join('..', 'data', 'processed', filename), parse_dates=parse_dates,  index_col=index_col)

    

def rcp_v2(filename, column_name_row_integer_location_list='infer', index_column_integer_location_list=None, parse_dates=None, data_directory_name='processed'):
    '''
    filename: name of file
    data_directory_name: name of data directory'''
    
    #read it back
    return pd.read_csv(filepath_or_buffer=os.path.join('..', 'data', data_directory_name, filename), sep=',', delimiter=None, header=column_name_row_integer_location_list, index_col=index_column_integer_location_list, usecols=None, squeeze=None, mangle_dupe_cols=True, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=parse_dates, infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None, quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None, encoding=None, encoding_errors='strict', dialect=None, error_bad_lines=None, warn_bad_lines=None, on_bad_lines=None, delim_whitespace=False, low_memory=True, memory_map=False, float_precision=None, storage_options=None)

    

def rpp(filename, parse_dates=None):
    '''
    Read a collection from the processed_data folder.'''

    relative_file_path = os.path.join('..', 'data', 'processed', filename)
        
    with (open(relative_file_path, "rb")) as open_file:
        return pickle.load(open_file)
    
def rcr(filename, parse_dates=None):
    '''
    Read a file from the raw_data folder.'''
    if parse_dates == None:
        return pd.read_csv(os.path.join('..', 'data', 'raw', filename))
    else:
        return pd.read_csv(os.path.join('..', 'data', 'raw',  filename), parse_dates=parse_dates)

def sr(df, column_name_list):
    '''
    Sort DataFrame by column(s) and reset its index.'''
    df = df.sort_values(column_name_list)
    return df.reset_index(drop=True)

def pl(list_):
    '''
    Print the list length and return the list.'''
    print(len(list_))
    return list_

def pdc(dict_):
    '''
    Print the dictionary length and return the dictionary.'''
    print(len(dict_))
    return dict_


def show_data_frames_in_memory(dir_):
    alldfs = [var for var in dir_ if isinstance(eval(var), pd.core.frame.DataFrame)]

    print(alldfs)


    
def get_column_name_list_left_not_in_right(df_left, df_right):

    column_name_list_in_both = list(set(df_left.columns).intersection(set(df_right.columns)))

    return [k for k in df_left.columns if k not in column_name_list_in_both]


##################################################################################################################################
#save and return object
##################################################################################################################################
def save_and_return_data_frame(df, filename, index=False, parse_dates=False, index_label=None):
    '''
    Save data frame and return it.'''
    
    relative_directory_path = os.path.join('..', 'data', 'processed')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)
    if not os.path.exists(relative_file_path):
        df.to_csv(relative_file_path, index=index, index_label=index_label)
    elif os.path.exists(relative_file_path):
        print('This file already exists.')

    return rcp(filename, parse_dates, index_col=index_label)

def save_and_return_data_frame_v2(df, filename, index=True, data_directory_name='processed'):
    '''
    Save data frame and return it.
    
    df:data frame to save and return
    filename: name of file to save to data processed folder
    index: boolean value of whether to include data frame index in save
    '''
    
    
    #initialize for write out
    relative_directory_path = os.path.join('..', 'data', data_directory_name)
    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)
        
    relative_file_path = os.path.join(relative_directory_path, filename)
    if not os.path.exists(relative_file_path):
        #write out file
        df.to_csv(path_or_buf=relative_file_path, sep=',', na_rep='', float_format=None, columns=None, header=True, index=index, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict', storage_options=None)
        
    elif os.path.exists(relative_file_path):
        print('This file already exists.')

    #initialize for readback
    if index==True:
        #get the index and column integer locations
        column_name_row_integer_location_list=[index for index in range(pd.DataFrame(df.columns.to_list()).shape[1])]
        index_column_integer_location_list=[index for index in range(pd.DataFrame(df.index.to_list()).shape[1])]
    elif index==False:
        column_name_row_integer_location_list=[index for index in range(pd.DataFrame(df.columns.to_list()).shape[1])]
        index_column_integer_location_list=None

    return rcp_v2(filename=filename, column_name_row_integer_location_list=column_name_row_integer_location_list, index_column_integer_location_list=index_column_integer_location_list, data_directory_name=data_directory_name)




def save_and_return_collection(data_frame_collection, filename):
    '''
    Save collection and return it.'''

    relative_directory_path = os.path.join('..', 'data', 'processed')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)
    
    if not os.path.exists(relative_file_path):
        file_object_wb =  open(relative_file_path, "wb")
        pickle.dump(data_frame_collection, file_object_wb)
        file_object_wb.close()

    elif os.path.exists(relative_file_path):
        print('This file already exists.')
        
    with (open(relative_file_path, "rb")) as open_file:
        data_frame_collection_readback = pickle.load(open_file)

    return data_frame_collection_readback


def save_and_return_model(model, filename, add_compressed_file=False):

    
    relative_directory_path = os.path.join('..', 'models')

    #make relative file direactory path if it doesn't exist
    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)
        
    #get relative file path name
    relative_file_path = os.path.join(relative_directory_path, filename)
        
    if os.path.exists(relative_file_path):
            print('This file already exists.')
            
    #if model file doesn't exist, then save it
    elif not os.path.exists(relative_file_path):
        file_object_wb =  open(relative_file_path, "wb")
        pickle.dump(model, file_object_wb)
        file_object_wb.close()
        
    
    #readback model file
    with (open(relative_file_path, "rb")) as open_file:
        model_readback = pickle.load(open_file)

    return model_readback

##################################################################################################################################
#return object if it exists
##################################################################################################################################

def return_processed_data_file_if_it_exists(filename, parse_dates=False):
    
    relative_directory_path = os.path.join('..', 'data', 'processed')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)

    if os.path.exists(relative_file_path):
        print('This file already exists')
        return rcp(filename, parse_dates)
    else:
        return pd.DataFrame({})
    
def return_processed_data_file_if_it_exists_v2(filename, column_name_row_integer_location_list='infer', index_column_integer_location_list=None, parse_dates=None):
    
    relative_directory_path = os.path.join('..', 'data', 'processed')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)

    if os.path.exists(relative_file_path):
        print('This file already exists')
        return rcp_v2(filename=filename, column_name_row_integer_location_list=column_name_row_integer_location_list, index_column_integer_location_list=index_column_integer_location_list, parse_dates=parse_dates)
    else:
        return pd.DataFrame({})

def True_False_data_filename_exists(filename):
    
    relative_directory_path = os.path.join('..', 'data', 'processed')
    
    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)
        
    relative_file_path = os.path.join(relative_directory_path, filename)
    
    if os.path.exists(relative_file_path):
        return True
    else:
        return False
    
    
    
    
def return_processed_collection_if_it_exists(filename, parse_dates=False):
    
    relative_directory_path = os.path.join('..', 'data', 'processed')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)

    if os.path.exists(relative_file_path):
        print('This file already exists')
        
        with (open(relative_file_path, "rb")) as openfile:
            data_frame_collection_readback = pickle.load(openfile)
        
        return data_frame_collection_readback
    
    else:
        return None
    
def return_saved_model_if_it_exists(filename):
    relative_directory_path = os.path.join('..', 'models')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)

    if os.path.exists(relative_file_path):
        print('This file already exists')
        
        with (open(relative_file_path, "rb")) as openfile:
            model_readback = pickle.load(openfile)
            
        return model_readback
    else:
        return None
    
    
    
def return_figure_if_it_exists(filename):

    import glob
    import imageio

    image = None
    for image_path in glob.glob(filename):
        image = imageio.imread(image_path)
        
    return image

################################################################################################################################## 







##################################################################################################################################
#data wrangling
##################################################################################################################################

def merge_data_frame_list(data_frame_list):
    return reduce(lambda left, right : pd.merge(left,
                                                right,
                                                on=[column_name for column_name in left if column_name in right],
                                                how='inner'), 
                  data_frame_list)


def concatenate_multiindex_objects(multiindex_list):
    '''
    multiindex_list: list of MultiIndex objects to be appended to each other
    return: single MultiIndex Object
    '''
    
    tuple_list=[]
    for multiindex in multiindex_list:
        tuple_list+=list(multiindex)
        
    return pd.MultiIndex.from_tuples(tuple_list)



##################################################################################################################################
#feature engineering
##################################################################################################################################



##################################################################################################################################
#exploratory data analysis
##################################################################################################################################




def get_metrics_from_two_features_in_data_frame(df, feature_column_name_list, conversion_rate_minimum, conversions_minimum):
    '''
    two features with one feature value each to filter for (i.e. select) responses for coupon recommendation'''
    
    metric_value_two_dimensional_list=[]
    
    feature_column_name_01_value_collection={}
    for feature_column_name_01_value_collection[0] in list(df.loc[:, feature_column_name_list[0]].unique()):
        for feature_column_name_01_value_collection[1] in list(df.loc[:, feature_column_name_list[1]].unique()):
            
            metric_value_list=[]
            
            feature_column_name_01_value_01_coupons_recommended_collection={}
            feature_column_name_01_value_01_conversions_collection={}
            
            
            #get y_predicted for 3 df_filtered - feature column name 0 value feature column name 1 value, feature column name 0 value, feature column name 1 value
            
            #get y_predict for feature column name 0 value, feature column name 1 value
            y_predicted_feature_column_name_collection={}
            for index in range(len(feature_column_name_list)):
                y_predicted_feature_column_name_collection[index]='y_predicted_'+feature_column_name_list[index]+'_'+str(feature_column_name_01_value_collection[index]).replace(' ', '_')
                df.loc[:, y_predicted_feature_column_name_collection[index]]=0                
                df.loc[df.loc[:, feature_column_name_list[index]]==feature_column_name_01_value_collection[index], y_predicted_feature_column_name_collection[index]]=1
                
                #get conversion per one feature one value filter 
                feature_column_name_01_value_01_conversions_collection[index]=df.loc[(df.loc[:, feature_column_name_list[index]]==feature_column_name_01_value_collection[index]) & (df.loc[:, 'Y']==1), :].shape[0]
                
                #get coupons recommended per one feature one value filter
                feature_column_name_01_value_01_coupons_recommended_collection[index]=df.loc[df.loc[:, feature_column_name_list[index]]==feature_column_name_01_value_collection[index], :].shape[0]
            
            
            #get y_predicted for feature column name 0 value feature column name 1 value
            y_predicted_feature_column_name_collection[2]='y_predicted_'+feature_column_name_list[0]+'_'+str(feature_column_name_01_value_collection[0]).replace(' ', '_')+'_'+feature_column_name_list[1]+'_'+str(feature_column_name_01_value_collection[1]).replace(' ', '_')
            df.loc[:, y_predicted_feature_column_name_collection[2]]=0
            df.loc[(df.loc[:, feature_column_name_list[0]]==feature_column_name_01_value_collection[0]) &
                   (df.loc[:, feature_column_name_list[1]]==feature_column_name_01_value_collection[1]), y_predicted_feature_column_name_collection[2]]=1
            
            df_y_actual_y_predicted_feature_column_name_0_1_01 = df.loc[:, ['Y', y_predicted_feature_column_name_collection[0], y_predicted_feature_column_name_collection[1], y_predicted_feature_column_name_collection[2]]]
            
            metric_value_list+=[feature_column_name_list[0], feature_column_name_01_value_collection[0], feature_column_name_list[1], feature_column_name_01_value_collection[1],]
            
            
            
            
            def get_metrics_for_two_feature_column_name_y_predicted_and_combination_filter(df, y_predicted_feature_column_name_collection, feature_column_name_01_value_01_coupons_recommended_collection, metric_value_list):
                
                
                def get_metrics_from_y_predicted_and_y_predicted_feature_features_column_name_feature_features_column_name_value_values(df, y_predicted_column_name, feature_column_name_01_value_01_coupons_recommended_collection, metric_value_list, index):
                    '''
                    get metrics for y_predicted of a one feature column name and one feature column name value OR
                                                     two feature column names and one feature column name value each'''
                    
                    y_true=df.loc[:, ['Y']]
                    y_predicted=df.loc[:, [y_predicted_column_name]]
                    
                    survey_coupons_recommended=df.shape[0]
                    survey_conversions=df.loc[df.loc[:, 'Y']==1,'Y'].shape[0]
                    
                    #conversion rate
                    metric_value_list+=[precision_score(y_true=y_true, y_pred=y_predicted)]

                    #recall
                    metric_value_list += [recall_score(y_true=y_true, y_pred=y_predicted)]
                    
                    confusion_matrix_ndarray = confusion_matrix(y_true=y_true, y_pred=y_predicted)
                    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix_ndarray.ravel()
                    
                    if index==2:
                        #get proportion of conversions for feature 0 and value feature 1 and value of feature 0 and value
                        metric_value_list+=[true_positives/feature_column_name_01_value_01_conversions_collection[0]]
                        
                        #get proportion of conversions for feature 0 and value feauture 1 and value of feature 1 and value
                        metric_value_list+=[true_positives/feature_column_name_01_value_01_conversions_collection[1]]
                    
                    #conversions
                    metric_value_list += [true_positives]                    
                    
                    #proportion of coupons recommended
                    coupons_recommended=true_positives+false_positives 
                    if (index == 0) | (index == 1):
                        metric_value_list += [coupons_recommended/survey_coupons_recommended]
                    
                    elif index == 2:
                        #proportion of feature 0 and value coupons recommended by feature 0 and value and feature 1 and value
                        metric_value_list += [coupons_recommended/feature_column_name_01_value_01_coupons_recommended_collection[0]]
                        
                        #proportion of feature 1 and value coupons recommended by feature 0 and value and feature 1 and value
                        metric_value_list += [coupons_recommended/feature_column_name_01_value_01_coupons_recommended_collection[1]]
                        
                    #coupons recommended
                    metric_value_list += [coupons_recommended]
                    
                    return metric_value_list
                    
                    
                    
                for index in range(3):           
                    metric_value_list=get_metrics_from_y_predicted_and_y_predicted_feature_features_column_name_feature_features_column_name_value_values(df, y_predicted_column_name=y_predicted_feature_column_name_collection[index], feature_column_name_01_value_01_coupons_recommended_collection=feature_column_name_01_value_01_coupons_recommended_collection, metric_value_list=metric_value_list, index=index)

                return metric_value_list
                
            metric_value_two_dimensional_list+=[get_metrics_for_two_feature_column_name_y_predicted_and_combination_filter(df=df_y_actual_y_predicted_feature_column_name_0_1_01, y_predicted_feature_column_name_collection=y_predicted_feature_column_name_collection, feature_column_name_01_value_01_coupons_recommended_collection=feature_column_name_01_value_01_coupons_recommended_collection, metric_value_list=metric_value_list,)]
        
    #get column name list
    column_name_list_feature_0_feature_0_value_feature_1_feature_1_value=['Feature '+str(index)+ column_name_substring for index in range(2) for column_name_substring in ['', ' Value',]]
    column_name_list_feature_0_metrics_feature_1_metrics= ['Feature ' + str(index) + ' ' + metric_name for index in range(2) for metric_name in ['Conversion Rate', 'Recall', 'Conversions', 'Proportion of Coupons Recommended', 'Coupons Recommended']]
    column_name_list_feature_1_feature_2_metrics=['Conversion Rate', 'Recall', 'Conversions to Feature 0 Conversions Ratio', 'Conversions to Feature 1 Conversions Ratio', 'Conversions', 'Coupons Recommended to Feature 0 Coupons Recommended Ratio', 'Coupons Recommended to Feature 1 Coupons Recommended Ratio', 'Coupons Recommended']
    #get column name list from column name lists
    column_name_list=column_name_list_feature_0_feature_0_value_feature_1_feature_1_value+column_name_list_feature_0_metrics_feature_1_metrics+column_name_list_feature_1_feature_2_metrics

    #convert list to data frame
    df_metrics=pd.DataFrame(metric_value_two_dimensional_list, columns=column_name_list).sort_values('Conversion Rate', ascending=False)

    #add metric differences
    df_metrics.loc[:, 'Conversion Rate and Feature 0 Conversion Rate Difference']=df_metrics.loc[:,'Conversion Rate']-df_metrics.loc[:,'Feature 0 Conversion Rate']
    df_metrics.loc[:, 'Conversion Rate and Feature 1 Conversion Rate Difference']=df_metrics.loc[:,'Conversion Rate']-df_metrics.loc[:,'Feature 1 Conversion Rate']
    df_metrics.loc[:, 'Recall and Feature 0 Recall Difference']=df_metrics.loc[:, 'Recall']-df_metrics.loc[:, 'Feature 0 Recall']
    df_metrics.loc[:, 'Recall and Feature 1 Recall Difference']=df_metrics.loc[:, 'Recall']-df_metrics.loc[:, 'Feature 1 Recall']
    df_metrics.loc[:, 'Conversions and Feature 0 Conversions Difference']=df_metrics.loc[:, 'Conversions']-df_metrics.loc[:, 'Feature 0 Conversions']
    df_metrics.loc[:, 'Conversions and Feature 1 Conversions Difference']=df_metrics.loc[:, 'Conversions']-df_metrics.loc[:, 'Feature 1 Conversions']
    df_metrics.loc[:, 'Coupons Recommended and Feature 0 Coupons Recommended Difference']=df_metrics.loc[:, 'Coupons Recommended']-df_metrics.loc[:, 'Feature 0 Coupons Recommended']
    df_metrics.loc[:, 'Coupons Recommended and Feature 1 Coupons Recommended Difference']=df_metrics.loc[:, 'Coupons Recommended']-df_metrics.loc[:, 'Feature 1 Coupons Recommended']
    
    #filter by conversion rate
    df_metrics_filtered=df_metrics.loc[(df_metrics.loc[:,'Conversion Rate'] > conversion_rate_minimum) &
                                       (df_metrics.loc[:,'Conversions'] > conversions_minimum), :]
    
    
    return df_metrics_filtered















# def get_survey_metrics_from_two_features_one_feature_value_each(df, feature_column_name_list):

    
    
#     feature_0_column_name_value_list=list(df.loc[:, feature_column_name_list[0]].unique())
#     feature_1_column_name_value_list=list(df.loc[:, feature_column_name_list[1]].unique())
    
#     metric_value_2_dimensional_list=[]

#     def get_metrics_from_data_frame_Y_filtered(df_Y_filtered, metric_value_list):

#         if df_Y_filtered.empty == False:

#             #get target variable unique value list
#             target_variable_value_unique_list=list(df_Y_filtered.loc[:, 'Y'].unique())

#             #initialize booleans coupon acceptance exists and coupon refusal exists
#             coupon_acceptance_exists=True if 1 in target_variable_value_unique_list else False
#             coupon_refusal_exists=True if 0 in target_variable_value_unique_list else False

#             #initialize coupons recommended, coupons acceptance/refusal count, coupons acceptance/refusal proportion
#             coupons_recommended=df_Y_filtered.shape[0]
#             df_Y_filtered_value_counts=df_Y_filtered.value_counts()
#             df_Y_filtered_value_counts_normalized=df_Y_filtered_value_counts/coupons_recommended

#             #get and add coupon acceptance rate, coupon refusal rate, coupons accepted, coupons refused, coupons recommended
#             metric_value_list+=[df_Y_filtered_value_counts_normalized[1] if coupon_acceptance_exists==True else 0, df_Y_filtered_value_counts_normalized[0] if coupon_refusal_exists==True else 0, df_Y_filtered_value_counts[1] if coupon_acceptance_exists==True else 0, df_Y_filtered_value_counts[0] if coupon_refusal_exists==True else 0, coupons_recommended]
#         else:
#             #coupon acceptance rate, coupon refusal rate, coupons accepted, coupons refused, coupons recommended
#             metric_value_list+=[None, None, 0, 0, 0]

#         return metric_value_list
    
    
#     for feature_0_column_name_value in feature_0_column_name_value_list:
        
#         for feature_1_column_name_value in feature_1_column_name_value_list:
            
#             metric_value_list=[]            
#             metric_value_list+=[feature_0_column_name_value, feature_1_column_name_value]
            
#             df_Y_filtered=df.loc[(df.loc[:, feature_column_name_list[0]]==feature_0_column_name_value) &
#                                  (df.loc[:, feature_column_name_list[1]]==feature_1_column_name_value), ['Y']]
            
#             metric_value_2_dimensional_list+=[get_metrics_from_data_frame_Y_filtered(df_Y_filtered, metric_value_list)]
            
#     column_name_list=feature_column_name_list+['Conversion Rate', 'Refusal Rate', 'Conversions', 'Refusals', 'Coupons Recommended']
#     return pd.DataFrame(metric_value_2_dimensional_list, columns=column_name_list).sort_values('Conversion Rate', ascending=False)










def reverse_key_value_of_dictionary(name_dictionary):
    return {name_dictionary[key]:key for key in name_dictionary.keys()}




def get_feature_target_frequency_data_frame(df, feature_column_name='income', target_column_name='Y', append_percentage_true_false=False):
    df = df.value_counts([target_column_name, feature_column_name]).reset_index().pivot(index=feature_column_name, columns=target_column_name).reset_index().droplevel(level=[None,], axis=1).rename(columns={'':feature_column_name, 0:'coupon refused', 1:'coupon accepted'}).loc[:, [feature_column_name, 'coupon accepted', 'coupon refused']]
    if append_percentage_true_false == False:
        return df
    elif append_percentage_true_false == True:
        df.loc[:, 'total'] = df.loc[:, ['coupon accepted', 'coupon refused']].sum(axis=1)
        df.loc[:, 'percentage accepted'] = df.loc[:, 'coupon accepted'] / df.loc[:, 'total'] * 100
        df.loc[:, 'percentage refused'] = df.loc[:, 'coupon refused'] / df.loc[:, 'total'] * 100
        return df



def sort_data_frame(df, feature_column_name, feature_value_order_list, ascending_true_false=True):
    
    feature_column_name_rank = feature_column_name + '_rank'
    value_order_dictionary = dict(zip(feature_value_order_list, range(len(feature_value_order_list))))
    df.loc[:, feature_column_name_rank] = df.loc[:, feature_column_name].map(value_order_dictionary)
    return df.sort_values([feature_column_name_rank], ascending=ascending_true_false)




def plot_vertical_bar_graph(df, feature_column_name, title, xlabel, color_list, figsize, ylabel='Frequency', multibar_column_name_list=['coupon accepted', 'coupon refused'], color_index_list=[3,0], figure_filename=None, dpi=100, xtick_rotation=90, feature_value_dictionary=None):
    
    feature_column_name_unique_value_count = df.loc[:, feature_column_name].drop_duplicates().shape[0]
    
    index_array = np.arange(feature_column_name_unique_value_count)
    bar_width = 0.35
    
    y_upper_limit = df.loc[:, multibar_column_name_list].to_numpy().max() * 1.1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    rects1 = ax.bar(index_array - bar_width/2, df.loc[:, 'coupon accepted'].to_list(), bar_width, label='Coupon accepted', color=color_list[color_index_list[0]])
    rects2 = ax.bar(index_array + bar_width/2, df.loc[:, 'coupon refused'].to_list(), bar_width, label='Coupon refused', color=color_list[color_index_list[1]])

    
    ax.set(xlabel=xlabel, xticks=index_array + bar_width, xlim=[2*bar_width - 1.25, feature_column_name_unique_value_count-0.5], ylim=[0, y_upper_limit],)


    ax.set_xticks(index_array, df.loc[:, feature_column_name].replace(feature_value_dictionary), rotation=xtick_rotation)
    ax.legend()
    
    ax.set_title(label=title, fontsize=18)
    ax.set_ylabel(ylabel=ylabel, fontsize=17)
    ax.set_xlabel(xlabel=xlabel, fontsize=17)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    ax.bar_label(rects1, padding=3, fontsize=13)
    ax.bar_label(rects2, padding=3, fontsize=13)
    
    

    fig.tight_layout()
    
    plt.savefig(figure_filename, bbox_inches='tight', dpi=dpi)

    plt.show()
    


def plot_horizontal_bar_graph(df, feature_column_name, color_list, title, ylabel, multibar_column_name_list=['coupon accepted', 'coupon refused'], xlabel='Frequency', color_index_list=[3, 0], figure_filename=None, dpi=100, figsize=(8,6), x_upper_limit=None, feature_value_dictionary=None):

    #initialize variables
    feature_column_name_unique_value_count = df.loc[:, feature_column_name].drop_duplicates().shape[0]
    
    if x_upper_limit == None:
        x_upper_limit = df.loc[:, multibar_column_name_list].to_numpy().max() * 1.1
    
    index_array = np.arange(feature_column_name_unique_value_count)

    bar_width = 0.4

    #setup subplot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    #setup horizontal bar plots
    rects1 = ax.barh(index_array + bar_width, df.loc[:, multibar_column_name_list[0]], bar_width, color=color_list[color_index_list[0]], label='Coupon accepted', )
    rects2 = ax.barh(index_array, df.loc[:, multibar_column_name_list[1]], bar_width, color=color_list[color_index_list[1]], label='Coupon refused', )

    if feature_value_dictionary != None:
    #setup x and y axis
        ax.set(yticks=index_array + bar_width, yticklabels=df.loc[:, feature_column_name].replace(feature_value_dictionary), ylim=[2*bar_width - 1, feature_column_name_unique_value_count], xlim=[0, x_upper_limit],)
    elif feature_value_dictionary == None:
        ax.set(yticks=index_array + bar_width, yticklabels=df.loc[:, feature_column_name], ylim=[2*bar_width - 1, feature_column_name_unique_value_count], xlim=[0, x_upper_limit],)
    
    ax.legend()

    ax.set_title(label=title, fontsize=18)
    ax.set_ylabel(ylabel=ylabel, fontsize=17)
    ax.set_xlabel(xlabel=xlabel, fontsize=17)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    ax.bar_label(rects1, padding=3, fontsize=14)
    ax.bar_label(rects2, padding=3, fontsize=14)


    fig.tight_layout()
    
    plt.savefig(figure_filename, bbox_inches='tight', dpi=dpi)

    plt.show()



def plot_vertical_stacked_bar_graph(df, feature_column_name, figure_filename, colors, feature_column_name_label, ylabel, xlabel, xtick_dictionary=None, annotation_text_size=11, dpi=100, xtick_rotation=0, annotation_type='frequency', frequency_annotation_round_by_number=-2, y_upper_limit=None, rectangle_annotation_y_offset=None, figsize=None):
    '''
    df : data frame with column frequency and column name as index'''
    if y_upper_limit == None:
        y_upper_limit = df.loc[:, 'total'].max() * 1.1
    if xtick_rotation==None:
        xtick_rotation = 0
    
    feature_column_name_unique_value_count = df.index.drop_duplicates().shape[0]

    bottom = np.zeros(feature_column_name_unique_value_count)

    
    if min(df.index) == 0:
        index_array = np.arange(feature_column_name_unique_value_count,)
    elif min(df.index) == 1: 
        index_array = [index+1 for index in np.arange(feature_column_name_unique_value_count,)]
    else:
        index_array = np.arange(feature_column_name_unique_value_count,)
        


    if figsize == None: figsize = (8,6)
    figure, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    if y_upper_limit != None:
        axes.set_ylim([0, y_upper_limit])
    for i, target_label_column_name in enumerate(df.loc[:, ['coupon accepted', 'coupon refused']].columns):
        axes.bar(df.index, df.loc[:, target_label_column_name], bottom=bottom, label=target_label_column_name.capitalize(), color=colors[i])
        if xtick_dictionary == None:
            axes.set_xticks(index_array, df.index, rotation=xtick_rotation)
        elif xtick_dictionary != None:
            axes.set_xticks(index_array, df.index.map(xtick_dictionary), rotation=xtick_rotation)
        bottom += np.array(df.loc[:, target_label_column_name])

        
    totals = df.loc[:, ['coupon accepted', 'coupon refused']].sum(axis=1)
    y_offset = 4
    for i, total in enumerate(totals):
        axes.text(totals.index[i], total + y_offset, round(total, frequency_annotation_round_by_number), ha='center', weight='bold', size=annotation_text_size)

    if rectangle_annotation_y_offset == None:
        rectangle_annotation_y_offset = -35

    if annotation_type == 'frequency':
        for rectangle in axes.patches:
            axes.text(rectangle.get_x() + rectangle.get_width() / 2, rectangle.get_height()/2 + rectangle.get_y() + rectangle_annotation_y_offset, round(int(rectangle.get_height()), frequency_annotation_round_by_number), ha='center', color='w', weight='bold', size=annotation_text_size)
    elif annotation_type == 'percentage':
        percentage_list = []
        for column_name in df.loc[:, ['percentage accepted', 'percentage refused']].columns:
            percentage_list += df.loc[:, column_name].to_list()
        for rectangle, percentage in zip(axes.patches, percentage_list):
            axes.text(rectangle.get_x() + rectangle.get_width() / 2, rectangle.get_height()/2 + rectangle.get_y() + rectangle_annotation_y_offset, '{:.0f}%'.format(round(percentage, 0)), ha='center', color='w', weight='bold', size=annotation_text_size)

    axes.set_title(str(feature_column_name_label) + ' Frequency Distribution', fontsize=18)
    axes.set_ylabel(ylabel=ylabel, fontsize=17)
    axes.set_xlabel(xlabel=xlabel, fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    axes.legend()
    plt.savefig(figure_filename, bbox_inches='tight', dpi=dpi)

    plt.show()

def plot_horizontal_stacked_bar_graph(df, title, figsize=None, rectangle_annotation_y_offset=None, annotation_text_size=None, x_upper_limit=None, color_list=None,):
    
    #initialize variables
    if figsize == None: 
        figsize=(11, 9)
    if rectangle_annotation_y_offset == None:
        rectangle_annotation_y_offset=-0.24
    if annotation_text_size == None:
        annotation_text_size=13
    
    #create figure and axes
    figure, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    #plot bars on figure axes
    b1=axes.barh(df.index, df.loc[:, 'coupon accepted'].to_list(), color=color_list[3])
    b2=axes.barh(df.index, df.loc[:, 'coupon refused'].to_list(), left=df.loc[:, 'coupon accepted'].to_list(), color=color_list[0])
    
    #plot annotations
    percentage_list = []
    for column_name in df.loc[:, ['percentage accepted', 'percentage refused']].columns:
        percentage_list += df.loc[:, column_name].to_list()
    for rectangle, percentage in zip(axes.patches, percentage_list):
        axes.text(rectangle.get_x() + rectangle.get_width() / 2, rectangle.get_height()/2 + rectangle.get_y() + rectangle_annotation_y_offset, '{:.0f}%'.format(round(percentage, 0)), ha='center', color='w', weight='bold', size=annotation_text_size)
    

    #plt.legend([b1, b2], ["Completed", "Pending"], title="Issues", loc="upper right")
    axes.legend([b1, b2], ['Coupon accepted', 'Coupon refused'])
    axes.set_title(title)
    
    if x_upper_limit != None:
        axes.set_xlim([0, x_upper_limit])

    plt.show()


def plot_vertical_multibar_bar_graph(df, xlabel_column_name, title, color_list, figsize, xlabel=None, ylabel='Frequency', multibar_column_name_list=None, color_index_list=[3,0], figure_filename=None, dpi=100, xtick_rotation=90, xtick_dictionary=None, bar_label_list=None):
    
    feature_column_name_unique_value_count = df.loc[:, xlabel_column_name].drop_duplicates().shape[0]
    
    index_array = np.arange(feature_column_name_unique_value_count)
    bar_width = 0.15
    
    
    y_upper_limit = df.loc[:, multibar_column_name_list].to_numpy().max() * 1.1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    rectangle_dictionary={}
    if bar_label_list == None:
        for index, column_name in zip(range(len(multibar_column_name_list)), multibar_column_name_list):
            rectangle_dictionary[index] = ax.bar(index_array + (index)*bar_width - .3, df.loc[:, column_name].to_list(), bar_width, label=column_name, color=color_list[color_index_list[index]])
    elif bar_label_list != None:
        for index, column_name in zip(range(len(multibar_column_name_list)), multibar_column_name_list):
            rectangle_dictionary[index] = ax.bar(index_array + (index)*bar_width - .3, df.loc[:, column_name].to_list(), bar_width, label=bar_label_list[index], color=color_list[color_index_list[index]])


    ax.set(xlabel=xlabel, xticks=index_array + bar_width, xlim=[2*bar_width - 1.25, feature_column_name_unique_value_count-0.5], ylim=[0, y_upper_limit],)


    ax.set_xticks(index_array, df.loc[:, xlabel_column_name].replace(xtick_dictionary), rotation=xtick_rotation)
    ax.legend()
    
    ax.set_title(label=title, fontsize=18)
    ax.set_ylabel(ylabel=ylabel, fontsize=17)
    ax.set_xlabel(xlabel=xlabel, fontsize=17)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    
    for index in range(len(multibar_column_name_list)):
        ax.bar_label(rectangle_dictionary[index], padding=3, fontsize=13, rotation=90)
    
    

    fig.tight_layout()
    
    plt.savefig(figure_filename, bbox_inches='tight', dpi=dpi)

    plt.show()

##################################################################################################################################
#data preprocessing
##################################################################################################################################

#permissible data split test using Y_test

#how does the Y_test coupon acceptance count of the 1000 runs compare to the stratified Y_test coupon acceptance count???
def get_Y_test_distribution_from_train_test_split_iterations(df, number_of_iterations=1000):
    data_frame_collection = {}
    #get Y_train_number and Y_test_number for 1000 train_test_split iterations 
    for index in range(number_of_iterations):
        _, _, data_frame_collection['Y_train' + str(index)], data_frame_collection['Y_test'+ str(index)] = \
        train_test_split(df.drop(columns=['Y']), df.loc[:, 'Y'], test_size=.2)

        
    Y_test_coupon_acceptance_count_1000_iterations = [data_frame_collection['Y_test' + str(index)].value_counts()[1] for index in range(number_of_iterations)]
    plt.hist(Y_test_coupon_acceptance_count_1000_iterations)
    
    #get Y_test coupon acceptance count
    _, _, data_frame_collection['Y_train'], data_frame_collection['Y_test'] = \
    train_test_split(df.drop(columns=['Y']), df.loc[:, 'Y'], test_size=.2, random_state=200, stratify=df.loc[:, 'Y'])
    
    print('stratified Y_test coupon acceptance count from train_test_split: ' + str(data_frame_collection['Y_test'].value_counts()[1]))
    




##################################################################################################################################

    
    
def column_name_value_sets_equal(df, column_name1, column_name2):
    """
    Parameters
    ----------
    df : DataFrame
        data frame selecting columns from
    column_name1 : str
        first column name to take unique values of
    column_name2 : str
        second column name to take unique values of
        
    Returns
    -------
    returns 1 for column value uniques are equal, otherwise 0
    """
    
    column_name_value_set1 = set(df.loc[:, column_name1].unique())
    
    column_name_value_set2 = set(df.loc[:, column_name2].unique())
    
    if column_name_value_set1 == column_name_value_set2:
        return 1
    elif column_name_value_set1 != column_name_value_set2:
        return 0
    
    
    
    
    
#################################################################################################################################
#Modeling
#################################################################################################################################




def get_data_frame_from_collection(collection_name, column_name='Y_predicted'):
    
    data_frame_list = [pd.DataFrame(collection_name['fold ' + str(fold_number)]) for fold_number in range(5)]

    data_frame = pd.concat(data_frame_list)

    return data_frame.rename(columns={0:column_name})




#Modeling Metrics
def plot_learning_curve(estimator, title, X, y, filename, axes=None, ylim=None, cv=None, n_jobs=None, scoring=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    
    filename_exists = True_False_data_filename_exists(filename)
    print('filename_exists: ' + str(filename_exists))
    if filename_exists == True:
        #read in file
        learning_curve_model_name = return_processed_collection_if_it_exists(filename=filename)
        
        train_scores_mean = learning_curve_model_name['learning_curve_mean_std']['test_scores_mean']
        train_scores_std = learning_curve_model_name['learning_curve_mean_std']['train_scores_std']
        test_scores_mean = learning_curve_model_name['learning_curve_mean_std']['test_scores_mean']
        test_scores_std = learning_curve_model_name['learning_curve_mean_std']['test_scores_std']
        fit_times_mean = learning_curve_model_name['learning_curve_mean_std']['fit_times_mean']
        fit_times_std = learning_curve_model_name['learning_curve_mean_std']['fit_times_std']
        
        train_sizes = learning_curve_model_name['learning_curve_raw']['train_sizes']
        train_scores = learning_curve_model_name['learning_curve_raw']['train_scores']
        test_scores = learning_curve_model_name['learning_curve_raw']['test_scores']
        fit_times = learning_curve_model_name['learning_curve_raw']['fit_times']

    else:
        #get learning curve stats
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True,)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
        
        #save it
        learning_curve_raw = {'train_sizes':train_sizes, 'train_scores':train_scores, 'test_scores':test_scores, 'fit_times':fit_times,}
        learning_curve_mean_std = {'train_sizes':train_sizes, 'train_scores_mean':train_scores_mean, 'train_scores_std':train_scores_std, 'test_scores_mean':test_scores_mean, 'test_scores_std':test_scores_std, 'fit_times_mean':fit_times_mean, 'fit_times_std':fit_times_std,}
        learning_curve_dictionary_raw_mean_std = {'learning_curve_raw':learning_curve_raw, 'learning_curve_mean_std':learning_curve_mean_std}
        learning_curve_model_name = save_and_return_collection(learning_curve_dictionary_raw_mean_std, filename=filename)

    
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,  alpha=0.1, color="r",)
    
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g",)
    
    axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    
    axes[0].plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    
    axes[0].legend(loc="best")
    
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,  fit_times_mean + fit_times_std, alpha=0.1,)
    
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(fit_time_sorted, test_scores_mean_sorted - test_scores_std_sorted, test_scores_mean_sorted + test_scores_std_sorted, alpha=0.1,)
    
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt, learning_curve_model_name




#################################################################################################################################
#Model Train Results
#################################################################################################################################

def precision_recall_auc_plot(y_true=None, probas_pred=None, model_name_coupon_type=None, precision_lower_upper=None, recall_lower_upper=None):
    markersize=1
    linewidth=1
    
    #calculate precision, recall, and decision threshold
    precision_array, recall_array, decision_threshold_array = precision_recall_curve(y_true=y_true, probas_pred=probas_pred)
    
    #get precision, recall, decsion threshold data frame

    #decision thresholds by precision .9
    decision_threshold_array = np.append(0, decision_threshold_array)
    df_decision_threshold_precision_recall = pd.DataFrame({'decision_threshold':decision_threshold_array, 'precision':precision_array, 'recall':recall_array})
    df_decision_threshold_precision_recall_filtered_precision_dot9 = df_decision_threshold_precision_recall.loc[(df_decision_threshold_precision_recall.loc[:,'precision'] > precision_lower_upper[0]) & (df_decision_threshold_precision_recall.loc[:,'precision'] < precision_lower_upper[1]), :]
    
    print(str(model_name_coupon_type) + ' .9 precision \n' + 'decision thresholds ' + str(df_decision_threshold_precision_recall_filtered_precision_dot9.loc[:, 'decision_threshold'].to_list()) + '\n')
    
    
    #decision thresholds by recall .8
    df_decision_threshold_precision_recall_filtered_recall_dot8 = df_decision_threshold_precision_recall.loc[(df_decision_threshold_precision_recall.loc[:,'recall'] > recall_lower_upper[0]) & (df_decision_threshold_precision_recall.loc[:,'recall'] < recall_lower_upper[1]), :]
    
    print(str(model_name_coupon_type) + ' .8 recall \n' + 'decision thresholds ' + str(df_decision_threshold_precision_recall_filtered_recall_dot8.loc[:, 'decision_threshold'].to_list()) + '\n')
    
    
    #calculate precision-recall auc
    auc_score = auc(recall_array, precision_array)

    #plot the precision-recall curve
    plt.plot(recall_array, precision_array, marker='.', markersize=markersize, linewidth=linewidth, label=str(model_name_coupon_type) +' AUC=' + str(round(auc_score, 3)))

    plt.xticks([.0, .1 ,.2, .3 ,.4, .5, .6 ,.7, .8, .9, 1 ])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall')
    plt.legend()
    
    
    
    
def precision_recall_auc_plots(df_collection, coupon_venue_type=None, precision_lower_upper=None, recall_lower_upper=None):
    print('row count: ' + str(df_collection[coupon_venue_type].shape[0]))
    
    precision_recall_auc_plot(y_true=df_collection[coupon_venue_type].loc[:, 'Y'], 
                              probas_pred=df_collection[coupon_venue_type].loc[:, 'Y_random_forest_prediction_probability'], 
                              model_name_coupon_type='Random Forest ' + str(coupon_venue_type) + ' Coupon',
                              precision_lower_upper=precision_lower_upper,
                              recall_lower_upper=recall_lower_upper)

    precision_recall_auc_plot(y_true=df_collection[coupon_venue_type].loc[:, 'Y'], 
                              probas_pred=df_collection[coupon_venue_type].loc[:, 'Y_gradient_boosting_prediction_probability'], 
                              model_name_coupon_type='Gradient Boosting ' + str(coupon_venue_type) + ' Coupon',
                              precision_lower_upper=precision_lower_upper,
                              recall_lower_upper=recall_lower_upper)

    

    
    
    
#################################################################################################################################
#Model Train or Test Results
#################################################################################################################################

def get_model_predictions_from_prediction_probabilities_and_decision_threshold_proportion_metric_estimated(df, model_precision_column_name, model_recall_column_name, model_decision_threshold_column_name, df_Y_train_test_model_prediction_probability, model_proportion_precision=None, model_proportion_recall=None, train_test='test'):
    '''
    df: contains the model decision threshold per model precision and recall
    model_proportion_precision: level of precision you want predictions to have
    model_precision_column_name: name of the precision column in df
    model_decision_threshold_column_name: name of the decision threshold column in df
    df_Y_train_test_model_prediction_probability: prediction probabilities for some target data (e.g. Y_test) by the same model type (e.g. random forest classifier) that produced df'''
    
    if model_proportion_precision != None:
        #sort df by model recall descending
        df=df.sort_values(model_recall_column_name, ascending=False)
    
        #get first decision threshold with at least 90% precision from random forest classifier on precision-recall curve
        model_decision_threshold_number_precision = df.loc[df.loc[:, model_precision_column_name] >= model_proportion_precision, model_decision_threshold_column_name].iloc[0]

        #get y_test predictions from prediction probabilties and decision threshold 90% precision estimated
        df_Y_train_test_model_prediction_probability = df_Y_train_test_model_prediction_probability.to_list()
        Y_train_test_predicted_list = [1 if prediction_probability > model_decision_threshold_number_precision else 0 for prediction_probability in df_Y_train_test_model_prediction_probability]
        df_Y_train_test_predicted = pd.DataFrame(Y_train_test_predicted_list, columns=['Y_'+str(train_test)+'_predicted'])
        
        return df_Y_train_test_predicted
    
    elif model_proportion_recall != None:
        #sort df by model precision descending
        df=df.sort_values(model_precision_column_name, ascending=False)
        
        #get first decision threshold with at least 80% recall from gradient boosting classifier on precision-recall curve
        model_decision_threshold_number_recall = df.loc[df.loc[:, model_recall_column_name] >= model_proportion_recall, model_decision_threshold_column_name].iloc[0]
        
        #get y_test predictions from prediction probabilties and decision threshold 80% recall estimated
        df_Y_train_test_model_prediction_probability = df_Y_train_test_model_prediction_probability.to_list()
        Y_train_test_predicted_list = [1 if prediction_probability > model_decision_threshold_number_recall else 0 for prediction_probability in df_Y_train_test_model_prediction_probability]
        df_Y_train_test_predicted = pd.DataFrame(Y_train_test_predicted_list, columns=['Y_'+str(train_test)+'_predicted'])
        
        return df_Y_train_test_predicted 


    

def get_survey_coupon_recommendations_by_recall_estimate(number_of_predictions, recall_estimated, random_state=200, train_test='test'):

    np.random.seed(random_state)
    class_1_probability=recall_estimated
    class_0_probability=1-class_1_probability
    print(class_1_probability)
    print(class_0_probability)

    Y_train_test_survey_recall_estimate_predicted=np.random.choice([0, 1], size=number_of_predictions, p=[class_0_probability, class_1_probability])

    df_Y_train_test_survey_number_recall_estimate_predicted=pd.DataFrame(Y_train_test_survey_recall_estimate_predicted, columns=['Y_'+str(train_test)+'_survey_'+str(round(recall_estimated*100))+'_recall_estimate_predicted'])

    return df_Y_train_test_survey_number_recall_estimate_predicted


##############################################################################################################################
#Model Test Results
##############################################################################################################################







def get_metric_multiple_index(proportion_or_percentage):
    
    metric_index_value_list=['Conversion Rate', 'Recall', proportion_or_percentage.capitalize()+' of Conversions', 'Conversions', proportion_or_percentage.capitalize()+' of Coupons Recommended', 'Coupons Recommended', 
                               'Conversions to Base Survey Coupons Recommended Ratio',
                               'Conversions to Survey Conversions Ratio',
                               'Coupons Recommended to Survey Coupons Recommended Ratio', 
                               'Coupons Recommended to Base Survey Coupons Recommended Ratio',]

    model_survey_index_value_list=['Model' for index in range(len(metric_index_value_list))]+\
                                  ['Survey' for index in range(len(metric_index_value_list))]+\
                                  ['Model-Survey Difference' for index in range(len(metric_index_value_list))]

    metric_index_value_list_tripled=metric_index_value_list+metric_index_value_list+metric_index_value_list

    tuple_list = list(zip(model_survey_index_value_list, metric_index_value_list_tripled))
    multiple_index=pd.MultiIndex.from_tuples(tuple_list)
    return multiple_index












#################################################################################################################################
#Get Survey or Model Average Coupon Recommendation Cost Estimated


def get_survey_or_model_metrics_conversions_conversion_rate_recall_coupons_recommended(df, column_name_y_actual, column_name_y_predicted, feature_column_name_filter, feature_column_name_filter_value_two_dimensional_list):
    """
    Calculate metrics (i.e. Conversions, Conversion Rate, Recall, and Coupons Recommended) for each rows filter Data Frame.

    Parameters:
    df: Input DataFrame.
    column_name_y_actual: Column name of true target values.
    column_name_y_predicted: Column name of predicted target values.
    feature_column_name_filter: Feature column name to filter on.
    feature_column_name_filter_value_two_dimensional_list: Two-dimensional list of feature values to filter on.

    Returns:
    metric_value_two_dimensional_list.
    """
    
    metric_value_two_dimensional_list=[]
    
    for feature_column_name_filter_value_list in feature_column_name_filter_value_two_dimensional_list:
        
        metric_value_list=[]

        df_filtered=df.loc[df.loc[:, feature_column_name_filter].isin(feature_column_name_filter_value_list), :]
        
        y_true=df_filtered.loc[:, column_name_y_actual]
        y_predicted=df_filtered.loc[:, column_name_y_predicted]
        
        true_negatives, false_positives, false_negatives, true_positives=confusion_matrix(y_true=y_true, y_pred=y_predicted, labels=None, sample_weight=None, normalize=None).ravel()
                
        #get number of conversions
        metric_value_list+=[true_positives]
        
        #get conversion rate
        metric_value_list+=[true_positives/(true_positives+false_positives)*100]
        
        #get recall
        metric_value_list+=[true_positives/(true_positives+false_negatives)*100]
        
        #get coupons recommended
        metric_value_list+=[true_positives+false_positives]
        
        
        metric_value_two_dimensional_list+=[metric_value_list]
    
    return metric_value_two_dimensional_list


def get_average_sale_and_survey_100_recall_metrics_per_coupon_venue_type(df_y_train_model_name_predicted_y_train_survey_recall_estimate_predicted_y_actual_coupon_venue_type,
                                                                         column_name_y_predicted,
                                                                         column_name_y_actual,
                                                                         feature_column_name_filter,
                                                                         feature_column_name_filter_value_two_dimensional_list,
                                                                         feature_column_name_filter_value_list_dictionary_key_list,
                                                                         venue_type_average_sale_dictionary={'Coffee House':[5.50], 'Bar':[15], 'Takeout':[15], 'Low-Cost Restaurant':[12], 'Mid-Range Restaurant':[35],}):

    metric_value_two_dimensional_list=\
    get_survey_or_model_metrics_conversions_conversion_rate_recall_coupons_recommended(df=df_y_train_model_name_predicted_y_train_survey_recall_estimate_predicted_y_actual_coupon_venue_type,
                                                                                       column_name_y_predicted=column_name_y_predicted, 
                                                                                       column_name_y_actual=column_name_y_actual, 
                                                                                       feature_column_name_filter=feature_column_name_filter,
                                                                                       feature_column_name_filter_value_two_dimensional_list=feature_column_name_filter_value_two_dimensional_list,)

    #convert to Data Frame from metric values two dimensional list
    metric_name_list=['Conversions', 'Conversion Rate', 'Percentage of Conversions Captured', 'Coupons Recommended']
    df_train_survey_100_recall_metrics=pd.DataFrame(metric_value_two_dimensional_list, index=feature_column_name_filter_value_list_dictionary_key_list, columns=metric_name_list)

    
    #Add Venue Type Average Sale to Survey 100% Recall Metrics (Per Coupon Venue Type) Table

    #get Data Frame Venue Type Average Sale
    df_venue_type_average_sale=pd.DataFrame.from_dict(venue_type_average_sale_dictionary)


    #Combine Survey 100 Percent Recall Estimated Metrics and Venue Type Average Sale
    df_train_survey_100_recall_metrics_venue_type_average_sale=pd.merge(df_train_survey_100_recall_metrics.reset_index(), 
                                                                        df_venue_type_average_sale.T.reset_index(),
                                                                        how='outer').rename(columns={'index':'Venue Type',0:'Average Sale Estimated'})
    
    return df_train_survey_100_recall_metrics_venue_type_average_sale



    
def extract_and_add_revenue_estimated_ad_spend_estimated_average_coupon_recommendation_cost_estimated(df):
    
    #get Revenue Estimated
    df.loc[:, 'Revenue Estimated']=df.loc[:, 'Average Sale Estimated']* df.loc[:, 'Conversions']

    #get Ad Spend Estimated
    df.loc[:, 'Ad Spend Estimated']=df.loc[:, 'Revenue Estimated']*.2

    #get Average Coupon Recommendation Cost Estimated
    df.loc[:, 'Average Coupon Recommendation Cost Estimated']=df.loc[:, 'Ad Spend Estimated']/df.loc[:, 'Coupons Recommended']
    
    return df


def get_metric_multiple_index_ROAS(model_survey_index_value_list, metric_index_value_list):
    
    tuple_list = list(zip(model_survey_index_value_list, metric_index_value_list))
    multiple_index=pd.MultiIndex.from_tuples(tuple_list)
    return multiple_index








def get_survey_or_model_average_coupon_recommendation_cost_estimated(df_y_train_model_name_predicted_y_train_survey_recall_estimate_predicted_y_actual_coupon_venue_type,
                                                                     column_name_y_predicted,
                                                                     column_name_y_actual,
                                                                     feature_column_name_filter,
                                                                     feature_column_name_filter_value_two_dimensional_list,
                                                                     feature_column_name_filter_value_list_dictionary_key_list,
                                                                     venue_type_average_sale_dictionary,
                                                                     model_survey='Survey'):

    #get average sale and survey/model number metric estimated metrics
    df_train_survey_model_number_metric_estimate_metrics_conversions_conversion_rate_recall_coupons_recommended_venue_type_average_sale=\
    get_average_sale_and_survey_100_recall_metrics_per_coupon_venue_type(df_y_train_model_name_predicted_y_train_survey_recall_estimate_predicted_y_actual_coupon_venue_type=df_y_train_model_name_predicted_y_train_survey_recall_estimate_predicted_y_actual_coupon_venue_type,
                                                                         column_name_y_predicted=column_name_y_predicted,
                                                                         column_name_y_actual=column_name_y_actual,
                                                                         feature_column_name_filter=feature_column_name_filter,
                                                                         feature_column_name_filter_value_two_dimensional_list=feature_column_name_filter_value_two_dimensional_list,
                                                                         feature_column_name_filter_value_list_dictionary_key_list=feature_column_name_filter_value_list_dictionary_key_list,
                                                                         venue_type_average_sale_dictionary=venue_type_average_sale_dictionary)

    #Get (by Venue Type) Average Coupon Recommendation Cost Estimated and Average Sale Estimated DataFrame

    df_train_survey_model_number_metric_estimate_metrics_venue_type_average_sale_revenue_estimated_ad_spend_estimated_average_coupon_recommendation_cost_estimated=\
    extract_and_add_revenue_estimated_ad_spend_estimated_average_coupon_recommendation_cost_estimated(df=df_train_survey_model_number_metric_estimate_metrics_conversions_conversion_rate_recall_coupons_recommended_venue_type_average_sale)

    #filter for Venue Type, Average Coupon Recommendation Cost, and Average Sale DataFrame
    column_name_list=['Venue Type', 'Average Coupon Recommendation Cost Estimated', 'Average Sale Estimated']
    df_train_survey_model_number_metric_estimate_average_coupon_recommendation_cost_estimated_venue_type=df_train_survey_model_number_metric_estimate_metrics_venue_type_average_sale_revenue_estimated_ad_spend_estimated_average_coupon_recommendation_cost_estimated.loc[:, column_name_list]


    #get value list from average coupon recommendation cost, average sale estimated data frame
    venue_type_average_coupon_recommendation_cost_estimated_venue_type_sale_estimated_two_dimensional_list=[list(df_train_survey_model_number_metric_estimate_average_coupon_recommendation_cost_estimated_venue_type.set_index('Venue Type').T.reset_index(drop=True).values[0])]+\
                                                                                                          [list(df_train_survey_model_number_metric_estimate_average_coupon_recommendation_cost_estimated_venue_type.set_index('Venue Type').T.reset_index(drop=True).values[1])]


    multiple_index_ROAS=get_metric_multiple_index_ROAS(model_survey_index_value_list=[model_survey, model_survey], metric_index_value_list=['Average Coupon Recommendation Cost Estimated', 'Average Sale Estimated'])


    df_survey_model_coupon_recommendation_cost_estimated_sale_estimated=\
    pd.DataFrame(data=venue_type_average_coupon_recommendation_cost_estimated_venue_type_sale_estimated_two_dimensional_list, 
                 index=multiple_index_ROAS,
                 columns=feature_column_name_filter_value_list_dictionary_key_list)

    return df_survey_model_coupon_recommendation_cost_estimated_sale_estimated


#################################################################################################################################



  


    
    
    
    







def get_model_and_survey_metrics(df, model_y_predicted_column_name, survey_number_recall_estimated_y_predicted_column_name, y_predicted_column_name_base_survey, y_actual_column_name, feature_column_name_filter, feature_column_name_filter_value_list, metrics_column_name_list=None,):
    '''
    df_feature_column_name_unfiltered: data frame that contains model y predicted, y actual, and feature for filtering
    model_y_predicted_column_name: name of the column containing model y predicted
    '''

    metric_list=[]
    
    #get filtered data frame
    df_feature_column_name_filtered = df.loc[df.loc[:, feature_column_name_filter].isin(feature_column_name_filter_value_list), :]

    
    def get_model_or_survey_metric_list_by_y_predicted_column_name(df_feature_column_name_unfiltered, df_feature_column_name_filtered, y_predicted_column_name, y_predicted_column_name_baseline, y_predicted_column_name_base_survey, y_actual_column_name):
        
        '''
        y_predicted_column_name_baseline: y_predicted from the survey number metric estimated'''
 
        metric_list=[]
        
        #get unfiltered y_true and y_predicted
        y_true_feature_column_name_filtered=df_feature_column_name_filtered.loc[:, y_actual_column_name]
        y_predicted_feature_column_name_filtered=df_feature_column_name_filtered.loc[:, y_predicted_column_name]
        
        #get filtered y_true and y_predicted
        y_true_feature_column_name_unfiltered=df_feature_column_name_unfiltered.loc[:, y_actual_column_name]
        y_predicted_feature_column_name_unfiltered=df_feature_column_name_unfiltered.loc[:, y_predicted_column_name]
        
        #get survey as a baseline y_predicted
        y_predicted_baseline_feature_column_name_filtered=df_feature_column_name_filtered.loc[:, y_predicted_column_name_baseline]

        #get base survey y_predicted
        y_predicted_base_survey_feature_column_name_filtered=df_feature_column_name_filtered.loc[:, y_predicted_column_name_base_survey]

        
        #get precision
        metric_list += [precision_score(y_true=y_true_feature_column_name_filtered, y_pred=y_predicted_feature_column_name_filtered)]

        #get recall
        metric_list += [recall_score(y_true=y_true_feature_column_name_filtered, y_pred=y_predicted_feature_column_name_filtered)]
        
        
        #tn, fp, fn, tp filtered
        confusion_matrix_ndarray_feature_column_name_filtered = confusion_matrix(y_true=y_true_feature_column_name_filtered, y_pred=y_predicted_feature_column_name_filtered)
        tn_feature_column_name_filtered, fp_feature_column_name_filtered, fn_feature_column_name_filtered, tp_feature_column_name_filtered = confusion_matrix_ndarray_feature_column_name_filtered.ravel()
        
        #tn, fp, fn, tp unfiltered
        confusion_matrix_ndarray_feature_column_name_unfiltered = confusion_matrix(y_true=df_feature_column_name_unfiltered.loc[:, y_actual_column_name], y_pred=df_feature_column_name_unfiltered.loc[:, y_predicted_column_name])
        tn_feature_column_name_unfiltered, fp_feature_column_name_unfiltered, fn_feature_column_name_unfiltered, tp_feature_column_name_unfiltered = confusion_matrix_ndarray_feature_column_name_unfiltered.ravel()
        
        #tn, fp, fn, tp baseline filtered
        confusion_matrix_ndarray_feature_column_name_filtered_baseline = confusion_matrix(y_true=y_true_feature_column_name_filtered, y_pred=y_predicted_baseline_feature_column_name_filtered)
        tn_feature_column_name_filtered_baseline, fp_feature_column_name_filtered_baseline, fn_feature_column_name_filtered_baseline, tp_feature_column_name_filtered_baseline = confusion_matrix_ndarray_feature_column_name_filtered_baseline.ravel()
        
        #tn, fp, fn, tp base survey filtered
        confusion_matrix_ndarray_feature_column_name_filtered_base_survey = confusion_matrix(y_true=y_true_feature_column_name_filtered, y_pred=y_predicted_base_survey_feature_column_name_filtered)
        tn_feature_column_name_filtered_base_survey, fp_feature_column_name_filtered_base_survey, fn_feature_column_name_filtered_base_survey, tp_feature_column_name_filtered_base_survey = confusion_matrix_ndarray_feature_column_name_filtered_base_survey.ravel()
        

        #get conversions proportion
        metric_list += [tp_feature_column_name_filtered/tp_feature_column_name_unfiltered]
        
        #get conversions
        metric_list += [tp_feature_column_name_filtered]
        
        #get coupons recommended proportion
        metric_list += [(tp_feature_column_name_filtered+fp_feature_column_name_filtered)/(tp_feature_column_name_unfiltered+fp_feature_column_name_unfiltered)]
        
        #get coupons recommended
        metric_list += [tp_feature_column_name_filtered+fp_feature_column_name_filtered]
        
        #get conversions to base survey coupons recommended ratio
        base_survey_coupons_recommended=tp_feature_column_name_filtered_base_survey+fp_feature_column_name_filtered_base_survey
        metric_list += [tp_feature_column_name_filtered/(base_survey_coupons_recommended)]

        #get conversions to survey conversions ratio
        metric_list += [(tp_feature_column_name_filtered)/(tp_feature_column_name_filtered_baseline)]
        
        #get conversions to base survey conversions ratio
        #metric_list += [(tp_feature_column_name_filtered)/(tp_feature_column_name_filtered_base_survey)]
        
        
        #get coupons recommended to survey coupons recommended ratio
        metric_list += [(tp_feature_column_name_filtered+fp_feature_column_name_filtered)/(tp_feature_column_name_filtered_baseline+fp_feature_column_name_filtered_baseline)]
        
        #get coupons recommended to base survey coupons recommended ratio
        metric_list += [(tp_feature_column_name_filtered+fp_feature_column_name_filtered)/(base_survey_coupons_recommended)]
        

        
        return metric_list
    
    
    #get model metric list
    model_metric_list=get_model_or_survey_metric_list_by_y_predicted_column_name(df_feature_column_name_unfiltered=df, df_feature_column_name_filtered=df_feature_column_name_filtered, y_predicted_column_name=model_y_predicted_column_name, y_predicted_column_name_baseline=survey_number_recall_estimated_y_predicted_column_name, y_predicted_column_name_base_survey=y_predicted_column_name_base_survey, y_actual_column_name=y_actual_column_name,)
    
    #get survey metric list
    survey_metric_list=get_model_or_survey_metric_list_by_y_predicted_column_name(df_feature_column_name_unfiltered=df, df_feature_column_name_filtered=df_feature_column_name_filtered, y_predicted_column_name=survey_number_recall_estimated_y_predicted_column_name, y_predicted_column_name_baseline=survey_number_recall_estimated_y_predicted_column_name, y_predicted_column_name_base_survey=y_predicted_column_name_base_survey, y_actual_column_name=y_actual_column_name)
    

    metric_list=model_metric_list+survey_metric_list

    return metric_list


def calculate_and_add_model_survey_difference(df_model_survey_metrics, multiple_index):
    
    column_name_list=df_model_survey_metrics.columns.to_list()
    
    #calculate and add model-survey difference metrics
    df_model_survey_difference_metrics=df_model_survey_metrics.iloc[0:10].reset_index().loc[:, column_name_list]-df_model_survey_metrics.iloc[10:20].reset_index().loc[:, column_name_list]

    df_model_survey_difference_metrics.index=multiple_index[20:30]

    #combine model survey difference metrics to model and survey metrics
    return pd.concat([df_model_survey_metrics, df_model_survey_difference_metrics], axis=0)    







def get_metric_quatiles_from_number_subsample_replicates_metrics(df, quantile_lower_upper_list):

    return df.quantile(q=quantile_lower_upper_list, axis=1, numeric_only=True, interpolation='linear').T

def get_metric_confidence_interval_table_by_feature_column_name_filter_value_list_dictionary_key(df_y_train_test_model_name_predicted_y_train_test_survey_recall_estimate_predicted_y_actual_feature_column_name_filter, feature_column_name_filter, feature_column_name_filter_value_list_dictionary_key, feature_column_name_filter_value_list_dictionary, multiple_index, number_of_replicates, quantile_lower_upper_list, model_type, survey_number_recall_estimated_y_predicted_column_name, filename_version, save_metric_replicates_feature_column_name_filter_value_list_dictionary_key_list=[], train_test='test', sample_size=None):

    
    def get_number_model_and_survey_metric_replicates_from_number_subsamples(df, number_of_replicates, model_type, survey_number_recall_estimated_y_predicted_column_name, feature_column_name_filter, feature_column_name_filter_value_list, sample_size=None,):
        metric_list_collection = {}

        np.random.seed(seed=200)
        for index in range(number_of_replicates):

            if sample_size == None:
                df_bootstrap_sample=df_y_train_test_model_name_predicted_y_train_test_survey_recall_estimate_predicted_y_actual_feature_column_name_filter.sample(n=None, frac=1, replace=True, weights=None, random_state=None, axis=0, ignore_index=False)
            elif sample_size != None:
                df_bootstrap_sample=df_y_train_test_model_name_predicted_y_train_test_survey_recall_estimate_predicted_y_actual_feature_column_name_filter.sample(n=sample_size, frac=None, replace=True, weights=None, random_state=None, axis=0, ignore_index=False)
                
            metric_list_collection[index]=get_model_and_survey_metrics(df=df_bootstrap_sample, model_y_predicted_column_name='Y_'+train_test+'_'+model_type+'_predicted', survey_number_recall_estimated_y_predicted_column_name=survey_number_recall_estimated_y_predicted_column_name, y_predicted_column_name_base_survey='Y_'+train_test+'_survey_100_recall_estimate_predicted', y_actual_column_name='Y', feature_column_name_filter=feature_column_name_filter, feature_column_name_filter_value_list=feature_column_name_filter_value_list, metrics_column_name_list=None,)

        df_model_number_metric_estimate_metrics_feature_filter_number_bootstrap_replicates_metric=pd.DataFrame(metric_list_collection, index=multiple_index[0:20])

        return df_model_number_metric_estimate_metrics_feature_filter_number_bootstrap_replicates_metric


    #get model and survey metric replicates from the 10,000 nonparametric or parametric subsamples
    if sample_size==None:
        df_filename='df_'+train_test+'_'+model_type+'_number_metric_estimated_'+str(number_of_replicates)+'_metric_replicates_from_'+str(number_of_replicates)+'_nonparametric_subsamples_'+ feature_column_name_filter_value_list_dictionary_key.lower().replace(" ", "_") +'_'+filename_version + '.csv'
    elif sample_size!=None:
        df_filename='df_'+train_test+'_'+model_type+'_number_metric_estimated_'+str(number_of_replicates)+'_metric_replicates_from_'+str(number_of_replicates)+'_parametric_subsamples_n_'+str(sample_size)+'_'+feature_column_name_filter_value_list_dictionary_key.lower().replace(" ", "_") +'_'+filename_version + '.csv'
    
    df_readback=return_processed_data_file_if_it_exists_v2(filename=df_filename, column_name_row_integer_location_list=[0,], index_column_integer_location_list=[0,1])
    if df_readback.empty != True:
        df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics=df_readback
    else:
        df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics=get_number_model_and_survey_metric_replicates_from_number_subsamples(df=df_y_train_test_model_name_predicted_y_train_test_survey_recall_estimate_predicted_y_actual_feature_column_name_filter, number_of_replicates=number_of_replicates, model_type=model_type, survey_number_recall_estimated_y_predicted_column_name=survey_number_recall_estimated_y_predicted_column_name, feature_column_name_filter=feature_column_name_filter, feature_column_name_filter_value_list=feature_column_name_filter_value_list_dictionary[feature_column_name_filter_value_list_dictionary_key], sample_size=sample_size,)


        #calculate metric replicate differences and append
        df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics=calculate_and_add_model_survey_difference(df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics, multiple_index)

        if feature_column_name_filter_value_list_dictionary_key in save_metric_replicates_feature_column_name_filter_value_list_dictionary_key_list:

            #save it
            df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics=save_and_return_data_frame_v2(df=df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics, filename=df_filename, index=True)


    
    #get 95% confidence interval upper and lower quantiles
    model_survey_quantiles_of_subsample_replicates_metrics=get_metric_quatiles_from_number_subsample_replicates_metrics(df=df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics, quantile_lower_upper_list=quantile_lower_upper_list)

    #rename columns to multi index
    column_name_number_confidence_interval=str((round((quantile_lower_upper_list[1] - quantile_lower_upper_list[0])*100)))+'%'+' Confidence Interval'
    multiple_index=pd.MultiIndex.from_tuples([(column_name_number_confidence_interval, quantile_lower_upper_list[0]), 
                                              (column_name_number_confidence_interval, quantile_lower_upper_list[1])],)
    model_survey_quantiles_of_subsample_replicates_metrics.columns=multiple_index




    def convert_quantile_columns_to_confidence_interval_column(model_survey_quantiles_of_subsample_replicates_metrics, feature_column_name_filter_value_list_dictionary_key):
        #transpose to wide
        model_survey_quantiles_of_subsample_replicates_metrics=model_survey_quantiles_of_subsample_replicates_metrics.T

        #convert positive counts to int64
        model_survey_quantiles_of_subsample_replicates_metrics\
        .loc[:, model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>1]=\
        model_survey_quantiles_of_subsample_replicates_metrics\
        .loc[:, model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>1].astype('int64')
        model_survey_quantiles_of_subsample_replicates_metrics
        
        #convert negative counts to int64
        model_survey_quantiles_of_subsample_replicates_metrics\
        .loc[:, model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<-1]=\
        model_survey_quantiles_of_subsample_replicates_metrics\
        .loc[:, model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<-1].astype('int64')
        model_survey_quantiles_of_subsample_replicates_metrics


        if convert_proportions_to_percentages=='yes':
            
            #convert to percentages from proportions in [0,1) and round to number of signficant digits
            model_survey_quantiles_of_subsample_replicates_metrics\
            .loc[:,(model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>=0) &
                 (model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<1)]=\
            round(model_survey_quantiles_of_subsample_replicates_metrics\
            .loc[:,(model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>=0) &
                 (model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<1)]*100, rate_number_of_significant_digits-2)
            
            
            #convert to percentages from proportions in (-1,0] and round to number of signficant digits
            model_survey_quantiles_of_subsample_replicates_metrics\
            .loc[:,(model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<=0) &
                 (model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>-1)]=\
            round(model_survey_quantiles_of_subsample_replicates_metrics\
            .loc[:,(model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<=0) &
                 (model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>-1)]*100, rate_number_of_significant_digits-2)


            #convert to 100 percent from proportions 1
            model_survey_quantiles_of_subsample_replicates_metrics.loc[:,model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]==1]=100

            #convert to values to type string
            model_survey_quantiles_of_subsample_replicates_metrics=model_survey_quantiles_of_subsample_replicates_metrics.astype('string')



            #add '%' to non-count column names
            column_name_count_list=[('Model', 'Conversions'), ('Model', 'Coupons Recommended'), ('Survey', 'Conversions'), ('Survey', 'Coupons Recommended'), ('Model-Survey Difference', 'Conversions'), ('Model-Survey Difference', 'Coupons Recommended')]
            column_name_list_metric_not_count=[column_name for column_name in model_survey_quantiles_of_subsample_replicates_metrics.columns.to_list() if not column_name in column_name_count_list]

            model_survey_quantiles_of_subsample_replicates_metrics.loc[:,column_name_list_metric_not_count]=\
            model_survey_quantiles_of_subsample_replicates_metrics.loc[:,column_name_list_metric_not_count]+'%'


            #transpose to tall
            model_survey_quantiles_of_subsample_replicates_metrics=model_survey_quantiles_of_subsample_replicates_metrics.T

            multiple_index=get_metric_multiple_index(proportion_or_percentage='percentage')[0:30]
            model_survey_quantiles_of_subsample_replicates_metrics.index=multiple_index


        elif convert_proportions_to_percentages=='no':

            model_survey_quantiles_of_subsample_replicates_metrics .loc[:,(model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>=0) & (model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<1)]=round(model_survey_quantiles_of_subsample_replicates_metrics.loc[:,(model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>=0) & (model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<1)], rate_number_of_significant_digits)

            model_survey_quantiles_of_subsample_replicates_metrics.loc[:,model_survey_quantiles_of_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]==1]=1

            #convert to values to type string
            model_survey_quantiles_of_subsample_replicates_metrics=model_survey_quantiles_of_subsample_replicates_metrics.astype('string')


            #transpose to tall
            model_survey_quantiles_of_subsample_replicates_metrics=model_survey_quantiles_of_subsample_replicates_metrics.T


        if keep_quantiles_confidence_interval_both=='quantiles':
            return model_survey_quantiles_of_subsample_replicates_metrics

        else:
            #get 95% confidence interval column from 2.5% and 97.5% quantile columns

            model_survey_number_confidence_interval_of_subsample_replicates_metrics= '('+model_survey_quantiles_of_subsample_replicates_metrics.loc[:, (column_name_number_confidence_interval, quantile_lower_upper_list[0])]+', '+model_survey_quantiles_of_subsample_replicates_metrics.loc[:, (column_name_number_confidence_interval, quantile_lower_upper_list[1])]+')'


            model_survey_number_confidence_interval_of_subsample_replicates_metrics=model_survey_number_confidence_interval_of_subsample_replicates_metrics.to_frame()
            model_survey_number_confidence_interval_of_subsample_replicates_metrics.columns=pd.MultiIndex.from_tuples([(column_name_number_confidence_interval, feature_column_name_filter_value_list_dictionary_key)])

            if keep_quantiles_confidence_interval_both=='confidence_interval':
                return model_survey_number_confidence_interval_of_subsample_replicates_metrics


            elif keep_quantiles_confidence_interval_both=='both':
                model_survey_quantile_and_number_confidence_interval_of_subsample_replicates_metrics=pd.concat([model_survey_quantiles_of_subsample_replicates_metrics, model_survey_number_confidence_interval_of_subsample_replicates_metrics],axis=1)

                return model_survey_quantile_and_number_confidence_interval_of_subsample_replicates_metrics



    keep_quantiles_confidence_interval_both='confidence_interval'

    convert_proportions_to_percentages='yes'

    rate_number_of_significant_digits=3


    model_survey_number_confidence_interval_of_subsample_replicates_metrics=convert_quantile_columns_to_confidence_interval_column(model_survey_quantiles_of_subsample_replicates_metrics=model_survey_quantiles_of_subsample_replicates_metrics, feature_column_name_filter_value_list_dictionary_key=feature_column_name_filter_value_list_dictionary_key)


    
    return model_survey_number_confidence_interval_of_subsample_replicates_metrics, df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics













###############################################################################################################################
#Calculate Overall and Coupon Venue Type Ad Revenue, Ad Spend, ROAS, Profit, Spend, and ROI 95% Confidence Intervals from metric replicates and append to metric confidence interval table

def get_model_survey_coupon_recommendation_cost_estimated_and_sale_estimated_replicate_collection_venue_type(df, column_name_list=None, column_name_drop_list=None, number_of_replicates=10000):
    '''
    df: data frame with columns containing feature values and indexes model-survey description and metric
    column_name_list: column names to use in creating colum name metric replicates collection
    column_name_drop_list: column names not to use
    '''
    
    df_model_survey_coupon_recommendation_cost_estimated_sale_estimated_replicate_collection={}
    
    if column_name_list==None:
        column_name_list=df.columns.to_list()
    
    if column_name_drop_list==None:
        column_name_drop_list=['Overall']
    
    
    #df_model_survey_coupon_recommendation_cost_estimated_sale_estimated_collection_venue_type={column_name:[df.loc[:, column_name] for _ in range(number_of_replicates)] for column_name in column_name_list}
    df_model_survey_coupon_recommendation_cost_estimated_sale_estimated_collection_venue_type={column_name:[df.loc[:, column_name]]*number_of_replicates for column_name in column_name_list}


    
    for coupon_venue_type in column_name_list:

        #get replicates of venue type coupon recommendation cost estimated and sale estimated metrics from data frame table
        df_model_survey_coupon_recommendation_cost_estimated_sale_estimated_replicates_coupon_venue_type=pd.concat(df_model_survey_coupon_recommendation_cost_estimated_sale_estimated_collection_venue_type[coupon_venue_type], axis=1,).T.reset_index(drop=True).T
        
        #fix column names to string from integers
        df_model_survey_coupon_recommendation_cost_estimated_sale_estimated_replicates_coupon_venue_type.columns=[str(integer) for integer in range(number_of_replicates)]

        df_model_survey_coupon_recommendation_cost_estimated_sale_estimated_replicate_collection[coupon_venue_type]=df_model_survey_coupon_recommendation_cost_estimated_sale_estimated_replicates_coupon_venue_type
    
    return df_model_survey_coupon_recommendation_cost_estimated_sale_estimated_replicate_collection















def get_Ad_Revenue_Ad_Spend_ROAS_replicate_metrics_from_venue_type_replicate_metrics(df, Ad_Revenue_Ad_Spend_ROAS_list=[True, True]):
    coupon_recommendation_cost_model_survey_list=['Model', 'Model']
    
    if Ad_Revenue_Ad_Spend_ROAS_list[0]==True:
        #Model Total Revenue, Total Ad Spend Metrics
        df.loc[('Model', 'Ad Revenue'), :]=df.loc[('Model', 'Conversions'), :]*df.loc[('Model', 'Average Sale Estimated'), :]

        df.loc[('Model', 'Ad Spend'), :]=df.loc[(coupon_recommendation_cost_model_survey_list[0], 'Average Coupon Recommendation Cost Estimated'), :]*df.loc[('Model', 'Coupons Recommended'), :]

        
        #Survey Total Revenue, Total Ad Spend Metrics    
        df.loc[('Survey', 'Ad Revenue'), :]=df.loc[('Survey', 'Conversions'), :]*df.loc[('Survey', 'Average Sale Estimated'), :]

        df.loc[('Survey', 'Ad Spend'), :]=df.loc[(coupon_recommendation_cost_model_survey_list[1], 'Average Coupon Recommendation Cost Estimated'), :]*df.loc[('Survey', 'Coupons Recommended'), :]

        
        #Model-Survey Total Revenue, Total Ad Spend Metrics
        df.loc[('Model-Survey Difference', 'Ad Revenue'), :]=df.loc[('Model', 'Ad Revenue'), :]-df.loc[('Survey', 'Ad Revenue'), :]
        df.loc[('Model-Survey Difference', 'Ad Spend'), :]=df.loc[('Model', 'Ad Spend'), :]-df.loc[('Survey', 'Ad Spend'), :]
        
    if Ad_Revenue_Ad_Spend_ROAS_list[1]==True:
        df.loc[('Model', 'ROAS'), :]=df.loc[('Model', 'Ad Revenue'), :]/df.loc[('Model', 'Ad Spend'), :]*100


        df.loc[('Survey', 'ROAS'), :]=df.loc[('Survey', 'Ad Revenue'), :]/df.loc[('Survey', 'Ad Spend'), :]*100


        df.loc[('Model-Survey Difference', 'ROAS'), :]=df.loc[('Model', 'ROAS'), :]-df.loc[('Survey', 'ROAS'), :]

    
    return df














def get_Overall_ROAS_Profit_Spend_ROI_per_Survey_Model_Survey_Difference(df, ROAS_Profit_Spend_ROI_list=[True, True, True, True]):

    df.loc[('Model', 'ROAS'), :]=df.loc[('Model', 'Ad Revenue'), :]/df.loc[('Model', 'Ad Spend'), :]*100
    df.loc[('Survey', 'ROAS'), :]=df.loc[('Survey', 'Ad Revenue'), :]/df.loc[('Survey', 'Ad Spend'), :]*100
    
    df.loc[('Model-Survey Difference', 'ROAS'), :]=df.loc[('Model', 'ROAS'), :]-df.loc[('Survey', 'ROAS'), :]

    
    def get_Profit_Spend_ROI_with_additional_spend(df, additional_spend=None):

        if additional_spend==None:
            additional_spend=200


        #Model ROI Metrics
        model_campaign_spend=df.loc[('Model', 'Ad Spend'), :]+additional_spend
        model_campaign_profit=df.loc[('Model', 'Ad Revenue'), :]-model_campaign_spend

        df.loc[('Model', 'Profit '+str(additional_spend)), :]=model_campaign_profit
        df.loc[('Model', 'Spend '+str(additional_spend)), :]=model_campaign_spend

        df.loc[('Model', 'ROI '+str(additional_spend)), :]=model_campaign_profit/model_campaign_spend*100


        #Survey ROI Metrics
        survey_campaign_spend=df.loc[('Survey', 'Ad Spend'), :]+additional_spend
        survey_campaign_profit=df.loc[('Survey', 'Ad Revenue'), :]-survey_campaign_spend

        df.loc[('Survey', 'Profit '+str(additional_spend)), :]=survey_campaign_profit
        df.loc[('Survey', 'Spend '+str(additional_spend)), :]=survey_campaign_spend

        df.loc[('Survey', 'ROI '+str(additional_spend)), :]=survey_campaign_profit/survey_campaign_spend*100

        #Survey-Model Difference ROI Metrics

        df.loc[('Model-Survey Difference', 'Profit '+str(additional_spend)), :]=df.loc[('Model', 'Profit '+str(additional_spend)), :]-\
                                                                                      df.loc[('Survey', 'Profit '+str(additional_spend)), :]

        df.loc[('Model-Survey Difference', 'Spend '+str(additional_spend)), :]=df.loc[('Model', 'Spend '+str(additional_spend)), :]-\
                                                                                     df.loc[('Survey', 'Spend '+str(additional_spend)), :]

        df.loc[('Model-Survey Difference', 'ROI '+str(additional_spend)), :]=df.loc[('Model', 'ROI '+str(additional_spend)), :]-\
                                                                             df.loc[('Survey', 'ROI '+str(additional_spend)), :]
        
        return df
    
    df=get_Profit_Spend_ROI_with_additional_spend(df, additional_spend=200)
    df=get_Profit_Spend_ROI_with_additional_spend(df, additional_spend=2000)
    df=get_Profit_Spend_ROI_with_additional_spend(df, additional_spend=20000)
    

    return df




def calculate_Overall_and_Coupon_Venue_Type_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_95_Confidence_Intervals_from_metric_replicates_and_append_to_metric_confidence_interval_table(
    df_model_name_model_survey_coupon_recommendation_cost_estimated_sale_estimated_replicate_collection,
    df_test_model_name_model_survey_95_confidence_interval_metric_feature_column_name_filter_value,
    test_model_name_metric_replicate_filename_collection,
    model_type,
    filename_version,
    number_of_replicates=1000,):
    
    #get Ad Revenue, Ad Spend, and ROAS (for Model, Survey, and Model-Survey Difference) by reading in the five (5) Coupon Venue Type Metric Replicates files
    column_name_list=['Coffee House', 'Bar', 'Takeout', 'Low-Cost Restaurant', 'Mid-Range Restaurant']
    df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type={}

    for column_name in column_name_list:
        #read in random forest coupon venue type metric replicates
        df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name]=\
        rcp(test_model_name_metric_replicate_filename_collection[column_name], index_col=[0,1])

        #add (Random Forest or Gradient Boosting) Model and Survey Coupon Recommendation Cost Estimated and Sale Estimated Replicates
        df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name]=pd.concat([df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name], 
                                                                                                                                   df_model_name_model_survey_coupon_recommendation_cost_estimated_sale_estimated_replicate_collection[column_name]], axis=0)

        #get and add Ad Spend, Ad Revenue, ROAS per coupon venue type
        df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name]=\
        get_Ad_Revenue_Ad_Spend_ROAS_replicate_metrics_from_venue_type_replicate_metrics(df=df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name], 
                                                                                         Ad_Revenue_Ad_Spend_ROAS_list=[True, True])





    #get 95% confidence interval quantile collection per Coupon Venue Type from Ad Revenue, Ad Spend, and ROAS replicate metrics
    column_name_list=['Coffee House', 'Bar', 'Takeout', 'Low-Cost Restaurant', 'Mid-Range Restaurant']
    tuple_index_name_list=[('Model', 'Average Coupon Recommendation Cost Estimated'), ('Model', 'Average Sale Estimated'), ('Survey', 'Average Coupon Recommendation Cost Estimated'), ('Survey', 'Average Sale Estimated'),
     ('Model', 'Ad Revenue'), ('Model', 'Ad Spend'),
     ('Survey', 'Ad Revenue'), ('Survey', 'Ad Spend'),
     ('Model-Survey Difference', 'Ad Revenue'), ('Model-Survey Difference', 'Ad Spend'), 
     ('Model', 'ROAS'), ('Survey', 'ROAS'), ('Model-Survey Difference', 'ROAS'),]
    df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection={}

    for column_name in column_name_list:
        df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection[column_name]=\
        get_metric_quatiles_from_number_subsample_replicates_metrics(df=df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name],
                                                                         quantile_lower_upper_list=[.025, .975]).loc[tuple_index_name_list,:]






    #get and add confidence interval column from two quantile columns
    multiple_index_tuple_list_usd=[('Model', 'Average Coupon Recommendation Cost Estimated'), ('Model', 'Average Sale Estimated'), ('Survey', 'Average Coupon Recommendation Cost Estimated'), ('Survey', 'Average Sale Estimated'), ('Model', 'Ad Revenue'), ('Model', 'Ad Spend'), ('Survey', 'Ad Revenue'), ('Survey', 'Ad Spend'), ('Model-Survey Difference', 'Ad Revenue'), ('Model-Survey Difference', 'Ad Spend'), ('Model', 'Profit 200'), ('Model', 'Spend 200'), ('Survey', 'Profit 200'), ('Survey', 'Spend 200'), ('Model-Survey Difference', 'Profit 200'), ('Model-Survey Difference', 'Spend 200'), ('Model', 'Profit 2000'), ('Model', 'Spend 2000'), ('Survey', 'Profit 2000'), ('Survey', 'Spend 2000'), ('Model-Survey Difference', 'Profit 2000'), ('Model-Survey Difference', 'Spend 2000'), ('Model', 'Profit 20000'), ('Model', 'Spend 20000'), ('Survey', 'Profit 20000'), ('Survey', 'Spend 20000'), ('Model-Survey Difference', 'Profit 20000'), ('Model-Survey Difference', 'Spend 20000'),]
    multiple_index_tuple_list_percent=[('Model', 'ROAS'), ('Survey', 'ROAS'), ('Model-Survey Difference', 'ROAS'), ('Model', 'ROI 200'), ('Survey', 'ROI 200'), ('Model-Survey Difference', 'ROI 200'), ('Model', 'ROI 2000'),('Survey', 'ROI 2000'), ('Model-Survey Difference', 'ROI 2000'),('Model', 'ROI 20000'), ('Survey', 'ROI 20000'), ('Model-Survey Difference', 'ROI 20000')]

    column_name_list=df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection['Coffee House'].columns.to_list()
    multiple_index_tuple_list=df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection['Coffee House'].index.to_list()
    venue_type_list=['Coffee House', 'Bar', 'Takeout', 'Low-Cost Restaurant', 'Mid-Range Restaurant']


    for venue_type in venue_type_list:

        for multiple_index_tuple in multiple_index_tuple_list:

            if multiple_index_tuple in multiple_index_tuple_list_usd:

                df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection[venue_type].loc[multiple_index_tuple, venue_type]=\
                '(\$'+str(round(df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection[venue_type].loc[multiple_index_tuple,column_name_list[0]], 2))+\
                ', \$'+str(round(df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection[venue_type].loc[multiple_index_tuple,column_name_list[1]], 2))+\
                ')'

            elif multiple_index_tuple in multiple_index_tuple_list_percent:
                df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection[venue_type].loc[multiple_index_tuple, venue_type]=\
                '('+str(round(df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection[venue_type].loc[multiple_index_tuple,column_name_list[0]], 2))+\
                '%, '+str(round(df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection[venue_type].loc[multiple_index_tuple,column_name_list[1]], 2))+\
                '%)'










    #get Ad Revenue, Ad Spend, ROAS 95% Confidence Interval per Coupon Venue Type DataFrame from 95% Confidence Interval and Quantile Collection (by Coupon Venue TYpe)
    venue_type_list=['Coffee House', 'Bar', 'Takeout', 'Low-Cost Restaurant', 'Mid-Range Restaurant']

    data_frame_list=[df_Ad_Revenue_Ad_Spend_ROAS_per_model_survey_model_survey_difference_quantile_collection[venue_type].loc[:, [venue_type]] for venue_type in venue_type_list]
    df_Ad_Revenue_Ad_Spend_ROAS_coupon_venue_type_confidence_interval=pd.concat(data_frame_list, axis=1)
    del data_frame_list













    #get Overall Ad Revenue, Ad Spend, ROAS, Profit, Spend, and ROI Metric Replicates from Coupon Venue Type Ad Revenue and Ad Spend Replicates Collection


    #Get Overall Ad Revenue and Ad Spend per Model, Survey, and Model-Survey Difference
    #Initialize variables
    column_name_list=['Coffee House', 'Bar', 'Takeout', 'Low-Cost Restaurant', 'Mid-Range Restaurant',]
    tuple_index_name_list=[('Model', 'Ad Revenue'), ('Model', 'Ad Spend'), ('Survey', 'Ad Revenue'), ('Survey', 'Ad Spend'), ('Model-Survey Difference', 'Ad Revenue'), ('Model-Survey Difference', 'Ad Spend'),]
    #Calculate Overall Ad Revenue and Ad Spend per Model, Survey, and Model-Survey Difference (via a sum up of Ad Revenue and Ad Spend per Coupon Venue Type)
    df_test_model_name_number_metric_estimated_10000_metric_replicates_overall=\
    df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name_list[0]].loc[tuple_index_name_list,:]+\
    df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name_list[1]].loc[tuple_index_name_list,:]+\
    df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name_list[2]].loc[tuple_index_name_list,:]+\
    df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name_list[3]].loc[tuple_index_name_list,:]+\
    df_test_model_name_number_metric_estimated_10000_metric_replicates_collection_coupon_venue_type[column_name_list[4]].loc[tuple_index_name_list,:]


    #Calculate and add Overall ROAS from Ad Revenue and Ad Spend by Model, Survey, and Model-Survey Difference (via Ad Revenue / Ad Spend)
    df_test_model_name_number_metric_estimated_10000_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall=\
    get_Overall_ROAS_Profit_Spend_ROI_per_Survey_Model_Survey_Difference(df_test_model_name_number_metric_estimated_10000_metric_replicates_overall)



    #save df_test_model_name_number_metric_estimated_10000_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall as DataFrame table with model type description????
    

    filename='df_test_'+str(model_type)+'_number_metric_estimated_10000_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall_v'+str(filename_version)+'.pkl'
    _=\
    save_and_return_data_frame_v2(df_test_model_name_number_metric_estimated_10000_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall, filename=filename)








    #get 95% Confidence Interval Quantile columns of Overall Ad Revenue, Ad Spend, ROAS, Profit, Spend, and ROI replicates
    df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles=\
    get_metric_quatiles_from_number_subsample_replicates_metrics(df=df_test_model_name_number_metric_estimated_10000_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall,
                                                                     quantile_lower_upper_list=[.025, .975])










    #convert to 95% Confidence Interval column from two Quantile columns


    #get multiple index tuple list
    multiple_index_tuple_list=df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles.index.to_list()

    multiple_index_tuple_list_usd=[('Model', 'Average Coupon Recommendation Cost Estimated'), ('Model', 'Average Sale Estimated'), ('Survey', 'Average Coupon Recommendation Cost Estimated'), ('Survey', 'Average Sale Estimated'), ('Model', 'Ad Revenue'), ('Model', 'Ad Spend'), ('Survey', 'Ad Revenue'), ('Survey', 'Ad Spend'), ('Model-Survey Difference', 'Ad Revenue'), ('Model-Survey Difference', 'Ad Spend'), ('Model', 'Profit 200'), ('Model', 'Spend 200'), ('Survey', 'Profit 200'), ('Survey', 'Spend 200'), ('Model-Survey Difference', 'Profit 200'), ('Model-Survey Difference', 'Spend 200'), ('Model', 'Profit 2000'), ('Model', 'Spend 2000'), ('Survey', 'Profit 2000'), ('Survey', 'Spend 2000'), ('Model-Survey Difference', 'Profit 2000'), ('Model-Survey Difference', 'Spend 2000'), ('Model', 'Profit 20000'), ('Model', 'Spend 20000'), ('Survey', 'Profit 20000'), ('Survey', 'Spend 20000'), ('Model-Survey Difference', 'Profit 20000'), ('Model-Survey Difference', 'Spend 20000'),]

    multiple_index_tuple_list_percent=[('Model', 'ROAS'), ('Survey', 'ROAS'), ('Model-Survey Difference', 'ROAS'), ('Model', 'ROI 200'), ('Survey', 'ROI 200'), ('Model-Survey Difference', 'ROI 200'), ('Model', 'ROI 2000'),('Survey', 'ROI 2000'), ('Model-Survey Difference', 'ROI 2000'),('Model', 'ROI 20000'), ('Survey', 'ROI 20000'), ('Model-Survey Difference', 'ROI 20000')]

    #get lower and upper quantile column names
    column_name_list=df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles.columns.to_list()


    #combine two columns into one based on multiple index name

    for multiple_index_tuple in multiple_index_tuple_list:

        if multiple_index_tuple in multiple_index_tuple_list_usd:
            df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles.loc[multiple_index_tuple, 'Overall']=\
            '(\$'+str(round(df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles.loc[multiple_index_tuple, column_name_list[0]], 2))+', \$'+\
            str(round(df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles.loc[multiple_index_tuple, column_name_list[1]], 2))+')'

        elif multiple_index_tuple in multiple_index_tuple_list_percent:
            df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles.loc[multiple_index_tuple, 'Overall']=\
            '('+str(round(df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles.loc[multiple_index_tuple, column_name_list[0]], 2))+'%, '+\
            str(round(df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles.loc[multiple_index_tuple, column_name_list[1]],2))+'%)'

    df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_confidence_interval_and_quantiles=df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles
    del df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_quantiles








    #combine Overall and Coupon Venue Type Ad Revenue, Ad Spend, ROAS, Profit, Spend, ROI 95% Confidence Interval metrics
    df_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_confidence_interval_overall_and_coupon_venue_type=\
    pd.concat([df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_confidence_interval_and_quantiles.loc[:, ['Overall']], df_Ad_Revenue_Ad_Spend_ROAS_coupon_venue_type_confidence_interval], axis=1)

    del df_Ad_Revenue_Ad_Spend_ROAS_coupon_venue_type_confidence_interval, df_Overall_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_confidence_interval_and_quantiles
    df_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_confidence_interval_overall_and_coupon_venue_type








    #add '95% Confidence Interval' as column index to Overall and Coupon Venue Type columns
    overall_and_coupon_venue_type_list=['Overall']+venue_type_list
    column_name_tuple_list=[('95% Confidence Interval', coupon_venue_type) for coupon_venue_type in overall_and_coupon_venue_type_list]
    multiple_index_overall_and_coupon_venue_type=pd.MultiIndex.from_tuples(column_name_tuple_list)

    df_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_confidence_interval_overall_and_coupon_venue_type.columns=multiple_index_overall_and_coupon_venue_type









    #get all metrics for Overall and Coupon Venue Type 95% Confidence Interval Table by adding Confidence Interval Metrics and Ad Revenue, Ad Spend, ROAS, Profit, Spend, ROI 95% Confidence Intervals
    df_model_name_95_confidence_interval_metrics_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI=\
    pd.concat([df_test_model_name_model_survey_95_confidence_interval_metric_feature_column_name_filter_value, 
               df_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_model_survey_model_survey_difference_metric_confidence_interval_overall_and_coupon_venue_type], axis=0)

    return df_model_name_95_confidence_interval_metrics_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI


###########################################################################################################################















def convert_collection_to_data_frame_and_drop_top_column_level(df_collection):
    #convert to data frame from collection
    df=pd.concat(df_collection, axis=1)

    #drop column name top level (from collection key)
    df.columns=df.columns.droplevel(level=0)
    
    return df


















############################################################
#############################################################

def get_model_predictions_decision_threshold_metric_aim_coupon_venue_type(model_name, metric_name, metric_quantity, coupon_name, coupon_name_short, Y_test_model_prediction_data_frame_collection, df_y_test_model_name_prediction_probability_y_actual_coupon_venue_type, decision_threshold_collection):
    Y_test_model_prediction_list_collection = {}
    
    key = model_name + '_' + metric_name + '_' + metric_quantity + '_' + coupon_name_short + '_coupon'
    column_name = model_name + '_prediction_' + metric_name + '_' + metric_quantity + '_' + coupon_name_short + '_coupon'
    model_name_metric_name = model_name + '_' + metric_name

    Y_test_model_prediction_list_collection[key] = \
    [1 if prediction_probability > decision_threshold_collection[coupon_name][model_name_metric_name] else 0 \
     for prediction_probability in df_y_test_model_name_prediction_probability_y_actual_coupon_venue_type.loc[:, 'Y_test_' + model_name + '_prediction_probability'].to_list()]

    Y_test_model_prediction_data_frame_collection[key] = \
    pd.DataFrame(Y_test_model_prediction_list_collection[key], columns=[column_name])
    
    return Y_test_model_prediction_data_frame_collection, key






####################################################
#Survey Analysis
################################################################

# def plot_bar_graph(df, 
#                    x='coupon_venue_type', 
#                    bar_category_list=['Refused Coupon', 'Accepted Coupon'],
#                    title='Coupon Venue Count and Percentage per Acceptance or Refusal', 
#                    color=['#8c6bb1', '#41ab5d'], 
#                    figsize=(12, 10),
#                    figure_filename=None,
#                    dpi=100):

#     #plt.figure(figsize=(10, 10))

#     #sns.set(style="darkgrid")

#     figure_filename_exists = os.path.isfile(figure_filename)
#     if figure_filename_exists == True:
#         img = mpimg.imread(figure_filename)
#         plt.figure(figsize=(20, 16))
#         plt.grid(False)
#         plt.axis('off')
#         plt.imshow(img)
#     else:
#         df.plot(x=x, kind ='bar', stacked=True, title=title, mark_right=True, color=color, figsize=(12, 10))

#         df_row_sum = df.loc[:, bar_category_list[0]] + df.loc[:, bar_category_list[1]]

#         df_stacked_bar_percentage = df.loc[:, df.columns[1:]].div(df_row_sum, 0) * 100

#         for column_name in df_stacked_bar_percentage:

#             for i, (cs, ab, pc) in enumerate(zip(df.iloc[:, 1:].cumsum(1).loc[:, column_name], df.loc[:, column_name], df_stacked_bar_percentage.loc[:, column_name])):
#                 plt.text(i, cs-ab/2, str(np.round(pc, 1)) + '%', verticalalignment='center', horizontalalignment='center', rotation=0, fontsize=14)
#         plt.ylabel('count')

#         #save it
#         plt.savefig(figure_filename, bbox_inches='tight', dpi=dpi)

#         plt.show()




def donut_plot(name_list, size_list, title, title_fontsize, figure_filename, dpi, color_list, circle_color='white'):
    figure_filename_exists = os.path.isfile(figure_filename)
    if figure_filename_exists == True:
        img = mpimg.imread(figure_filename)
        plt.figure(figsize=(10, 8))
        plt.grid(False)
        plt.axis('off')
        plt.imshow(img)
    else:
        white_circle = plt.Circle((0,0), 0.7, color=circle_color)
        plt.title(title, fontsize=title_fontsize)
        plt.pie(size_list, labels=name_list, colors=color_list)
        p = plt.gcf()
        p.gca().add_artist(white_circle)
        
        #save it
        plt.savefig(figure_filename, bbox_inches='tight', dpi=dpi)

        plt.show()

        
        
        
        
        
        
        
#######################################################################################################################
# Model Metrics By Feature Pair

def get_model_model_feature_and_model_feature_two_tuple_metrics(df, column_name_y_predicted, feature_column_name_two_tuple_list):
    
    metric_value_two_dimensional_list=[]
    
    def get_model_metric_value_list(df, column_name_y_predicted):
        
        model_metric_value_list=[]
        
        y_true=df.loc[:, 'Y']
        y_predicted=df.loc[:, column_name_y_predicted]
        
        #precision
        model_precision=precision_score(y_true=y_true, y_pred=y_predicted)
        
        #recall
        model_recall=recall_score(y_true=y_true, y_pred=y_predicted)
        
        true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_true=y_true, y_pred=y_predicted).ravel()
        
        #conversions
        model_conversions=true_positive
        
        #coupons recommended
        model_coupons_recommended=false_positive+true_positive
        
        model_metric_value_list+=[model_precision, model_recall, model_conversions, model_coupons_recommended,]
        return model_metric_value_list
        
    model_metric_value_list=get_model_metric_value_list(df=df, column_name_y_predicted=column_name_y_predicted)
    #print(model_metric_value_list)
    
    
    
    #get model feature two-tuple metrics
    
    ####
    for feature_column_name_two_tuple in feature_column_name_two_tuple_list:
        
        feature_column_name_0_value_list=list(df.loc[:, feature_column_name_two_tuple[0]].unique())
        feature_column_name_1_value_list=list(df.loc[:, feature_column_name_two_tuple[1]].unique())
        
        ####
        for feature_column_name_0_value in feature_column_name_0_value_list:
            
            def get_model_feature_and_value_metric_list(df, 
                                                        feature_column_name_two_tuple,
                                                        feature_column_name_two_tuple_index, 
                                                        feature_column_name_number_value, 
                                                        model_metric_value_list):
                
                feature_number_and_value_metric_value_list=[]

                y_predicted_feature_column_name_number_and_value_name='y_predicted_'+str(feature_column_name_two_tuple[feature_column_name_two_tuple_index])+'_'+str(feature_column_name_number_value)

                #get y_predicted by feature number and value
                df.loc[:, y_predicted_feature_column_name_number_and_value_name]=0
                df.loc[(df.loc[:, feature_column_name_two_tuple[feature_column_name_two_tuple_index]]==feature_column_name_number_value), y_predicted_feature_column_name_number_and_value_name]=\
                df.loc[(df.loc[:, feature_column_name_two_tuple[feature_column_name_two_tuple_index]]==feature_column_name_number_value), column_name_y_predicted]


                y_true=df.loc[:, 'Y']
                y_predicted=df.loc[:, y_predicted_feature_column_name_number_and_value_name]
                #get feature number and value metrics

                #precision
                model_feature_number_and_value_precision=precision_score(y_true=y_true, y_pred=y_predicted)

                #recall
                model_feature_number_and_value_recall=recall_score(y_true=y_true, y_pred=y_predicted)

                true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_true=y_true, y_pred=y_predicted).ravel()

                #conversions
                model_feature_number_and_value_conversions=true_positive

                #coupons recommended
                model_feature_number_and_value_coupons_recommended=false_positive+true_positive
                
                #get metric differences
                model_feature_number_and_value_precision_model_precision_difference=model_feature_number_and_value_precision-model_metric_value_list[0]
                
                model_feature_number_and_value_recall_model_recall_difference=model_feature_number_and_value_recall-model_metric_value_list[1]
                
                model_feature_number_and_value_conversions_model_conversions_difference=model_feature_number_and_value_conversions-model_metric_value_list[2]
                
                model_feature_number_and_value_coupons_recommended_model_coupons_recommended_difference=model_feature_number_and_value_coupons_recommended-model_metric_value_list[3]


                feature_number_and_value_metric_value_list=[feature_column_name_two_tuple[feature_column_name_two_tuple_index],
                                                            feature_column_name_number_value,
                                                            model_feature_number_and_value_precision, 
                                                            model_feature_number_and_value_recall,
                                                            model_feature_number_and_value_conversions,
                                                            model_feature_number_and_value_coupons_recommended,
                                                            model_feature_number_and_value_precision_model_precision_difference,
                                                            model_feature_number_and_value_recall_model_recall_difference,
                                                            model_feature_number_and_value_conversions_model_conversions_difference,
                                                            model_feature_number_and_value_coupons_recommended_model_coupons_recommended_difference,]
                return feature_number_and_value_metric_value_list
            
            
            model_feature_0_and_value_metric_value_list=get_model_feature_and_value_metric_list(df, 
                                                                                                feature_column_name_two_tuple=feature_column_name_two_tuple,
                                                                                                feature_column_name_two_tuple_index=0, 
                                                                                                feature_column_name_number_value=feature_column_name_0_value,
                                                                                                model_metric_value_list=model_metric_value_list)
            #print(model_feature_0_and_value_metric_value_list)
            
            
            
            ####
            for feature_column_name_1_value in feature_column_name_1_value_list:
                
                model_feature_1_and_value_metric_value_list=get_model_feature_and_value_metric_list(df, 
                                                                                                    feature_column_name_two_tuple,
                                                                                                    feature_column_name_two_tuple_index=1, 
                                                                                                    feature_column_name_number_value=feature_column_name_1_value,
                                                                                                    model_metric_value_list=model_metric_value_list)
                #print(model_feature_1_and_value_metric_value_list)

                
                
                model_feature_0_and_value_feature_1_and_value_metric_value_list=[]
                
                y_model_feature_column_name_0_and_value_feature_column_name_1_and_value_predicted=\
                'y_model_'+feature_column_name_two_tuple[0]+'_'+str(feature_column_name_0_value).replace(' ', '_')+'_'+feature_column_name_two_tuple[1]+'_'+str(feature_column_name_1_value).replace(' ', '_')+'_predicted'
                
                #print(y_model_feature_column_name_0_and_value_feature_column_name_1_and_value_predicted)
                
                
                
                df.loc[:, y_model_feature_column_name_0_and_value_feature_column_name_1_and_value_predicted]=0
                df.loc[(df.loc[:, feature_column_name_two_tuple[0]]==feature_column_name_0_value) &
                       (df.loc[:, feature_column_name_two_tuple[1]]==feature_column_name_1_value), y_model_feature_column_name_0_and_value_feature_column_name_1_and_value_predicted]=\
                df.loc[(df.loc[:, feature_column_name_two_tuple[0]]==feature_column_name_0_value) &
                       (df.loc[:, feature_column_name_two_tuple[1]]==feature_column_name_1_value), column_name_y_predicted]
                
                
                y_true=df.loc[:, 'Y']
                y_predicted=df.loc[:, y_model_feature_column_name_0_and_value_feature_column_name_1_and_value_predicted]
                
                
                #get feature 0 and value and feature 1 and value metric value list

                #get precision
                model_feature_0_and_value_feature_1_and_value_precision=precision_score(y_true=y_true, y_pred=y_predicted)
                
                #get recall
                model_feature_0_and_value_feature_1_and_value_recall=recall_score(y_true=y_true, y_pred=y_predicted)
                
                true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(y_true=y_true, y_pred=y_predicted).ravel()
                
                #get conversions
                model_feature_0_and_value_feature_1_and_value_conversions=true_positives
                
                #get coupons recommended
                model_feature_0_and_value_feature_1_and_value_coupons_recommended=true_positives+false_positives
                
                
                
                #metric differences: precision, recall, conversions, coupons recommended
                
                #precision difference
                model_feature_0_and_value_feature_1_and_value_precision_model_feature_0_and_value_precision_difference=\
                model_feature_0_and_value_feature_1_and_value_precision-model_feature_0_and_value_metric_value_list[2]
                
                model_feature_0_and_value_feature_1_and_value_precision_model_feature_1_and_value_precision_difference=\
                model_feature_0_and_value_feature_1_and_value_precision-model_feature_1_and_value_metric_value_list[2]
                
                #recall difference
                model_feature_0_and_value_feature_1_and_value_recall_model_feature_0_and_value_recall_difference=\
                model_feature_0_and_value_feature_1_and_value_recall-model_feature_0_and_value_metric_value_list[3]
                
                model_feature_0_and_value_feature_1_and_value_recall_model_feature_1_and_value_recall_difference=\
                model_feature_0_and_value_feature_1_and_value_recall-model_feature_1_and_value_metric_value_list[3]
                
                #conversions
                model_feature_0_and_value_feature_1_and_value_conversions_model_feature_0_and_value_conversions_difference=\
                model_feature_0_and_value_feature_1_and_value_conversions-model_feature_0_and_value_metric_value_list[4]
                
                model_feature_0_and_value_feature_1_and_value_conversions_model_feature_1_and_value_conversions_difference=\
                model_feature_0_and_value_feature_1_and_value_conversions-model_feature_1_and_value_metric_value_list[4]
                
                #coupons recommended
                model_feature_0_and_value_feature_1_and_value_coupons_recommended_model_feature_0_and_value_coupons_recommended_difference=\
                model_feature_0_and_value_feature_1_and_value_coupons_recommended-model_feature_0_and_value_metric_value_list[5]
                
                model_feature_0_and_value_feature_1_and_value_coupons_recommended_model_feature_1_and_value_coupons_recommended_difference=\
                model_feature_0_and_value_feature_1_and_value_coupons_recommended-model_feature_1_and_value_metric_value_list[5]
                
                

                model_feature_0_and_value_feature_1_and_value_metric_value_list+=\
                [model_feature_0_and_value_feature_1_and_value_precision, 
                 model_feature_0_and_value_feature_1_and_value_recall, 
                 model_feature_0_and_value_feature_1_and_value_conversions,
                 model_feature_0_and_value_feature_1_and_value_coupons_recommended,
                 model_feature_0_and_value_feature_1_and_value_precision_model_feature_0_and_value_precision_difference,
                 model_feature_0_and_value_feature_1_and_value_precision_model_feature_1_and_value_precision_difference,
                 model_feature_0_and_value_feature_1_and_value_recall_model_feature_0_and_value_recall_difference,
                 model_feature_0_and_value_feature_1_and_value_recall_model_feature_1_and_value_recall_difference,
                 model_feature_0_and_value_feature_1_and_value_conversions_model_feature_0_and_value_conversions_difference,
                 model_feature_0_and_value_feature_1_and_value_conversions_model_feature_1_and_value_conversions_difference,
                 model_feature_0_and_value_feature_1_and_value_coupons_recommended_model_feature_0_and_value_coupons_recommended_difference,
                 model_feature_0_and_value_feature_1_and_value_coupons_recommended_model_feature_1_and_value_coupons_recommended_difference,]
                
                metric_value_two_dimensional_list+=[model_metric_value_list+\
                                                    model_feature_0_and_value_metric_value_list+\
                                                    model_feature_1_and_value_metric_value_list+\
                                                    model_feature_0_and_value_feature_1_and_value_metric_value_list]
                
    
    #get feature number and value column names
    feature_number_and_value_column_name_list=['Feature '+str(index)+substring for index in [0,1] for substring in ['', ' Value']]

    #get feature number metric column names
    substring_list='Conversion Rate', 'Recall', 'Conversions', 'Coupons Recommended', 'Conversion Rate Model Conversion Rate Difference', 'Recall Model Recall Difference', 'Conversions Model Conversions Difference', 'Coupons Recommended Model Coupons Recommended Difference',
    model_feature_number_column_name_list=['Model Feature '+str(index)+' '+substring for index in [0,1] for substring in substring_list]

    #get feature 0 feature 1 metric column names
    base_metric_list=['Conversion Rate', 'Recall', 'Conversions', 'Coupons Recommended']
    feature_0_feature_1_metric_list_first=['Model Feature 0 Feature 1 '+metric for metric in base_metric_list]
    feature_0_feature_1_metric_list_second=['Model Feature 0 Feature 1 '+metric+' Feature '+str(feature_number)+' '+metric+' Difference' for metric in base_metric_list for feature_number in [0,1]]
    feature_0_feature_1_metric_list=feature_0_feature_1_metric_list_first+feature_0_feature_1_metric_list_second

        
    column_name_list=['Model Conversion Rate', 'Model Recall', 'Model Conversions', 'Model Coupons Recommended']+\
                      feature_number_and_value_column_name_list[0:2]+\
                      model_feature_number_column_name_list[0:8]+\
                      feature_number_and_value_column_name_list[2:4]+\
                      model_feature_number_column_name_list[8:16]+\
                      feature_0_feature_1_metric_list
                      
    
    
    df_metrics=pd.DataFrame(metric_value_two_dimensional_list, columns=column_name_list)
    
    return df_metrics






















################################################################################################################################
#Model Metrics by Feature Number and Value Star Number

def get_model_feature_and_value_star_zero_metric_value_list(df, column_name_y_predicted, model_feature_number_and_value_star_number_metric_value_list_collection):

    model_feature_number_and_value_star_number_metric_value_list_collection['model']=[None]*4

    y_true=df.loc[:, 'Y']
    y_predicted=df.loc[:, column_name_y_predicted]

    #precision
    model_feature_number_and_value_star_number_metric_value_list_collection['model'][0]=precision_score(y_true=y_true, y_pred=y_predicted)

    #recall
    model_feature_number_and_value_star_number_metric_value_list_collection['model'][1]=recall_score(y_true=y_true, y_pred=y_predicted)

    true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_true=y_true, y_pred=y_predicted).ravel()

    #conversions
    model_feature_number_and_value_star_number_metric_value_list_collection['model'][2]=true_positive

    #coupons recommended
    model_feature_number_and_value_star_number_metric_value_list_collection['model'][3]=false_positive+true_positive

    return model_feature_number_and_value_star_number_metric_value_list_collection['model']




def get_model_feature_and_value_star_one_metric_list(df, 
                                                     column_name_y_predicted,
                                                     feature_column_name_tuple_triple,
                                                     feature_column_name_tuple_triple_index, 
                                                     feature_column_name_number_value, 
                                                     model_feature_number_and_value_star_number_metric_value_list_collection):

    feature_number_string='feature_'+str(feature_column_name_tuple_triple_index)
    feature_number_and_model_difference_string='feature_'+str(feature_column_name_tuple_triple_index)+'_and_model_difference'
    
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string]=[None]*6
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_and_model_difference_string]=[None]*4
    
    
    #get feature number column name
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string][0]=\
    feature_column_name_tuple_triple[feature_column_name_tuple_triple_index]
    
    #get feature number value
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string][1]=\
    feature_column_name_number_value
    

    y_predicted_feature_column_name_number_and_value_name='y_predicted_'+str(feature_column_name_tuple_triple[feature_column_name_tuple_triple_index])+'_'+str(feature_column_name_number_value)

    #get y_predicted by feature number and value
    df.loc[:, y_predicted_feature_column_name_number_and_value_name]=0
    df.loc[(df.loc[:, feature_column_name_tuple_triple[feature_column_name_tuple_triple_index]]==feature_column_name_number_value), y_predicted_feature_column_name_number_and_value_name]=\
    df.loc[(df.loc[:, feature_column_name_tuple_triple[feature_column_name_tuple_triple_index]]==feature_column_name_number_value), column_name_y_predicted]


    y_true=df.loc[:, 'Y']
    y_predicted=df.loc[:, y_predicted_feature_column_name_number_and_value_name]
    #get feature number and value metrics

    #precision
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string][2]=precision_score(y_true=y_true, y_pred=y_predicted)

    #recall
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string][3]=recall_score(y_true=y_true, y_pred=y_predicted)

    true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_true=y_true, y_pred=y_predicted).ravel()

    #conversions
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string][4]=true_positive

    #coupons recommended
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string][5]=false_positive+true_positive

    
    
    
    #get Model Single Feature and Model No Filter Difference Metrics
    
    #precision
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_and_model_difference_string][0]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string][2]-\
    model_feature_number_and_value_star_number_metric_value_list_collection['model'][0]

    
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_and_model_difference_string][1]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string][3]-\
    model_feature_number_and_value_star_number_metric_value_list_collection['model'][1]

    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_and_model_difference_string][2]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string][4]-\
    model_feature_number_and_value_star_number_metric_value_list_collection['model'][2]

    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_and_model_difference_string][3]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string][5]-\
    model_feature_number_and_value_star_number_metric_value_list_collection['model'][3]
    
    
    return model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_string], \
           model_feature_number_and_value_star_number_metric_value_list_collection[feature_number_and_model_difference_string]










def get_model_feature_and_value_star_two_metric_lists_to_collection(df, 
                                                                    feature_column_name_tuple_triple, 
                                                                    feature_number_index_list,
                                                                    feature_number_single_value_collection,
                                                                    column_name_y_predicted,
                                                                    model_feature_number_and_value_star_number_metric_value_list_collection,):
    
    #get y_predicted column
    
    #get y_predicted column string
    y_model_feature_column_name_number_and_value_star_two_predicted=\
    'y_model_'+feature_column_name_tuple_triple[feature_number_index_list[0]]+'_'+\
    str(feature_number_single_value_collection['feature_'+str(feature_number_index_list[0])]).replace(' ', '_')+'_'+\
    feature_column_name_tuple_triple[feature_number_index_list[1]]+'_'+\
    str(feature_number_single_value_collection['feature_'+str(feature_number_index_list[1])]).replace(' ', '_')+'_predicted'
    
    
    #get feature number key index
    key_value_list=[None]*5
    
    key_value_list[3]='feature_'+str(feature_number_index_list[0])
    key_value_list[4]='feature_'+str(feature_number_index_list[1])
    

    
    #get predictions column by feature and value, feature and value filter. And 0 for exclusion by filter
    df.loc[:, y_model_feature_column_name_number_and_value_star_two_predicted]=0
    df.loc[(df.loc[:, feature_column_name_tuple_triple[feature_number_index_list[0]]]==feature_number_single_value_collection[key_value_list[3]]) &
           (df.loc[:, feature_column_name_tuple_triple[feature_number_index_list[1]]]==feature_number_single_value_collection[key_value_list[4]]), y_model_feature_column_name_number_and_value_star_two_predicted]=\
    df.loc[(df.loc[:, feature_column_name_tuple_triple[feature_number_index_list[0]]]==feature_number_single_value_collection[key_value_list[3]]) &
           (df.loc[:, feature_column_name_tuple_triple[feature_number_index_list[1]]]==feature_number_single_value_collection[key_value_list[4]]), column_name_y_predicted]


    y_true=df.loc[:, 'Y']
    y_predicted=df.loc[:, y_model_feature_column_name_number_and_value_star_two_predicted]


    #calculate feature 0 and value and feature 1 and value metric value list: precision, recall, conversions, coupons recommended

    #get key value list for feature number and value star two metric calculations
    
    key_value_list[0]='feature_'+str(feature_number_index_list[0])+'_feature_'+str(feature_number_index_list[1])
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]]=[None]*4
    
    key_value_list[1]='feature_'+str(feature_number_index_list[0])+'_feature_'+str(feature_number_index_list[1])+'_and_feature_'+str(feature_number_index_list[0])+'_difference'
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[1]]=[None]*4

    key_value_list[2]='feature_'+str(feature_number_index_list[0])+'_feature_'+str(feature_number_index_list[1])+'_and_feature_'+str(feature_number_index_list[1])+'_difference'
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[2]]=[None]*4
    
    #get precision
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][0]=precision_score(y_true=y_true, y_pred=y_predicted)

    #get recall
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][1]=recall_score(y_true=y_true, y_pred=y_predicted)

    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(y_true=y_true, y_pred=y_predicted).ravel()

    #get conversions
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][2]=true_positives

    #get coupons recommended
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][3]=true_positives+false_positives

    
    
    
    
    
    #calculate metrics, e.g. feature 0 and value, feature 1 and value, feature 0 and value difference; 
    #                        feature 0 and value, feature 1 and value, feature 1 and value difference
    #metrics: precision, recall, conversions, coupons recommended

    
    #precision difference
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[1]][0]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][0]-model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[3]][2]

    #recall difference
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[1]][1]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][1]-model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[3]][3]

    #conversions
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[1]][2]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][2]-model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[3]][4]

    #coupons recommended
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[1]][3]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][3]-model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[3]][5]

    
    

    
    #precision difference
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[2]][0]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][0]-model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[4]][2]


    #recall difference
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[2]][1]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][1]-model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[4]][3]

    #conversions
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[2]][2]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][2]-model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[4]][4]

    #coupons recommended
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[2]][3]=\
    model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]][3]-model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[4]][5]

    return model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[0]],\
           model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[1]],\
           model_feature_number_and_value_star_number_metric_value_list_collection[key_value_list[2]]





def get_feature_number_and_value_star_number_and_feature_number_and_value_star_number_difference_metric_value_list_collection(df, key_list, model_feature_number_and_value_star_number_metric_value_list_collection):

    
    
    feature_number_star_one_list=['feature_0','feature_1','feature_2']
    
    #precision difference, recall difference, conversions, coupons recommended
    if not key_list[1] in feature_number_star_one_list:
        for index in range(4):
            model_feature_number_and_value_star_number_metric_value_list_collection[key_list[2]][index]=\
            model_feature_number_and_value_star_number_metric_value_list_collection[key_list[0]][index]-\
            model_feature_number_and_value_star_number_metric_value_list_collection[key_list[1]][index]

    elif key_list[1] in feature_number_star_one_list:
        for index in range(4):
            model_feature_number_and_value_star_number_metric_value_list_collection[key_list[2]][index]=\
            model_feature_number_and_value_star_number_metric_value_list_collection[key_list[0]][index]-\
            model_feature_number_and_value_star_number_metric_value_list_collection[key_list[1]][index+2]

    return model_feature_number_and_value_star_number_metric_value_list_collection[key_list[2]]




def get_model_feature_and_value_star_three_metric_lists_to_collection(df, 
                                                                      feature_column_name_tuple_triple,
                                                                      feature_number_single_value_collection,
                                                                      column_name_y_predicted, 
                                                                      model_feature_number_and_value_star_number_metric_value_list_collection,):

    
    #get y_predicted column
    
    #get y_predicted column string
    y_model_feature_column_name_number_and_value_star_three_predicted=\
    'y_model_'+feature_column_name_tuple_triple[0]+'_'+\
               str(feature_number_single_value_collection['feature_0']).replace(' ', '_')+'_'+\
               feature_column_name_tuple_triple[1]+'_'+\
               str(feature_number_single_value_collection['feature_1']).replace(' ', '_')+'_'+\
               feature_column_name_tuple_triple[2]+'_'+\
               str(feature_number_single_value_collection['feature_2']).replace(' ', '_')+'_predicted'
    

    #print(y_model_feature_column_name_number_and_value_star_two_predicted)

    
    #get feature number key index

    
    #get predictions column by feature and value, feature and value filter. And 0 for exclusion by filter
    df.loc[:, y_model_feature_column_name_number_and_value_star_three_predicted]=0
    df.loc[(df.loc[:, feature_column_name_tuple_triple[0]]==feature_number_single_value_collection['feature_0']) & \
           (df.loc[:, feature_column_name_tuple_triple[1]]==feature_number_single_value_collection['feature_1']) & \
           (df.loc[:, feature_column_name_tuple_triple[2]]==feature_number_single_value_collection['feature_2']), y_model_feature_column_name_number_and_value_star_three_predicted]=\
    df.loc[(df.loc[:, feature_column_name_tuple_triple[0]]==feature_number_single_value_collection['feature_0']) & \
           (df.loc[:, feature_column_name_tuple_triple[1]]==feature_number_single_value_collection['feature_1']) & \
           (df.loc[:, feature_column_name_tuple_triple[2]]==feature_number_single_value_collection['feature_2']), column_name_y_predicted]


    y_true=df.loc[:, 'Y']
    y_predicted=df.loc[:, y_model_feature_column_name_number_and_value_star_three_predicted]


    #calculate feature 0 and value and feature 1 and value metric value list: precision, recall, conversions, coupons recommended

    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2']=[None]*4
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_feature_1_difference']=[None]*4
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_feature_2_difference']=[None]*4
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_1_feature_2_difference']=[None]*4
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_difference']=[None]*4
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_1_difference']=[None]*4
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_2_difference']=[None]*4
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_model_difference']=[None]*4
    
    

    
    
    
    #get precision
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'][0]=precision_score(y_true=y_true, y_pred=y_predicted)

    #get recall
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'][1]=recall_score(y_true=y_true, y_pred=y_predicted)

    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(y_true=y_true, y_pred=y_predicted).ravel()

    #get conversions
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'][2]=true_positives

    #get coupons recommended
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'][3]=true_positives+false_positives



    
    #calculate difference metrics: precision, recall, conversions, coupons recommended
    feature_number_star_number_list=['feature_0_feature_1', 'feature_0_feature_2', 'feature_1_feature_2', 'feature_0', 'feature_1', 'feature_2', 'model']
    key_two_dimensional_list=[['feature_0_feature_1_feature_2', feature_number_star_number] for feature_number_star_number in feature_number_star_number_list]
    
    
    
    for key_list in key_two_dimensional_list:
        
        #add output key to key list
        key_list+=[key_list[0]+'_and_'+key_list[1]+'_difference']
        
        model_feature_number_and_value_star_number_metric_value_list_collection[key_list[2]]=[None]*4
        
        model_feature_number_and_value_star_number_metric_value_list_collection[key_list[2]]=\
        get_feature_number_and_value_star_number_and_feature_number_and_value_star_number_difference_metric_value_list_collection(df=df,
                                                                                                                                  key_list=key_list,
                                                                                                                                  model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection)

        
        
    return model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'],\
           model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_feature_1_difference'],\
           model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_feature_2_difference'],\
           model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_1_feature_2_difference'],\
           model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_difference'],\
           model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_1_difference'],\
           model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_2_difference'],\
           model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_model_difference']






def get_model_feature_tuple_zero_model_feature_tuple_single_model_feature_two_tuple_and_model_feature_tuple_triple_metrics(df, column_name_y_predicted, feature_column_name_tuple_triple_list):
    
    metric_value_two_dimensional_list=[]
        
    model_feature_number_and_value_star_number_metric_value_list_collection={}
    
    #<><>get Model metric value list
    model_feature_number_and_value_star_number_metric_value_list_collection['model']=\
    get_model_feature_and_value_star_zero_metric_value_list(df=df, column_name_y_predicted=column_name_y_predicted, model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection)


    #get model feature two-tuple metrics
    
    ####
    for feature_column_name_tuple_triple in feature_column_name_tuple_triple_list:
        
        feature_column_name_number_value_list_collection={}
        
        feature_column_name_number_value_list_collection['feature_0']=list(df.loc[:, feature_column_name_tuple_triple[0]].unique())
        feature_column_name_number_value_list_collection['feature_1']=list(df.loc[:, feature_column_name_tuple_triple[1]].unique())
        feature_column_name_number_value_list_collection['feature_2']=list(df.loc[:, feature_column_name_tuple_triple[2]].unique())
        
        ####

        feature_number_single_value_collection={}
        
        for feature_number_single_value_collection['feature_0'] in feature_column_name_number_value_list_collection['feature_0']:
            
            #<><>
            model_feature_number_and_value_star_number_metric_value_list_collection['feature_0'],\
            model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_and_model_difference']=\
            get_model_feature_and_value_star_one_metric_list(df=df,\
                                                             column_name_y_predicted=column_name_y_predicted,\
                                                             feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                             feature_column_name_tuple_triple_index=0,\
                                                             feature_column_name_number_value=feature_number_single_value_collection['feature_0'],\
                                                             model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection)
            
            
            ####
            
            for feature_number_single_value_collection['feature_1'] in feature_column_name_number_value_list_collection['feature_1']:
                
                #<><>
                model_feature_number_and_value_star_number_metric_value_list_collection['feature_1'],\
                model_feature_number_and_value_star_number_metric_value_list_collection['feature_1_and_model_difference']=\
                get_model_feature_and_value_star_one_metric_list(df=df,\
                                                                 column_name_y_predicted=column_name_y_predicted,\
                                                                 feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                                 feature_column_name_tuple_triple_index=1,\
                                                                 feature_column_name_number_value=feature_number_single_value_collection['feature_1'],\
                                                                 model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection)
                
                
#                 #print(model_feature_number_and_value_star_number_metric_value_list_collection['feature_1'])


                  
                
                #<><>get Model Feature number and value star two metrics: feature 0, feature 1; 
                #                                                         feature 0, feature 1 and feature 0 difference;
                #                                                         feature 0, feature 1 and feature 1 difference
                model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1'],\
                model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_and_feature_0_difference'],\
                model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_and_feature_1_difference']=\
                get_model_feature_and_value_star_two_metric_lists_to_collection(df=df,\
                                                                                feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                                                feature_number_index_list=[0,1],\
                                                                                feature_number_single_value_collection=feature_number_single_value_collection,\
                                                                                column_name_y_predicted=column_name_y_predicted,\
                                                                                model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection)

                


    


                
                ####
                
                for feature_number_single_value_collection['feature_2'] in feature_column_name_number_value_list_collection['feature_2']:
                    
                    
                    #<><>get feature 2 and value metrics
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_2'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_2_and_model_difference']=\
                    get_model_feature_and_value_star_one_metric_list(df=df,\
                                                                     column_name_y_predicted=column_name_y_predicted,\
                                                                     feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                                     feature_column_name_tuple_triple_index=2,\
                                                                     feature_column_name_number_value=feature_number_single_value_collection['feature_2'],\
                                                                     model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection,)


                    #get feature 0 and value, feature 2 and value metrics
                    #<><>get Model Feature number and value star two metrics
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_2'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_2_and_feature_0_difference'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_2_and_feature_2_difference']=\
                    get_model_feature_and_value_star_two_metric_lists_to_collection(df=df,\
                                                                                    feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                                                    feature_number_index_list=[0,2],\
                                                                                    feature_number_single_value_collection=feature_number_single_value_collection,\
                                                                                    column_name_y_predicted=column_name_y_predicted,\
                                                                                    model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection,)
                                                  

                    #get feature 1 and value, feature 2 and value metrics                                                             
                    #<><>get Model Feature number and value star two metrics
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_1_feature_2'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_1_feature_2_and_feature_1_difference'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_1_feature_2_and_feature_2_difference']=\
                    get_model_feature_and_value_star_two_metric_lists_to_collection(df=df, \
                                                                                    feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                                                    feature_number_index_list=[1,2],\
                                                                                    feature_number_single_value_collection=feature_number_single_value_collection,\
                                                                                    column_name_y_predicted=column_name_y_predicted,\
                                                                                    model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection,)
                    
                    
                    
                    
                    
                        
                        
                        
                    
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_feature_1_difference'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_feature_2_difference'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_1_feature_2_difference'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_difference'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_1_difference'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_2_difference'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_model_difference']=\
                    get_model_feature_and_value_star_three_metric_lists_to_collection(df=df,\
                                                                                      feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                                                      feature_number_single_value_collection=feature_number_single_value_collection,\
                                                                                      column_name_y_predicted=column_name_y_predicted,\
                                                                                      model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection,)
                    

                    
                    
                    #end of the line, get all metrics into the output list
                    metric_value_two_dimensional_list+=[model_feature_number_and_value_star_number_metric_value_list_collection['model']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_1']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_2']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_and_feature_0_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_and_feature_1_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_2']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_2_and_feature_0_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_2_and_feature_2_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_1_feature_2']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_1_feature_2_and_feature_1_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_1_feature_2_and_feature_2_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_feature_1_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_feature_2_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_1_feature_2_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_0_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_1_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_feature_2_difference']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_model_difference']]
                    
                    
                    
    return metric_value_two_dimensional_list



def get_model_feature_and_value_star_three_metric_lists_to_collection_v2(df, 
                                                                         feature_column_name_tuple_triple,
                                                                         feature_number_single_value_collection,
                                                                         column_name_y_predicted, 
                                                                         model_feature_number_and_value_star_number_metric_value_list_collection,):

    
    #get y_predicted column
    
    #get y_predicted column string
    y_model_feature_column_name_number_and_value_star_three_predicted=\
    'y_model_'+feature_column_name_tuple_triple[0]+'_'+\
               str(feature_number_single_value_collection['feature_0']).replace(' ', '_')+'_'+\
               feature_column_name_tuple_triple[1]+'_'+\
               str(feature_number_single_value_collection['feature_1']).replace(' ', '_')+'_'+\
               feature_column_name_tuple_triple[2]+'_'+\
               str(feature_number_single_value_collection['feature_2']).replace(' ', '_')+'_predicted'
    

    #print(y_model_feature_column_name_number_and_value_star_two_predicted)

    
    #get feature number key index

    
    #get predictions column by feature and value, feature and value filter. And 0 for exclusion by filter
    df.loc[:, y_model_feature_column_name_number_and_value_star_three_predicted]=0
    df.loc[(df.loc[:, feature_column_name_tuple_triple[0]]==feature_number_single_value_collection['feature_0']) & \
           (df.loc[:, feature_column_name_tuple_triple[1]]==feature_number_single_value_collection['feature_1']) & \
           (df.loc[:, feature_column_name_tuple_triple[2]]==feature_number_single_value_collection['feature_2']), y_model_feature_column_name_number_and_value_star_three_predicted]=\
    df.loc[(df.loc[:, feature_column_name_tuple_triple[0]]==feature_number_single_value_collection['feature_0']) & \
           (df.loc[:, feature_column_name_tuple_triple[1]]==feature_number_single_value_collection['feature_1']) & \
           (df.loc[:, feature_column_name_tuple_triple[2]]==feature_number_single_value_collection['feature_2']), column_name_y_predicted]


    y_true=df.loc[:, 'Y']
    y_predicted=df.loc[:, y_model_feature_column_name_number_and_value_star_three_predicted]


    #calculate feature 0 and value and feature 1 and value metric value list: precision, recall, conversions, coupons recommended

    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2']=[None]*4
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_model_difference']=[None]*4
    
    

    
    
    
    #get precision
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'][0]=precision_score(y_true=y_true, y_pred=y_predicted)

    #get recall
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'][1]=recall_score(y_true=y_true, y_pred=y_predicted)

    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(y_true=y_true, y_pred=y_predicted).ravel()

    #get conversions
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'][2]=true_positives

    #get coupons recommended
    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'][3]=true_positives+false_positives



    
    #calculate difference metrics: precision, recall, conversions, coupons recommended
    feature_number_star_number_list=['model'] #['feature_0_feature_1', 'feature_0_feature_2', 'feature_1_feature_2', 'feature_0', 'feature_1', 'feature_2',]
    key_two_dimensional_list=[['feature_0_feature_1_feature_2', feature_number_star_number] for feature_number_star_number in feature_number_star_number_list]
    
    
    
    for key_list in key_two_dimensional_list:
        
        #add output key to key list
        key_list+=[key_list[0]+'_and_'+key_list[1]+'_difference']
        
        model_feature_number_and_value_star_number_metric_value_list_collection[key_list[2]]=[None]*4
        
        model_feature_number_and_value_star_number_metric_value_list_collection[key_list[2]]=\
        get_feature_number_and_value_star_number_and_feature_number_and_value_star_number_difference_metric_value_list_collection(df=df,
                                                                                                                                  key_list=key_list,
                                                                                                                                  model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection)

        
        
    return model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'],\
           model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_model_difference']




def get_model_feature_tuple_zero_model_feature_tuple_single_model_feature_two_tuple_and_model_feature_tuple_triple_metrics_v2(df, column_name_y_predicted, feature_column_name_tuple_triple_list, random_forest_gradient_boosting_survey, tuples_completed, filename_version):
    
    metric_value_two_dimensional_list=[]
        
    model_feature_number_and_value_star_number_metric_value_list_collection={}
    
    #<><>get Model metric value list
    model_feature_number_and_value_star_number_metric_value_list_collection['model']=\
    get_model_feature_and_value_star_zero_metric_value_list(df=df, column_name_y_predicted=column_name_y_predicted, model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection)


    #get model feature two-tuple metrics
    
    combination_count=0
    
    ####
    for feature_column_name_tuple_triple in feature_column_name_tuple_triple_list:
        
        feature_column_name_number_value_list_collection={}
        
        feature_column_name_number_value_list_collection['feature_0']=list(df.loc[:, feature_column_name_tuple_triple[0]].unique())
        feature_column_name_number_value_list_collection['feature_1']=list(df.loc[:, feature_column_name_tuple_triple[1]].unique())
        feature_column_name_number_value_list_collection['feature_2']=list(df.loc[:, feature_column_name_tuple_triple[2]].unique())
        
        
        #########
        #save every 100 combinations and then reset metric_value_two_dimensional_list
        if ((combination_count%100==0) and (combination_count!=0)):

            filename='model_feature_number_and_value_star_number_metric_value_list_collection_tuple_combination_count_'+str(random_forest_gradient_boosting_survey)+'_'+str(combination_count+tuples_completed).zfill(4)+'_v'+filename_version+'.csv'

            df_feature_0_and_value_out=pd.DataFrame(metric_value_two_dimensional_list)

            #save it
            _=\
            save_and_return_data_frame_v2(df=df_feature_0_and_value_out, 
                                              filename=filename,)

            metric_value_two_dimensional_list=[]

            
        combination_count+=1
        
        ####

        feature_number_single_value_collection={}
        

        for feature_number_single_value_collection['feature_0'] in feature_column_name_number_value_list_collection['feature_0']:
                    
            #<><>
            model_feature_number_and_value_star_number_metric_value_list_collection['feature_0'],\
            _=\
            get_model_feature_and_value_star_one_metric_list(df=df,\
                                                             column_name_y_predicted=column_name_y_predicted,\
                                                             feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                             feature_column_name_tuple_triple_index=0,\
                                                             feature_column_name_number_value=feature_number_single_value_collection['feature_0'],\
                                                             model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection)

            

            
            for feature_number_single_value_collection['feature_1'] in feature_column_name_number_value_list_collection['feature_1']:

                #<><>
                model_feature_number_and_value_star_number_metric_value_list_collection['feature_1'],\
                _=\
                get_model_feature_and_value_star_one_metric_list(df=df,\
                                                                 column_name_y_predicted=column_name_y_predicted,\
                                                                 feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                                 feature_column_name_tuple_triple_index=1,\
                                                                 feature_column_name_number_value=feature_number_single_value_collection['feature_1'],\
                                                                 model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection)


             
                for feature_number_single_value_collection['feature_2'] in feature_column_name_number_value_list_collection['feature_2']:
                    
                    #<><>get feature 2 and value metrics
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_2'],\
                    _=\
                    get_model_feature_and_value_star_one_metric_list(df=df,\
                                                                     column_name_y_predicted=column_name_y_predicted,\
                                                                     feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                                     feature_column_name_tuple_triple_index=2,\
                                                                     feature_column_name_number_value=feature_number_single_value_collection['feature_2'],\
                                                                     model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection,)


                    
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2'],\
                    model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_model_difference']=\
                    get_model_feature_and_value_star_three_metric_lists_to_collection_v2(df=df,\
                                                                                         feature_column_name_tuple_triple=feature_column_name_tuple_triple,\
                                                                                         feature_number_single_value_collection=feature_number_single_value_collection,\
                                                                                         column_name_y_predicted=column_name_y_predicted,\
                                                                                         model_feature_number_and_value_star_number_metric_value_list_collection=model_feature_number_and_value_star_number_metric_value_list_collection,)
                    

                    
                    
                    #end of the line, get all metrics into the output list
                    metric_value_two_dimensional_list+=[model_feature_number_and_value_star_number_metric_value_list_collection['model']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_1']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_2']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2']+\
                                                        model_feature_number_and_value_star_number_metric_value_list_collection['feature_0_feature_1_feature_2_and_model_difference']]
    
    if(combination_count==1540-tuples_completed):

        filename='model_feature_number_and_value_star_number_metric_value_list_collection_tuple_combination_count_'+str(random_forest_gradient_boosting_survey)+'_'+str(combination_count+tuples_completed)+'_v'+filename_version+'.csv'

        df_feature_0_and_value_out=pd.DataFrame(metric_value_two_dimensional_list)

        #save it
        _=\
        save_and_return_data_frame_v2(df=df_feature_0_and_value_out, 
                                          filename=filename,)

        metric_value_two_dimensional_list=[]

                    
                    
                    
    return metric_value_two_dimensional_list















################################################################################################################################
#Get Ad Revenue, Ad Spend, ROAS, Profit, Spend, and ROI



def get_metrics_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI(df):
    venue_type_list=['Coffee House', 'Bar', 'Takeout', 'Low-Cost Restaurant', 'Mid-Range Restaurant']

    coupon_recommendation_cost_model_survey_list=['Model', 'Model']
    
    #Model Revenue, Ad Spend Metrics
    #per venue type
    df.loc[('Model', 'Ad Revenue'), :]=df.loc[('Model', 'Conversions'), :]*df.loc[('Model', 'Average Sale Estimated'), :]

    #overall
    df.loc[('Model', 'Ad Revenue'), 'Overall']=df.loc[('Model', 'Ad Revenue'), venue_type_list].sum()

    
    #per venue type
    df.loc[('Model', 'Ad Spend'), :]=df.loc[(coupon_recommendation_cost_model_survey_list[0], 'Average Coupon Recommendation Cost Estimated'), :]*df.loc[('Model', 'Coupons Recommended'), :]

    #overall
    df.loc[('Model', 'Ad Spend'), 'Overall']=df.loc[('Model', 'Ad Spend'), venue_type_list].sum()



    #Survey Revenue, Ad Spend Metrics
    #per venue type
    df.loc[('Survey', 'Ad Revenue'), :]=df.loc[('Survey', 'Conversions'), :]*df.loc[('Survey', 'Average Sale Estimated'), :]

    #overall
    df.loc[('Survey', 'Ad Revenue'), 'Overall']=df.loc[('Survey', 'Ad Revenue'), venue_type_list].sum()
    
    
    #per venue type
    df.loc[('Survey', 'Ad Spend'), :]=df.loc[(coupon_recommendation_cost_model_survey_list[1], 'Average Coupon Recommendation Cost Estimated'), :]*df.loc[('Survey', 'Coupons Recommended'), :]

    #overall
    df.loc[('Survey', 'Ad Spend'), 'Overall']=df.loc[('Survey', 'Ad Spend'), venue_type_list].sum()





    #model-survey difference metrics
    df.loc[('Model-Survey Difference', 'Ad Revenue'), :]=df.loc[('Model', 'Ad Revenue'), :]-df.loc[('Survey', 'Ad Revenue'), :]
    df.loc[('Model-Survey Difference', 'Ad Spend'), :]=df.loc[('Model', 'Ad Spend'), :]-df.loc[('Survey', 'Ad Spend'), :]
    
    
    #model, survey, model-survey difference ROAS metrics
    df.loc[('Model', 'ROAS'), :]=df.loc[('Model', 'Ad Revenue'), :]/df.loc[('Model', 'Ad Spend'), :]*100


    df.loc[('Survey', 'ROAS'), :]=df.loc[('Survey', 'Ad Revenue'), :]/df.loc[('Survey', 'Ad Spend'), :]*100


    df.loc[('Model-Survey Difference', 'ROAS'), :]=df.loc[('Model', 'ROAS'), :]-df.loc[('Survey', 'ROAS'), :]
    
    
    def calculate_Profit_Spend_ROI_with_added_production_cost(df, added_production_cost=2000):
        model_campaign_spend=df.loc[('Model', 'Ad Spend'), 'Overall']+added_production_cost
        model_campaign_profit=df.loc[('Model', 'Ad Revenue'), 'Overall']-model_campaign_spend
        
        df.loc[('Model', 'Profit '+str(added_production_cost)), 'Overall']=model_campaign_profit
        df.loc[('Model', 'Spend '+str(added_production_cost)), 'Overall']=model_campaign_spend
        df.loc[('Model', 'ROI '+str(added_production_cost)), 'Overall']=model_campaign_profit/model_campaign_spend*100

        
        
        survey_campaign_spend=df.loc[('Survey', 'Ad Spend'), 'Overall']+added_production_cost
        survey_campaign_profit=df.loc[('Survey', 'Ad Revenue'), 'Overall']-survey_campaign_spend
        
        df.loc[('Survey', 'Profit '+str(added_production_cost)), 'Overall']=survey_campaign_profit
        df.loc[('Survey', 'Spend '+str(added_production_cost)), 'Overall']=survey_campaign_spend
        df.loc[('Survey', 'ROI '+str(added_production_cost)), 'Overall']=survey_campaign_profit/survey_campaign_spend*100

        
        df.loc[('Model-Survey Difference', 'Profit '+str(added_production_cost)), 'Overall']=model_campaign_profit-survey_campaign_profit
        df.loc[('Model-Survey Difference', 'Spend '+str(added_production_cost)), 'Overall']=model_campaign_spend-survey_campaign_spend
        df.loc[('Model-Survey Difference', 'ROI '+str(added_production_cost)), 'Overall']=df.loc[('Model', 'ROI '+str(added_production_cost)), 'Overall']-df.loc[('Survey', 'ROI '+str(added_production_cost)), 'Overall']
        
        return df
    
    df=calculate_Profit_Spend_ROI_with_added_production_cost(df, added_production_cost=200)    
    df=calculate_Profit_Spend_ROI_with_added_production_cost(df, added_production_cost=2000)
    df=calculate_Profit_Spend_ROI_with_added_production_cost(df, added_production_cost=20000)

    return df

################################################################################################################################







################################################################################################################################
def get_campaign_roi_from_ad_revenue_ad_spend_additional_production_cost(ad_revenue, ad_spend, additional_production_cost):
    '''Calculate roi from ad revenue, ad spend, and additional production cost values
    
    Args:
        ad_revenue (float64): campaign ad revenue
        ad_spend (float64): camapaign ad spend
        additional_production_cost (ndarray): additional production cost
    
    Returns
        ndarray: roi values
    '''
    return (ad_revenue-ad_spend-additional_production_cost)/(ad_spend+additional_production_cost)
################################################################################################################################













#################################################################################################################################

def combine_model_metric_replicates_and_ad_revenue_ad_spend_roas_profit_spend_roi_replicates(model_type,
                                                                                             number_metric,
                                                                                             filename_version):
    '''
    Read in model name metric replicates and model name Ad Revenue, Ad Spend, ROAS, Profit, Spend, and ROI replicate files. 
    Concatenate the DataFrame's, save the result, and return it.'''
    
    #get filename_list
    filename_list=['df_'+str(model_type)+'_'+number_metric+'_estimated_feature_filter_number_bootstrap_replicates_metrics_collection_v'+str(filename_version)+'.pkl','df_test_'+str(model_type)+'_number_metric_estimated_10000_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall_v'+str(filename_version)+'.pkl','df_test_'+str(model_type)+'_number_metric_estimated_10000_metric_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall_v'+str(filename_version)+'.pkl']

    #read in files
    df_model_name_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics_overall=rpp(filename=filename_list[0])['Overall']
    df_test_model_name_number_metric_estimated_10000_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall=rcp(filename=filename_list[1], index_col=[0,1])
    
    
    #combine random forest metric replicates overall and Ad Revenue Ad Spend ROAS Profit Spend ROI replicates overall
    df_test_model_name_number_metric_estimated_10000_metric_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall=pd.concat([df_model_name_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics_overall,df_test_model_name_number_metric_estimated_10000_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall], axis=0)
    
    #save it
    df_test_model_name_number_metric_estimated_10000_metric_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall=save_and_return_data_frame_v2(df_test_model_name_number_metric_estimated_10000_metric_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall,filename=filename_list[2])
    
    return df_test_model_name_number_metric_estimated_10000_metric_Ad_Revenue_Ad_Spend_ROAS_Profit_Spend_ROI_replicates_overall

#################################################################################################################################









