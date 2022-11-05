#get libraries
import pandas as pd
import os
import numpy as np
#from functools import reduce


#get visualization libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from io import StringIO

class color:
   BOLD = '\033[1m'
   END = '\033[0m'

#ML preprocessing
from sklearn.preprocessing import StandardScaler

#get ML functions
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import __version__ as sklearn_version
import datetime
#from sklearn.pipeline import make_pipeline

#get ML metric functions
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score




####################
import os
import pandas as pd


#data wrangling
from functools import reduce

#data preprocessing


#ML
import pickle



#save object
import zipfile
import shutil


def initialize_custom_notebook_settings():
    '''
    initialize the jupyter notebook display width'''
        
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:99% !important; }</style>"))

    
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
    Read a file from the processed_data folder.'''
    if parse_dates == None:
        return pd.read_csv(os.path.join('..', 'data', 'processed', filename), index_col=index_col)
    else:
        return pd.read_csv(os.path.join('..', 'data', 'processed', filename), parse_dates=parse_dates,  index_col=index_col)

def rpp(filename, parse_dates=None):
    '''
    Save collection and return it.'''

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


    
def get_column_name_list_left_not_in_right(df_left,
                                           df_right):

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
    
    
    
    
    
##################################################################################################################################
#Modeling

##################################################################################################################################







