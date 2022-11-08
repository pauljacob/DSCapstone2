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




def plot_learning_curve(estimator, title, X, y, filename, axes=None, ylim=None, cv=None, n_jobs=None, scoring=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    
    filename_exists = icr.True_False_data_filename_exists(filename)
    print('filename_exists: ' + str(filename_exists))
    if filename_exists == True:
        #read in file
        learning_curve_model_name = icr.return_processed_collection_if_it_exists(filename=filename)
        
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
        learning_curve_model_name = icr.save_and_return_collection(learning_curve_dictionary_raw_mean_std, filename=filename)

    
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


