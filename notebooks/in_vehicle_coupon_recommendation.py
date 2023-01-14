#get libraries
import pandas as pd
import os
import numpy as np
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
    Read a file from the processed_data folder.'''
    if parse_dates == None:
        return pd.read_csv(os.path.join('..', 'data', 'processed', filename), index_col=index_col)
    else:
        return pd.read_csv(os.path.join('..', 'data', 'processed', filename), parse_dates=parse_dates,  index_col=index_col)

    

def rcp_v2(filename, column_name_row_integer_location_list='infer', index_column_integer_location_list=None, parse_dates=None):
        #read it back
        return pd.read_csv(filepath_or_buffer=os.path.join('..', 'data', 'processed', filename), sep=',', delimiter=None, header=column_name_row_integer_location_list, index_col=index_column_integer_location_list, usecols=None, squeeze=None, mangle_dupe_cols=True, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=parse_dates, infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None, quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None, encoding=None, encoding_errors='strict', dialect=None, error_bad_lines=None, warn_bad_lines=None, on_bad_lines=None, delim_whitespace=False, low_memory=True, memory_map=False, float_precision=None, storage_options=None)

    

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

def save_and_return_data_frame_v2(df, filename, index=True):
    '''
    Save data frame and return it.
    
    df:data frame to save and return
    filename: name of file to save to data processed folder
    index: boolean value of whether to include data frame index in save
    '''
    
    
    #initialize for write out
    relative_directory_path = os.path.join('..', 'data', 'processed')
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
    
    return rcp_v2(filename=filename, column_name_row_integer_location_list=column_name_row_integer_location_list, index_column_integer_location_list=index_column_integer_location_list)




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


##################################################################################################################################
#feature engineering
##################################################################################################################################



##################################################################################################################################
#exploratory data analysis
##################################################################################################################################

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
    
    
    
    
    
##################################################################################################################################
#Modeling

##################################################################################################################################




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





#Train Modeling Results

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

    
    
    
#########################################################################################################
#Model Metrics
#########################################################################################################

def get_metrics_by_venue_type(df, coupon_venue_type, prediction_column_name):
    print('coupon_venue_type: ' + str(coupon_venue_type))
    df_filtered_by_venue = df.loc[df.loc[:, 'coupon_venue_type'] == coupon_venue_type, :]
    
    precision = precision_score(y_true=df_filtered_by_venue.loc[:, 'Y_test'], y_pred=df_filtered_by_venue.loc[:, prediction_column_name])
    print('precision: ' + str(precision))

    recall = recall_score(y_true=df_filtered_by_venue.loc[:, 'Y_test'], y_pred=df_filtered_by_venue.loc[:, prediction_column_name])
    print('recall: ' + str(recall))

    confusion_matrix_ndarray = confusion_matrix(y_true=df_filtered_by_venue.loc[:, 'Y_test'], y_pred=df_filtered_by_venue.loc[:, prediction_column_name])
    tn, fp, fn, number_of_conversions_correctly_predicted = confusion_matrix_ndarray.ravel()
    print('number_of_conversions_correctly_predicted: ' + str(number_of_conversions_correctly_predicted))

    number_of_conversions_predicted = \
    number_of_conversions_correctly_predicted + fp
    print('number_of_conversions_predicted: ' + str(number_of_conversions_predicted))

    number_of_conversions = \
    number_of_conversions_correctly_predicted + fn
    print('number_of_conversions: ' + str(number_of_conversions))

    
    
    
    
    
    
    
    
    
    
##############################################################################################################################
#Model Test Results
##############################################################################################################################

def get_model_predictions_from_prediction_probabilities_and_decision_threshold_proportion_metric_estimated(df, model_precision_column_name, model_recall_column_name, model_decision_threshold_column_name, df_Y_test_model_prediction_probability, model_proportion_precision=None, model_proportion_recall=None,):
    '''
    df: contains the model decision threshold per model precision and recall
    model_proportion_precision: level of precision you want predictions to have
    model_precision_column_name: name of the precision column in df
    model_decision_threshold_column_name: name of the decision threshold column in df
    df_Y_test_model_prediction_probability: prediction probabilities for some target data (e.g. Y_test) by the same model type (e.g. random forest classifier) that produced df'''
    
    if model_proportion_precision != None:
        #sort df by model recall descending
        df=df.sort_values(model_recall_column_name, ascending=False)
    
        #get first decision threshold with at least 90% precision from random forest classifier on precision-recall curve
        model_decision_threshold_number_precision = df.loc[df.loc[:, model_precision_column_name] >= model_proportion_precision, model_decision_threshold_column_name].iloc[0]

        #get y_test predictions from prediction probabilties and decision threshold 90% precision estimated
        df_Y_test_model_prediction_probability = df_Y_test_model_prediction_probability.to_list()
        Y_test_predicted_list = [1 if prediction_probability > model_decision_threshold_number_precision else 0 for prediction_probability in df_Y_test_model_prediction_probability]
        df_Y_test_predicted = pd.DataFrame(Y_test_predicted_list, columns=['Y_test_predicted'])
        
        return df_Y_test_predicted
    
    elif model_proportion_recall != None:
        #sort df by model precision descending
        df=df.sort_values(model_precision_column_name, ascending=False)
        
        #get first decision threshold with at least 80% recall from gradient boosting classifier on precision-recall curve
        model_decision_threshold_number_recall = df.loc[df.loc[:, model_recall_column_name] >= model_proportion_recall, model_decision_threshold_column_name].iloc[0]
        
        #get y_test predictions from prediction probabilties and decision threshold 80% recall estimated
        df_Y_test_model_prediction_probability = df_Y_test_model_prediction_probability.to_list()
        Y_test_predicted_list = [1 if prediction_probability > model_decision_threshold_number_recall else 0 for prediction_probability in df_Y_test_model_prediction_probability]
        df_Y_test_predicted = pd.DataFrame(Y_test_predicted_list, columns=['Y_test_predicted'])
        
        return df_Y_test_predicted

def get_survey_coupon_recommendations_by_recall_estimate(number_of_predictions, recall_estimated, random_state=200):

    np.random.seed(random_state)
    class_1_probability=recall_estimated
    class_0_probability=1-class_1_probability

    Y_test_survey_recall_estimate_predicted=np.random.choice([0, 1], size=number_of_predictions, p=[class_0_probability, class_1_probability])

    df_Y_test_survey_number_recall_estimate_predicted=pd.DataFrame(Y_test_survey_recall_estimate_predicted, columns=['Y_test_survey_'+str(round(recall_estimated*100))+'_recall_estimate_predicted'])

    return df_Y_test_survey_number_recall_estimate_predicted







def get_model_and_survey_metrics(df, model_y_predicted_column_name, survey_number_recall_estimated_y_predicted_column_name, y_actual_column_name, feature_column_name_filter, feature_column_name_filter_value_list, metrics_column_name_list=None,):
    '''
    df: data frame that contains model y predicted, y actual, and feature for filtering
    model_y_predicted_column_name: name of the column column containing model y predicted
    '''
    
    if metrics_column_name_list == None:
        metrics_column_name_list = ['model conversion rate', 'model recall', 'model number converted proportion', 'model number converted', 'model number of coupons recommended proportion','model number of coupons recommended', 'survey conversion rate', 'survey recall', 'survey number converted proportion', 'survey number converted', 'survey number of coupons recommended proportion', 'survey number of coupons recommended', 'model-survey ease of conversion', ]

    metric_list=[]
    
    #get filtered data frame
    df_filtered = df.loc[df.loc[:, feature_column_name_filter].isin(feature_column_name_filter_value_list), :]

    
    def get_model_or_survey_metric_list_by_y_predicted_column_name(df, df_filtered, y_predicted_column_name, y_actual_column_name):
        metric_list=[]
        
        #get y_true and y_predicted
        y_true=df_filtered.loc[:, y_actual_column_name]
        y_predicted=df_filtered.loc[:, y_predicted_column_name]
        
        #get precision
        metric_list += [precision_score(y_true=y_true, y_pred=y_predicted)]

        #get recall
        metric_list += [recall_score(y_true=y_true, y_pred=y_predicted)]
        
        confusion_matrix_ndarray = confusion_matrix(y_true=y_true, y_pred=y_predicted)
        tn, fp, fn, tp = confusion_matrix_ndarray.ravel()
        
        confusion_matrix_ndarray_unfiltered = confusion_matrix(y_true=df.loc[:, y_actual_column_name], y_pred=df.loc[:, y_predicted_column_name])
        tn_unfiltered, fp_unfiltered, fn_unfiltered, tp_unfiltered = confusion_matrix_ndarray_unfiltered.ravel()
        
        #get number converted proportion
        metric_list += [tp/tp_unfiltered]
        
        #get number converted
        metric_list += [tp]
        
        #get number of coupons recommended proportion
        metric_list += [(tp+fp)/(tp_unfiltered+fp_unfiltered)]
        
        #get number of coupons recommended
        metric_list += [tp+fp]
        
        metric_list += [tp/(tn+fp+fn+tp)]

        return metric_list
    
    
    #get model metric list
    model_metric_list=get_model_or_survey_metric_list_by_y_predicted_column_name(df=df, df_filtered=df_filtered, y_predicted_column_name=model_y_predicted_column_name, y_actual_column_name=y_actual_column_name,)
    
    #get survey metric list
    survey_metric_list=get_model_or_survey_metric_list_by_y_predicted_column_name(df=df, df_filtered=df_filtered, y_predicted_column_name=survey_number_recall_estimated_y_predicted_column_name, y_actual_column_name=y_actual_column_name)
    

    metric_list=model_metric_list+survey_metric_list

    return metric_list


    
    
def get_metric_multiple_index(proportion_or_percentage):

        
    metric_index_value_list = ['Conversion Rate', 'Recall', proportion_or_percentage.capitalize()+' of Conversions', 'Conversions', proportion_or_percentage.capitalize()+' of Coupons Recommended', 'Coupons Recommended', 'Conversions to Base Survey Coupons Recommended Ratio',]

    model_survey_index_value_list=['Model' for index in range(len(metric_index_value_list))]+['Survey' for index in range(len(metric_index_value_list))]

    metric_index_value_list_doubled = metric_index_value_list+metric_index_value_list

    tuple_list = list(zip(model_survey_index_value_list, metric_index_value_list_doubled))
    multiple_index=pd.MultiIndex.from_tuples(tuple_list)
    return multiple_index

    
def get_metric_confidence_interval_table_by_feature_column_name_filter_value_list_dictionary_key(df_y_test_model_name_predicted_y_test_survey_recall_estimate_predicted_y_actual_feature_column_name_filter, feature_column_name_filter, feature_column_name_filter_value_list_dictionary_key, feature_column_name_filter_value_list_dictionary, multiple_index, number_of_replicates, quantile_lower_upper_list, model_type,survey_number_recall_estimated_y_predicted_column_name, filename_version, save_metric_replicates_feature_column_name_filter_value_list_dictionary_key_list=[],):


    def get_number_model_and_survey_metric_replicates_from_number_parametric_subsamples(df, number_of_replicates, model_type, survey_number_recall_estimated_y_predicted_column_name, feature_column_name_filter, feature_column_name_filter_value_list,):
        metric_list_collection = {}

        np.random.seed(seed=200)
        for index in range(number_of_replicates):

            df_parametric_bootstrap_sample=df_y_test_model_name_predicted_y_test_survey_recall_estimate_predicted_y_actual_feature_column_name_filter.sample(n=None, frac=1, replace=True, weights=None, random_state=None, axis=0, ignore_index=False)


            metric_list_collection[index]=get_model_and_survey_metrics(df=df_parametric_bootstrap_sample, model_y_predicted_column_name='Y_test_'+model_type+'_predicted', survey_number_recall_estimated_y_predicted_column_name=survey_number_recall_estimated_y_predicted_column_name, y_actual_column_name='Y', feature_column_name_filter=feature_column_name_filter, feature_column_name_filter_value_list=feature_column_name_filter_value_list, metrics_column_name_list=None,)

        df_model_number_metric_estimate_metrics_feature_filter_number_bootstrap_replicates_metric=pd.DataFrame(metric_list_collection, index=multiple_index)

        return df_model_number_metric_estimate_metrics_feature_filter_number_bootstrap_replicates_metric

    
    
    
    
    #get model and survey metric replicates from the 10,000 parametric subsamples
    df_filename='df_'+model_type+'_number_metric_estimated_'+str(number_of_replicates)+'_metric_replicates_from_'+str(number_of_replicates)+'_parametric_subsamples_'+ feature_column_name_filter_value_list_dictionary_key.lower() +'_'+ filename_version + '.csv'

    df_readback = return_processed_data_file_if_it_exists_v2(filename=df_filename, column_name_row_integer_location_list=[0,], index_column_integer_location_list=[0,1])
    if df_readback.empty != True:
        df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics = df_readback
    else:
        df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics=get_number_model_and_survey_metric_replicates_from_number_parametric_subsamples(df=df_y_test_model_name_predicted_y_test_survey_recall_estimate_predicted_y_actual_feature_column_name_filter, number_of_replicates=number_of_replicates, model_type=model_type, survey_number_recall_estimated_y_predicted_column_name=survey_number_recall_estimated_y_predicted_column_name, feature_column_name_filter=feature_column_name_filter, feature_column_name_filter_value_list=feature_column_name_filter_value_list_dictionary[feature_column_name_filter_value_list_dictionary_key])

        if feature_column_name_filter_value_list_dictionary_key in save_metric_replicates_feature_column_name_filter_value_list_dictionary_key_list:

            #save it
            df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics=save_and_return_data_frame_v2(df=df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics, filename=df_filename, index=True)
            #print(df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics.columns)
            #print(df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics.index)

    def get_metric_quatiles_from_number_parametric_subsample_replicates_metrics(df, quantile_lower_upper_list):

        return df.quantile(q=quantile_lower_upper_list, axis=1, numeric_only=True, interpolation='linear').T

    #get 95% confidence interval upper and lower quantiles
    model_survey_quantiles_of_parametric_subsample_replicates_metrics=get_metric_quatiles_from_number_parametric_subsample_replicates_metrics(df=df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics, quantile_lower_upper_list=quantile_lower_upper_list)


    #rename columns to multi index
    column_name_number_confidence_interval=str((round((quantile_lower_upper_list[1] - quantile_lower_upper_list[0])*100)))+'%'+' Confidence Interval'
    multiple_index=pd.MultiIndex.from_tuples([(column_name_number_confidence_interval, quantile_lower_upper_list[0]), 
                                              (column_name_number_confidence_interval, quantile_lower_upper_list[1])],)
    model_survey_quantiles_of_parametric_subsample_replicates_metrics.columns=multiple_index





    def convert_quantile_columns_to_confidence_interval_column(model_survey_quantiles_of_parametric_subsample_replicates_metrics, feature_column_name_filter_value_list_dictionary_key):
        #transpose to wide
        model_survey_quantiles_of_parametric_subsample_replicates_metrics=model_survey_quantiles_of_parametric_subsample_replicates_metrics.T

        #convert counts to int64
        model_survey_quantiles_of_parametric_subsample_replicates_metrics\
        .loc[:, model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>1]=\
        model_survey_quantiles_of_parametric_subsample_replicates_metrics\
        .loc[:, model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>1].astype('int64')
        model_survey_quantiles_of_parametric_subsample_replicates_metrics


        if convert_proportions_to_percentages=='yes':
            #convert to percentages from proportions in [0,1) and round to number of signficant digits

            model_survey_quantiles_of_parametric_subsample_replicates_metrics\
            .loc[:,(model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>=0) &
                 (model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<1)]=\
            round(model_survey_quantiles_of_parametric_subsample_replicates_metrics\
            .loc[:,(model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>=0) &
                 (model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<1)]*100, rate_number_of_significant_digits-2)


            #convert to 100 percent from proportions 1
            model_survey_quantiles_of_parametric_subsample_replicates_metrics\
            .loc[:,model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]==1]=100

            #convert to values to type string
            model_survey_quantiles_of_parametric_subsample_replicates_metrics=\
            model_survey_quantiles_of_parametric_subsample_replicates_metrics.astype('string')



            #add '%' to non-count column names
            column_name_count_list=[('Model', 'Conversions'), ('Model', 'Coupons Recommended'), ('Survey', 'Conversions'), ('Survey', 'Coupons Recommended')]
            column_name_list_metric_not_count=\
            [column_name for column_name in model_survey_quantiles_of_parametric_subsample_replicates_metrics.columns.to_list() if not column_name in column_name_count_list]

            model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[:,column_name_list_metric_not_count]=\
            model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[:,column_name_list_metric_not_count]+'%'


            #transpose to tall
            model_survey_quantiles_of_parametric_subsample_replicates_metrics=\
            model_survey_quantiles_of_parametric_subsample_replicates_metrics.T

            multiple_index=get_metric_multiple_index(proportion_or_percentage='percentage')
            model_survey_quantiles_of_parametric_subsample_replicates_metrics.index=multiple_index


        elif convert_proportions_to_percentages=='no':

            model_survey_quantiles_of_parametric_subsample_replicates_metrics .loc[:,(model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>=0) & (model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<1)]=round(model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[:,(model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]>=0) & (model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]<1)], rate_number_of_significant_digits)

            model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[:,model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[(column_name_number_confidence_interval, quantile_lower_upper_list[0]),:]==1]=1

            #convert to values to type string
            model_survey_quantiles_of_parametric_subsample_replicates_metrics=model_survey_quantiles_of_parametric_subsample_replicates_metrics.astype('string')


            #transpose to tall
            model_survey_quantiles_of_parametric_subsample_replicates_metrics=model_survey_quantiles_of_parametric_subsample_replicates_metrics.T


        if keep_quantiles_confidence_interval_both=='quantiles':
            return model_survey_quantiles_of_parametric_subsample_replicates_metrics

        else:
            #get 95% confidence interval column from 2.5% and 97.5% quantile columns

            model_survey_number_confidence_interval_of_parametric_subsample_replicates_metrics= model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[:, (column_name_number_confidence_interval, quantile_lower_upper_list[0])]+'-'+model_survey_quantiles_of_parametric_subsample_replicates_metrics.loc[:, (column_name_number_confidence_interval, quantile_lower_upper_list[1])]


            model_survey_number_confidence_interval_of_parametric_subsample_replicates_metrics=model_survey_number_confidence_interval_of_parametric_subsample_replicates_metrics.to_frame()
            model_survey_number_confidence_interval_of_parametric_subsample_replicates_metrics.columns=pd.MultiIndex.from_tuples([(column_name_number_confidence_interval, feature_column_name_filter_value_list_dictionary_key)])

            if keep_quantiles_confidence_interval_both=='confidence_interval':
                return model_survey_number_confidence_interval_of_parametric_subsample_replicates_metrics


            elif keep_quantiles_confidence_interval_both=='both':
                model_survey_quantile_and_number_confidence_interval_of_parametric_subsample_replicates_metrics=\
                pd.concat([model_survey_quantiles_of_parametric_subsample_replicates_metrics, model_survey_number_confidence_interval_of_parametric_subsample_replicates_metrics],axis=1)

                return model_survey_quantile_and_number_confidence_interval_of_parametric_subsample_replicates_metrics





    keep_quantiles_confidence_interval_both='confidence_interval'

    convert_proportions_to_percentages='yes'

    rate_number_of_significant_digits=3


    model_survey_number_confidence_interval_of_parametric_subsample_replicates_metrics= convert_quantile_columns_to_confidence_interval_column(model_survey_quantiles_of_parametric_subsample_replicates_metrics=model_survey_quantiles_of_parametric_subsample_replicates_metrics, feature_column_name_filter_value_list_dictionary_key=feature_column_name_filter_value_list_dictionary_key)

    return model_survey_number_confidence_interval_of_parametric_subsample_replicates_metrics, df_model_number_metric_estimated_feature_filter_number_bootstrap_replicates_metrics
    
    


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


def get_survey_metrics(df, feature_column_name, feature_column_name_value_list, target_column_name):
    print('SURVEY')
    print(str(feature_column_name) + ': ' + str(feature_column_name_value_list))

    number_of_coupons_recommendeded = df.loc[df.loc[:, feature_column_name].isin(feature_column_name_value_list), 'Y'].shape[0]
    print('number_of_coupons_recommendeded: ' + str(number_of_coupons_recommendeded))
    
    conversion_rate = (df.loc[df.loc[:, feature_column_name].isin(feature_column_name_value_list), 'Y'].value_counts() / number_of_coupons_recommendeded)[1]
    print('conversion_rate: ' + str(conversion_rate))




####################################################
#Survey Analysis
################################################################

def plot_bar_graph(df, 
                   x='coupon_venue_type', 
                   bar_category_list=['Refused Coupon', 'Accepted Coupon'],
                   title='Coupon Venue Count and Percentage per Acceptance or Refusal', 
                   color=['#8c6bb1', '#41ab5d'], 
                   figsize=(12, 10),
                   figure_filename=None,
                   dpi=100):

    #plt.figure(figsize=(10, 10))

    #sns.set(style="darkgrid")

    figure_filename_exists = os.path.isfile(figure_filename)
    if figure_filename_exists == True:
        img = mpimg.imread(figure_filename)
        plt.figure(figsize=(20, 16))
        plt.grid(False)
        plt.axis('off')
        plt.imshow(img)
    else:
        df.plot(x=x, kind ='bar', stacked=True, title=title, mark_right=True, color=color, figsize=(12, 10))

        df_row_sum = df.loc[:, bar_category_list[0]] + df.loc[:, bar_category_list[1]]

        df_stacked_bar_percentage = df.loc[:, df.columns[1:]].div(df_row_sum, 0) * 100

        for column_name in df_stacked_bar_percentage:

            for i, (cs, ab, pc) in enumerate(zip(df.iloc[:, 1:].cumsum(1).loc[:, column_name], df.loc[:, column_name], df_stacked_bar_percentage.loc[:, column_name])):
                plt.text(i, cs-ab/2, str(np.round(pc, 1)) + '%', verticalalignment='center', horizontalalignment='center', rotation=0, fontsize=14)
        plt.ylabel('count')

        #save it
        plt.savefig(figure_filename, bbox_inches='tight', dpi=dpi)

        plt.show()




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
        
