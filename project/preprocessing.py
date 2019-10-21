import os
import glob

import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf

from typing import *
from collections import Counter

import likes_preprocessing

# notes: merging in pandas
# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

def get_text_data(input_dir):
    """
    Input
        input_dir {string} : path to input_directory (ex, "~/Train")
    Output:
        id_list {numpy array of strings}: array of ids (to order feature and label matrices)
        liwc_scaler {fitted sklearn MinMaxScaler}: fitted scaler to scale liwc test
                                                    data similarly to the train data
        text_data {pandas DataFrame of float}: normalized text data (liwc and nrc)
    """
    # Build list of subject ids
    liwc = pd.read_csv(os.path.join(input_dir, 'Text/liwc.csv'), sep = ',')
    liwc = liwc.sort_values(by=['userId'])

    nrc = pd.read_csv(os.path.join(input_dir, 'Text/nrc.csv'), sep = ',')
    nrc = nrc.sort_values(by=['userId'])

    # check if same subject lists in both sorted DataFrames (liwc and nrc)
    if np.array_equal(liwc['userId'], nrc['userId']):
        id_list = liwc['userId'].to_numpy()
    else:
        # returns numpy arrays of sorted unique subject ids
        liwc_list = np.sort(liwc['userId'].unique(), axis=None)
        nrc_list = np.sort(nrc['userId'].unique(), axis=None)

        id_list = np.union1d(liwc_list, nrc_list)
        '''
        TO DO: re-order liwc and nrc matrices to match id_list order
        '''
        # code here...
    '''
    normalize liwc data (range between 0 and 85100) by squishing between 0 and 1
    Note: nrc data is already between 0 and 1
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029
    '''
    liwc.set_index('userId', inplace=True)
    liwc_scaler = preprocessing.MinMaxScaler()
    liwc_scaler = liwc_scaler.fit(liwc)
    liwc_scaled_vals = liwc_scaler.transform(liwc) #Note: returns numpy array, not pd.DataFrame
    liwc_scaled = pd.DataFrame(liwc_scaled_vals, index = liwc.index, columns=liwc.columns)

    # merge liwc and nrc dataframes
    nrc.set_index('userId', inplace=True)
    # surprise data scaled from 0-0.5: double its sort_value
    nrc['surprise'] = nrc['surprise']*2
    text_data = pd.concat([liwc_scaled, nrc], axis=1, sort=False)

    return id_list, liwc_scaler, text_data

def get_image_dataset(data_dir, sub_ids):
    # Features extracted from Image (face metrics)
    # 7915 entries (9500 ids), some without faces, a few people with multiple faces.
    oxford = pd.read_csv(os.path.join(input_dir, 'Image/oxford.csv'), sep = ',')
    # list of ids with at least one entry
    # TO DO: determine which to chose
    ox_list = np.sort(oxford['userId'].unique(), axis=None)
    # list of ids with multiple faces in one picture
    # TO DO: create two additional features (1 hot) : 1 if multiple faces, 1 if no face
    ox_multiples = oxford['userId'][oxford['userId'].duplicated()] # 741
    # list of ids in text_list who have no face metrics in oxford
    ox_noface = np.setdiff1d(text_list, ox_list)

    return image_data

def preprocess_train(data_dir, return_scaler=True):
    '''
    Input
        data_dir {string}: path to Train data directory
        return_scaler {boolean}: if True, function returns liwc scaler
    Output:
        x_train {pandas DataFrame}: train set  vectorized features
        y_train {pandas DataFrame}: train set labels
        liwc_scaler {fitted sklearn MinMaxScaler}: fitted scaler to scale liwc test
                                                    data similarly to the train data

    TO DO: convert outputted pandas to tensorflow tf.data.Dataset...
    https://www.tensorflow.org/guide/data
    '''
    # get numpy array of subject ids,
    # a scaler object to scale the liwc test data,
    # and a pandas DataFrame of preprocessed text data (liwc and nrc)
    sub_ids, liwc_scaler, text_data = get_text_data(data_dir)

    # get pandas dataframe of oxford data
    image_data = get_image_dataset(data_dir, sub_ids)

    if return_scaler:
        return x_train, y_train, liwc_scaler
    else:
        return x_train, y_train

def get_train_val_sets(data_dir):
    '''
    Input
        data_dir {string}: path to Train data directory
    Output:
        x_train, x_val {pandas DataFrames}: train and validation sets vectorized features
        y_train, y_val {pandas DataFrames}: train and validation set labels

    TO DO: convert outputted pandas to tensorflow tf.data.Dataset?...
    https://www.tensorflow.org/guide/data
    '''
    X_data, Y_data = preprocess_train(data_dir, False)


def preprocess_test(data_dir, liwc_scaler):
    '''
    Input:
        datadir {string}: path to Test data directory
        liwc_scaler {sklearn MinMaxScaler}: to scale the liwc test data just like the train data
        the function returns a DataFrame of labels (default = False)
    Output:
        x_test {pandas DataFrame}: test set  vectorized features
    '''
    return
