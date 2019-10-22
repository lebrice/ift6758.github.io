import os
import glob

import numpy as np
import pandas as pd

from sklearn import model_selection
import tensorflow as tf

from typing import *
from collections import Counter

import likes_preprocessing
from user import age_group_string


def get_text_data(input_dir):
    """
    Purpose: preprocess liwc and nrc
    Input
        input_dir {string} : path to input_directory (ex, "~/Train")
    Output:
        id_list {numpy array of strings}: array of user ids sorted alphabetically
                                        (to order feature and label matrices)
        liwc_min_max {tupple of two pandas Series}: min and max values of liwc features in the train set,
                                                    to scale test set liwc features.
        text_data {pandas DataFrame of float}: normalized text data (liwc and nrc combined)
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
        raise Exception('userIds do not match between liwc and nrc data')

    '''
    normalize liwc data with MinMaxScaler equation (squish between 0 and 1)
    Note: nrc data is already between 0 and 1
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029
    '''
    liwc.set_index('userId', inplace=True)

    liwc_min = liwc.min()
    liwc_max = liwc.max()

    liwc_scaled = (liwc - liwc_min) / (liwc_max - liwc_min)
    liwc_min_max = (liwc_min, liwc_max)

    # merge liwc and nrc dataframes
    nrc.set_index('userId', inplace=True)
    # surprise data scaled from 0-0.5: double its value?
    #nrc['surprise'] = nrc['surprise']*2
    text_data = pd.concat([liwc_scaled, nrc], axis=1, sort=False)

    return id_list, liwc_min_max, text_data

def get_image_data(data_dir, sub_ids):
    '''
    Purpose: preprocess oxford metrics derived from profile pictures
    Input
        input_dir {string} : path to input_directory (ex, "~/Train")
        sub_ids {numpy array of strings}: ordered list of userIDs
    Output:
        image_min_max {tupple of two pandas Series}: min and max values of oxford features in the train set,
                                                    to scale test set oxford features.
        image_data_scaled {pandas DataFrame of float}: normalized oxford image data

    '''
    # Features extracted from Image (face metrics)
    # 7915 entries (9500 ids), some without faces, a few people with multiple faces per image.
    oxford = pd.read_csv(os.path.join(input_dir, 'Image/oxford.csv'), sep = ',')
    oxford = oxford.sort_values(by=['userId'])
    '''
    NOTE: headPose_pitch has NO RANGE, drop that feature
    '''
    oxford.drop(['headPose_pitch'], axis=1, inplace=True)

    # list of ids with at least one face on image: 7174 out of 9500
    ox_list = np.sort(oxford['userId'].unique(), axis=None)
    # list of ids in text_list who have no face metrics in oxford
    ox_noface = np.setdiff1d(sub_ids, ox_list)

    # create rows with face metrics means as values for users with no face on their image
    ox_nf = pd.DataFrame(ox_noface, columns = ['userId'])
    columns = oxford.columns[2:].tolist()
    means = oxford.iloc[:, 2:].mean().tolist()
    for i in range(0, len(columns)):
        ox_nf.insert(loc=ox_nf.shape[1], column=columns[i], value=means[i], allow_duplicates=True)
    # insert column 'noface' = 1 if no face in image, else 0
    ox_nf.insert(loc=ox_nf.shape[1], column='noface', value=1, allow_duplicates=True)
    # insert column 'multiface' = 1 if many faces in image, else 0
    ox_nf.insert(loc=ox_nf.shape[1], column='multiface', value=0, allow_duplicates=True)
    ox_nf.set_index('userId', inplace=True)

    # insert column 'noface' = 1 if no face in image, else 0
    oxford.insert(loc=oxford.shape[1], column='noface', value=0, allow_duplicates=True)
    # list userIds with multiple faces
    ox_multiples = oxford['userId'][oxford['userId'].duplicated()].tolist() # length = 714
    # insert column 'multiface' = 1 if many faces in image, else 0
    oxford.insert(loc=oxford.shape[1], column='multiface', value=0, allow_duplicates=True)
    multi_mask = pd.Series([uid in ox_multiples for uid in oxford['userId']])
    i = oxford[multi_mask].index
    oxford.loc[i, 'multiface'] = 1
    # drop duplicates with same userId (keep first entry)
    oxford.drop_duplicates(subset ='userId', keep='first', inplace=True)

    # merge the two datasets
    oxford.drop(['faceID'], axis=1, inplace=True)
    oxford.set_index('userId', inplace=True)
    image_data = pd.concat([ox_nf, oxford], axis=0, sort=False).sort_values(by=['userId'])

    if not np.array_equal(image_data.index, sub_ids):
        raise Exception('userIds do not match between oxford file and id list')

    image_min = image_data.min()
    image_max = image_data.max()

    image_data_scaled = (image_data - image_min) / (image_max - image_min)
    image_min_max = (image_min, image_max)

    return image_min_max, image_data_scaled

def get_relations(data_dir, sub_ids, num_features):
    '''
    Purpose: preprocess relations dataset ('likes')

    Input:
        data_dir {str} -- the parent input directory
        sub_ids {numpy array of strings} -- the ordered list of userids
        num_features {int} -- number of features to keep (with top frequencies)

    Returns:
        relations_data -- multihot matrix of the like_id. Rows are indexed with userid
    '''
    relation = pd.read_csv(os.path.join(input_dir, 'Relation/Relation.csv'))
    relation = relation.drop(['Unnamed: 0'], axis=1)

    freq_like_id = relation['like_id'].value_counts()

    relation_data = pd.DataFrame(relation['userid'].unique(), columns = ['userid'])
    relation_data.set_index('userid', inplace=True)

    batch_size = int(num_features/10)
    for i in range(10):
        likes_kept = freq_like_id[i*batch_size:(i+1)*batch_size]
        likes_kept_inds = likes_kept.keys()
        filtered_table = relation[relation['like_id'].isin(likes_kept_inds)]
        relHot = pd.get_dummies(filtered_table, columns=['like_id'])
        relHot = relHot.groupby(['userid']).sum() # makes userid the index
        relation_data = pd.concat([relation_data, relHot], axis=1, sort=True)
        i += 1

    relation_data.fillna(0)

    if not np.array_equal(relation_data.index, sub_ids):
        raise Exception('userIds do not match between oxford file and id list')

    return relation_data

def preprocess_labels(data_dir, sub_ids):
    '''
    Purpose: preprocess entry labels from training set
    Input:
        datadir {string} : path to training data directory
        sub_ids {numpy array of strings}: list of subject ids ordered alphabetically
    Output:
        data_labels {dictionary of pandas DataFrames/Series}:
    '''
    labels = pd.read_csv(os.path.join(input_dir, 'Profile/Profile.csv'), sep = ',')
    labels = labels.sort_values(by=['userid'])
    # check if same subject ids in labels and sub_ids
    if not np.array_equal(labels['userid'].to_numpy(), sub_ids):
        raise Exception('userIds do not match between profiles labels and id list')

    gender = labels['gender']

    labels = labels.assign(age_xx_24 = lambda dt: pd.Series([int(age) <= 24 for age in dt["age"]]))
    labels = labels.assign(age_25_34 = lambda dt: pd.Series([25 <= int(age) <= 34 for age in dt["age"]]))
    labels = labels.assign(age_35_49 = lambda dt: pd.Series([35 <= int(age) <= 49 for age in dt["age"]]))
    labels = labels.assign(age_50_xx = lambda dt: pd.Series([50 <= int(age) for age in dt["age"]]))

    age_grps = labels[['age_xx_24', 'age_25_34', 'age_35_49', 'age_50_xx']]

    labels_dict = {}
    labels_dict['userid'] = labels['userid']
    labels_dict['gender'] = gender
    labels_dict['age_grps'] = age_grps
    labels_dict['ope'] = labels['ope']
    labels_dict['con'] = labels['con']
    labels_dict['ext'] = labels['ext']
    labels_dict['agr'] = labels['agr']
    labels_dict['neu'] = labels['neu']

    return labels_dict

def preprocess_train(data_dir):
    '''
    Purpose: preprocesses training dataset (with labels)
    Input
        data_dir {string}: path to Train data directory
    Output:
        features {dictionary of pandas DataFrames}: vectorized features for
                text, image and relation data from the training set
        labels {dictionary of pandas DataFrames}: ordered labels per task for the training set
        min_max {dictionary of tupples of 2 pandas Series}: min and max values of scaled features
                in train dataset, to scale test data consistently

    TO DO: convert outputted pandas to tensorflow tf.data.Dataset...
    https://www.tensorflow.org/guide/data
    '''
    # sub_ids: a numpy array of subject ids ordered alphabetically,
    # liwc_min_max: a tupple of 2 pandas series, the min and max values from liwc training features,
    # text_data: a pandas DataFrame of preprocessed text data (liwc and nrc)
    sub_ids, liwc_min_max, text_data = get_text_data(data_dir, True)

    # image_data: pandas dataframe of oxford data
    # image_min_max: a tupple of 2 pandas series, the min and max values from oxford training features
    image_data, image_min_max = get_image_data(data_dir, sub_ids)

    # multi-hot matrix of likes from train data
    likes_data = get_relations(data_dir, num_features)

    # dictionary of training set labels, per task
    labels = preprocess_labels(data_dir, sub_ids)

    features = {}
    features['userid'] = sub_ids
    features['text_data'] = text_data
    features['image_data'] = image_data
    features['likes_data'] = likes_data

    min_max = {}
    min_max['liwc'] = liwc_min_max
    min_max['image'] = image_min_max

    return features, min_max, labels

def preprocess_test(data_dir, min_max):
    '''
    Purpose: preprocesses test dataset (no labels)
    Input:
        datadir {string}: path to Test data directory
        min_max {...}: min and max values for liwc and oxford features (from train set)
    Output:
        x_test {pandas DataFrame}: test set  vectorized features
    '''
    return

def get_train_val_sets(data_dir, val_prop):
    '''
    Purpose: Splits training dataset into a train and a validation set (x = features, y = labels)
    Input
        data_dir {string}: path to Train data directory
        val_prop {float between 0 and 1}: proportion of sample in validation set
    Output:
        x_train, x_val {pandas DataFrames}: vectorized features for train and validation sets
        y_train, y_val {pandas DataFrames}: train and validation set labels

    TO DO: convert outputted pandas to tensorflow tf.data.Dataset?...
    https://www.tensorflow.org/guide/data
    '''
    x_data, y_data = preprocess_train(data_dir, False)

    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        x_data, # training features to split
        y_data, # training labels to split
        test_size = val_prop, # between 0 and 1, proportion of sample in validation set (e.g., 0.2)
        shuffle= True,
        #stratify = y_data[:1],
        random_state = 42
        )

    return x_train, x_val, y_train, y_val
