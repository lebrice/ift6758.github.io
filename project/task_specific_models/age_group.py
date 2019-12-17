# function

import os
import glob

import numpy as np
import pandas as pd

import tensorflow as tf

from typing import *
from collections import Counter

import sklearn

num_layers=2
dense_units=62
l1_reg=0.0025
l2_reg=0.005
dropout_rate=0.1
num_text_features = 91
num_image_features = 65
max_len = 2000

def get_text_data(input_dir):
    """
    Purpose: preprocess liwc and nrc
    Input
        input_dir {string} : path to input_directory (ex, "~/Train")
    Output:
        id_list {numpy array of strings}: array of user ids sorted alphabetically,
                                        to determine order of features and labels DataFrames
        text_data {pandas DataFrame of float}: unscaled text data (liwc and nrc combined)
    """
    # Load and sort text data
    liwc = pd.read_csv(os.path.join(input_dir, "Text", "liwc.csv"), sep = ',')
    liwc = liwc.sort_values(by=['userId'])

    nrc = pd.read_csv(os.path.join(input_dir, "Text", "nrc.csv"), sep = ',')
    nrc = nrc.sort_values(by=['userId'])

    # Build list of subject ids ordered alphabetically
    # Check if same subject lists in both sorted DataFrames (liwc and nrc)
    if np.array_equal(liwc['userId'], nrc['userId']):
        id_list = liwc['userId'].to_numpy()
    else:
        raise Exception('userIds do not match between liwc and nrc data')

    # merge liwc and nrc DataFrames using userId as index
    liwc.set_index('userId', inplace=True)
    nrc.set_index('userId', inplace=True)

    text_data = pd.concat([liwc, nrc], axis=1, sort=False)

    return id_list, text_data

def get_image_raw(data_dir):
    '''
    Purpose: preprocess oxford metrics derived from profile pictures (part 1)
    Input
        input_dir {string} : path to input_directory (ex, "~/Train")
    Output:
        image_data {pandas DataFrame of float}: unscaled oxford image data
    '''
    # Load data of oxford features extracted from profile picture (face metrics)
    # 7915 entries; some users have no face, some have multiple faces on image.
    # userids with 1+ face on image: 7174 out of 9500 (train set)
    # duplicated entries (userids with > 1 face on same image): 741 in train set
    oxford = pd.read_csv(os.path.join(data_dir, "Image", "oxford.csv"), sep = ',')
    #oxford = oxford.sort_values(by=['userId'])
    '''
    NOTE: headPose_pitch has NO RANGE, drop that feature
    '''
    oxford.drop(['headPose_pitch'], axis=1, inplace=True)

    return oxford

def get_image_clean(sub_ids, oxford, means):
    '''
    Purpose: preprocess oxford metrics derived from profile pictures (part 2)
    Input:
        sub_ids {numpy array of strings}: ordered list of userIDs
        oxford {pandas DataFrame of floats}: unscaled oxford features of users with 1+ face
        means {list of float}: mean values for each feature averaged from train set,
                    to replace missing values for userids with no face (train and test set)
    Output:
        image_data {pandas DataFrame of float}: unscaled oxford image data
                with mean values replacing missing entries
    '''
    # list of ids with at least one face on image: 7174 out of 9500 in train set
    ox_list = np.sort(oxford['userId'].unique(), axis=None)
    # list of ids in text_list who have no face metrics in oxford.csv (2326 in train set)
    ox_noface = np.setdiff1d(sub_ids, ox_list)

    # Create DataFrame for userids with no face (1 row per userid)
    # values are mean metrics averaged from users with entries (training set)
    ox_nf = pd.DataFrame(ox_noface, columns = ['userId'])
    columns = oxford.columns[2:].tolist()
    for column, mean in zip(columns, means):
        ox_nf.insert(loc=ox_nf.shape[1], column=column, value=mean, allow_duplicates=True)
    # insert column 'noface' = 1 if no face in image, else 0
    ox_nf.insert(loc=ox_nf.shape[1], column='noface', value=1, allow_duplicates=True)
    # insert column 'multiface' = 1 if many faces in image, else 0
    ox_nf.insert(loc=ox_nf.shape[1], column='multiface', value=0, allow_duplicates=True)
    ox_nf.set_index('userId', inplace=True)

    # Format DataFrame from userids with 1+ face
    # insert column 'noface' = 1 if no face in image, else 0
    oxford.insert(loc=oxford.shape[1], column='noface', value=0, allow_duplicates=True)
    # list userIds with multiple faces (714 in train set)
    ox_multiples = oxford['userId'][oxford['userId'].duplicated()].tolist()
    # insert column 'multiface' = 1 if many faces in image, else 0
    oxford.insert(loc=oxford.shape[1], column='multiface', value=0, allow_duplicates=True)
    multi_mask = pd.Series([uid in ox_multiples for uid in oxford['userId']])
    i = oxford[multi_mask].index
    oxford.loc[i, 'multiface'] = 1
    # drop duplicate entries with same userId (keep first entry per userId)
    oxford.drop_duplicates(subset ='userId', keep='first', inplace=True)

    # merge the two DataFrames
    oxford.drop(['faceID'], axis=1, inplace=True)
    oxford.set_index('userId', inplace=True)
    image_data = pd.concat([ox_nf, oxford], axis=0, sort=False).sort_values(by=['userId'])

    if not np.array_equal(image_data.index, sub_ids):
        raise Exception('userIds do not match between oxford file and id list')

    return image_data

def get_relations(data_dir: str, sub_ids: List[str], like_ids_to_keep: List[str]):
    '''
    Purpose: preprocess relations dataset ('likes')

    Input:
        data_dir {str} -- the parent input directory
        sub_ids {numpy array of strings} -- the ordered list of userids
        like_ids_to_keep {List[str]} -- The list of page IDs to keep.

    Returns:
        relations_data -- multihot matrix of the like_id. Rows are indexed with userid, entries are boolean.
    '''
    relation = pd.read_csv(os.path.join(data_dir, "Relation", "Relation.csv")) #, index_col=1)
    relation = relation.drop(['Unnamed: 0'], axis=1)

    ## One HUGE step:
    # likes_to_keep = like_ids_to_keep.keys()
    # kept_relations = relation[relation.like_id.isin(likes_to_keep)]
    # multi_hot_relations = pd.get_dummies(kept_relations, columns=["like_id"], prefix="")
    # multi_hot = multi_hot_relations.groupby(("userid")).sum()
    # return multi_hot_relations
    ###
    total_num_pages = len(like_ids_to_keep)
    # Create a multihot likes matrix of booleans (rows = userids, cols = likes), by batch
    batch_size = 1000

    # Create empty DataFrame with sub_ids as index list
    relation_data = pd.DataFrame(sub_ids, columns = ['userid'])
    relation_data.set_index('userid', inplace=True)

    for start_index in range(0, total_num_pages, batch_size):
        end_index = min(start_index + batch_size, total_num_pages)

        # sets are better for membership testing than lists.
        like_ids_for_this_batch = set(like_ids_to_keep[start_index:end_index])

        filtered_table = relation[relation['like_id'].isin(like_ids_for_this_batch)]
        ## THIS is the slow part:
        relHot = pd.get_dummies(filtered_table, columns=['like_id'], prefix="", prefix_sep="")
        ##
        relHot = relHot.groupby(['userid']).sum().astype(float) # this makes userid the index

        relation_data = pd.concat([relation_data, relHot], axis=1, sort=True)

    relation_data = relation_data.reindex(like_ids_to_keep, axis=1)
    relation_data.fillna(0.0, inplace=True)
    relation_data = relation_data.astype("bool")

    # will be different if users in relation.csv are not in sub_ids
    if not np.array_equal(relation_data.index, sub_ids):
        raise Exception(f"""userIds do not match between relation file and id list:
    {relation_data.index}
    {sub_ids}

    """)

    return relation_data

def get_likes_lists(likes_data, max_num_likes):
    '''
    Purpose: make list of lists of indices of liked pages per user
    Input:
        likes_data {pandas DataFrame}: multihot matrix of the like_id. Rows are indexed with userid, entries are boolean
    Output:
        lists_of_likes {list of lists of int}: indices of pages liked by each user,
                padded with zeros to lenght = max_num_likes

    '''
    # create list of lists of indices (one per user) corresponding to liked pages in one-hot matrix
    index_lists = []
    for index in likes_data.index:
        likes_indices = np.nonzero(likes_data.loc[index].to_numpy())[0].tolist()
        index_lists.append(likes_indices)

    # pad each list of indices with 0s to set lenght = max_num_likes
    lists_padded = tf.keras.preprocessing.sequence.pad_sequences(index_lists,
    padding='post', maxlen=max_num_likes)

    lists_of_likes = pd.DataFrame(lists_padded)

    lists_of_likes.insert(loc=lists_of_likes.shape[1], column='userid', value=likes_data.index, allow_duplicates=True)
    lists_of_likes.set_index('userid', inplace=True)

    return lists_of_likes

def preprocess_test_agemodel(test_dir, max_num_likes=2000):
    # TO DO: get the values from train set from .json file
    # q10_q90_train =
    # image_means_train =
    # likes_kept_train =
    sub_ids, text_data = get_text_data(test_dir)

    image_data_raw = get_image_raw(test_dir)
    image_data = get_image_clean(sub_ids, image_data_raw, image_means_train)

    features_to_scale = pd.concat([text_data, image_data.iloc[:, :-2]], axis=1, sort=False)

    feat_q10 = q10_q90_train[0]
    feat_q90 = q10_q90_train[1]
    feat_scaled = (features_to_scale - feat_q10) / (feat_q90 - feat_q10)

    likes_data = get_relations(test_dir, sub_ids, likes_kept_train)

    test_likes_lists = get_likes_lists(likes_data, max_num_likes)

    test_features = pd.concat([feat_scaled, image_data.iloc[:, -2:], test_likes_lists], axis=1, sort=False)

    x_test_txt = test_features.iloc[:, :91].values
    x_test_img = test_features.iloc[:, 91:156].values
    x_test_lik = test_features.iloc[:, 156:].values

    return x_test_txt, x_test_img, x_test_lik

def get_age_model() -> tf.keras.Model:
    age_model_path = 'saved_models/age_model_embedding_2000_fullset.h5'

    image_features = tf.keras.Input([num_image_features], dtype=tf.float32, name="image_features")
    text_features  = tf.keras.Input([num_text_features], dtype=tf.float32, name="text_features")
    likes_features = tf.keras.Input([max_len], dtype=tf.int32, name="likes_features")

    likes_embedding_block = tf.keras.Sequential(name="likes_embedding_block")
    likes_embedding_block.add(tf.keras.layers.Embedding(10000, 8, input_length=max_len, mask_zero=True))
    likes_embedding_block.add(tf.keras.layers.Flatten())

    condensed_likes = likes_embedding_block(likes_features)

    dense_layers = tf.keras.Sequential(name="dense_layers")
    dense_layers.add(tf.keras.layers.Concatenate())
    for i in range(num_layers):
        dense_layers.add(tf.keras.layers.Dense(
            units=dense_units,
            activation= 'tanh', #'tanh',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
            ))

        dense_layers.add(tf.keras.layers.Dropout(dropout_rate))

    features = dense_layers([text_features, image_features, condensed_likes])

    age_group = tf.keras.layers.Dense(units=4, activation="softmax", name="age_group")(features)

    model_age = tf.keras.Model(
        inputs=[text_features, image_features, likes_features],
        outputs= age_group
    )

    model_age.compile(
        optimizer = tf.keras.optimizers.get({"class_name": 'ADAM',
                                   "config": {"learning_rate": 0.0001}}),
        loss = 'categorical_crossentropy',
        metrics = ['acc', 'categorical_accuracy']
    )

    model_age.load_weights(age_model_path)

    return model_age
