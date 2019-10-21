import pandas as pd
import numpy as np

import tensorflow as tf
from typing import *


def make_dataset(input_dir, userids):
    """
    Creates DataFrame of preprocessed text dataset (1 row per userid).

    Input:
    input_dir -- the parent input directory (e.g. ~/Train)
    userids -- numpy arrays of unique userids

    Output:
    text_dataset -- pandas DataFrame of preprocessed text data (liwc and nrc).
    """
    liwc = pd.read_csv(os.path.join(input_dir, 'Text/liwc.csv'), sep = ',')
    liwc_list = np.sort(liwc['userId'].unique(), axis=None) # returns numpy array
    nrc = pd.read_csv(os.path.join(input_dir, 'Text/nrc.csv'), sep = ',')
    nrc_list = np.sort(nrc['userId'].unique(), axis=None) # returns numpy array

    userid_dataset = tf.data.Dataset.from_tensor_slices(userids)

    text_preprocessing_fn = get_text_preprocessing_fn()
    preprocessing_batch_size = 100

    text_dataset = (
        get_text_dataset(input_dir, userids)
        .batch(preprocessing_batch_size)
        .map(text_preprocessing_fn)
        .unbatch()
    )
    return text_dataset
