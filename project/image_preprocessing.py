import pandas as pd
import numpy as np

import tensorflow as tf
from typing import *

def make_dataset(input_dir: str, userids: List[str]) -> tf.data.Dataset:
    """Creates the image dataset for the given userid's.

    Arguments:
        input_dir {str} -- the parent input directory
        userids {List[str]} -- the list of userids

    Returns:
        tf.data.Dataset -- the preprocessed image dataset, where each entry is the feature vector.
    """
    userid_dataset = tf.data.Dataset.from_tensor_slices(userids)

    image_preprocessing_fn = get_image_preprocessing_fn()

    @tf.function
    def load_image(userid: tf.Tensor) -> tf.Tensor:
        return get_image_for_user(input_dir, userid)

    preprocessing_batch_size = 100
    image_dataset = (
        userid_dataset
        .map(load_image)
        .batch(preprocessing_batch_size)
        .map(image_preprocessing_fn)
        .unbatch()
    )
    return image_dataset
