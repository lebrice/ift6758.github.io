import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from typing import *

#def make_dataset(input_dir: str, userids: List[str], COUNT_CUTOFF: int, saveTocsv: bool) -> tf.data.Dataset:


def make_multihot_like_mat(input_dir: str, userids: List[str], COUNT_CUTOFF: int, saveTocsv: bool):
    """Creates the preprocessed text dataset for the given userid's.
    
    Arguments:
        input_dir {str} -- the parent input directory
        userids {List[str]} -- the list of userids
    
    Returns:
        tf.data.Dataset -- the preprocessed text dataset, where each entry is the feature vector.
    """
    # Get raw data
    df = pd.read_csv(input_dir)
    df = df.drop(['Unnamed: 0'], axis=1)

    freq_like_id = df["like_id"].value_counts()
    likes_kept = freq_like_id[freq_like_id > COUNT_CUTOFF]
    likes_kept_inds = likes_kept.keys()
    filtered_table = df[df["like_id"].isin(likes_kept_inds)]

    relHot = pd.get_dummies(filtered_table, columns=["like_id"])
    relHot = relHot.groupby(['userid']).sum()

    if saveTocsv:
        # create a userid row
        userid = relHot.index
        relHot.insert(0, "userid", userid)

        # create string: Relation_Multihot_CUTOFF.csv
        PATH = "/home/mila/teaching/user07/IsabelleWorkshop/"
        output_filename = "Relation_Multihot_" + str(COUNT_CUTOFF) + ".csv"
        # save to csv
        relHot.to_csv(PATH + output_filename, index=None, header=True)

        relHot = relHot.drop(["userid"], axis=1)


    return relHot



    raise NotImplementedError()