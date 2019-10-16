import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from typing import *

def make_dataset(input_dir: str, userids: List[str]) -> tf.data.Dataset:
    """Creates the preprocessed text dataset for the given userid's.
    
    Arguments:
        input_dir {str} -- the parent input directory
        userids {List[str]} -- the list of userids
    
    Returns:
        tf.data.Dataset -- the preprocessed text dataset, where each entry is the feature vector.
    """
    # TODO
    #open Relationship.csv file
    df = pd.read_csv(input_dir)
    df = df.drop(['index'], axis=1)

    #Save unique userID column
    userid = df['userid'].unique()

    #Create the multihot matrix
    relHot1 = pd.get_dummies(df, columns=["like_id"])
    relHot1 = relHot1.groupby(['userid']).sum()

    #Insert the userID column
    relHot1.insert(0, "userid", userid)

    # save to csv
    #relHot1.to_csv('multiHot.csv', index=None, header=True)

    return relHot1



    raise NotImplementedError()


#Isa: working on making the likes onehotmatrix

#It isnt possible to make a oneHot matrix with all profile entries

#Sol 1: Make batches
""" 
slice the data frame into batches (for loop)
create onehot_mat_i
store it into an array?
List of onehot_mat
"""

#Sol 1.5: Get ride of the likes with a freq < LIKE_FREQ_CUTOFF

#Sol2: Dimension reduction solutions ex:PCA