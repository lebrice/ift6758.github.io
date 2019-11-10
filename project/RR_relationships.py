import numpy as np
import pandas as pd
from scipy import stats
import os

def rel_high_var(data_dir='~/Train', k=100, threshold=0.5):
    '''
    Purpose: get list of likes to keep as features
    Input:
        data_dir {str} : the parent input directory
        k {int} : the number of likes to keep as features,
                        starting from those with highest frequencies
        (optional) threshold {float} : the minimum variance criteria
    Output:
        high_var {List of strings}: width-wise truncated dataframe of likes with columns sorted by descending variance, indexed by like_id
    '''

<<<<<<< HEAD
    path="/home/rd/PycharmProjects/UdeM/6758/project" #rd local
    #path=os.path.join(data_dir, "Relation") #server

    #relation = pd.read_csv(os.path.join(data_dir, "Relation", "Relation.csv"), index_col=0)
    relation = pd.read_csv(os.path.join(path, "dummyRel.csv"))#, index_col=0)
=======
    #path="/home/rd/PycharmProjects/UdeM/6758/project" #rd local
    path=os.path.join(data_dir, "Relation") #server

    relation = pd.read_csv(os.path.join(path, "Relation.csv"))#, index_col=0)
    #relation = pd.read_csv(os.path.join(path, "dummyRel.csv"))#, index_col=0)
>>>>>>> bb4a5959ce4ca3b8b7dc04d8c36f9a0889e13eec
    relation = relation.drop(['Unnamed: 0'], axis=1)
    relation['value']=1
    #columns=relation.loc[:, 'like_id'].unique()
    #print(len(columns))
<<<<<<< HEAD
    #get count by page_id
    matrix_rel2 = relation.pivot_table(index=['like_id']).fillna(0).astype(int)
    print(matrix_rel2)
    page_id_count = relation.pivot_table(index=['like_id'], aggfunc=np.sum)['value']
    page_id_mean = relation.pivot_table(index=['like_id'], aggfunc=np.mean)['value']
    page_id_std = relation.pivot_table(index=['like_id'], aggfunc=np.std)['value']

    print (page_id_count,"\n",page_id_mean,"\n", page_id_std)
#    matrix_rel=relation.pivot_table(index=['userid'], columns=['like_id'])['value'].fillna(0).astype(int)
#    matrix_rel.to_csv(os.path.join(path, "matrix_rel.csv"), header=True)
#    cutoff=matrix_rel.std()
    ##t,p=stats.ttest_ind((matrix_rel))
#    truncated=matrix_rel.loc[:, cutoff > threshold]
    ##print(truncated)#,t,p)
#    truncated.to_csv(os.path.join(path, "", "trunk_rel.csv"), header=True)

    #return truncated

rel_high_var()
=======
    matrix_rel=relation.pivot_table(index=['userid'], columns=['like_id'])['value'].fillna(0).astype(int)
    cutoff=matrix_rel.std()
    #t,p=stats.ttest_ind((matrix_rel))
    truncated=matrix_rel.loc[:, cutoff > threshold]
    #print(truncated)#,t,p)
    truncated.to_csv(os.path.join(path, "trunk_rel.csv"), header=True)
    #matrix_rel.to_csv(os.path.join(path, "matrix_rel.csv"), header=True)
    return truncated

>>>>>>> bb4a5959ce4ca3b8b7dc04d8c36f9a0889e13eec
