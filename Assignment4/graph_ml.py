"""DataScience Assignment 4 - Fabrice Normandin, ID 20142128

Graph ML
"""
# This set of assingments will teach you the differences between various node representations in graphs.
# Note that all questions are programming assingments but you do not need to use loss function to optimize the claculation of thesee embeddings.

# 1- (5 points) Write a function randadjmat(n,p) in Python which returns an adjacency matrix for a "random graph" on n vertices.
# Here p is the probability of having an edge between any pair of vertices.

import numpy as np
import itertools
import random

from typing import *


def randadjmat(n: int, p: float) -> np.ndarray:
    """Returns an adjacency matrix for a "random graph" on n vertices.

    # TODO: must the graph be necessarily connected?

    Args:
        n (int): The number of vertices in the graph.
        p (float): The probability of having an edge between any pair of vertices.

    Returns:
        np.ndarray: The resulting adjacency matrix. (an ndarray of shape [n,n] and dtype bool)
    """
    adjacency_matrix = np.zeros([n, n], dtype=bool)
    for (i, j) in itertools.permutations(range(n), 2):
        if random.random() < p:
            adjacency_matrix[i, j] = adjacency_matrix[j, i] = True
    return adjacency_matrix


def node_degrees(adjacency_matrix: np.ndarray) -> np.ndarray:
    return np.sum(adjacency_matrix, axis=1)

def adjacency_matrix_without_disconnected_nodes(n: int, p: float, max_attempts=100) -> np.ndarray:
    adjacency_matrix = randadjmat(n, p)
    for i in range(max_attempts):
        if np.all(node_degrees(adjacency_matrix) != 0):
            # print(f"succeded after {i} attempts")
            return adjacency_matrix
        adjacency_matrix = randadjmat(n, p)
    raise RuntimeError(
        f"Unable to create an adjacency matrix without any disconnected nodes after {max_attempts} attempts.")

# 2- (5 points) Write a function transionmat(A) which, given an adjacency matrix A,
# generate a transition matrix T where probability of each edge (u,v) is calculated as 1/degree(u).


def transionmat(A: np.ndarray) -> np.ndarray:
    transition_matrix = np.copy(A).astype(float)
    assert np.array_equal(transition_matrix, transition_matrix.T)

    degrees = np.sum(transition_matrix, axis=1)
    assert np.all(degrees != 0), "adjacency matrix contains nodes of degree 0!"

    nonzero_indices = np.nonzero(adjacency_matrix)
    for (u, v) in zip(*nonzero_indices):
        transition_matrix[u, v] = 1 / degrees[u]

    assert np.all(np.isclose(np.sum(transition_matrix, axis=1), 1))
    return transition_matrix



# 3- (5 points) Write a function hotembd(A) which, given an adjacency matrix A,
# generate an embedding matrix H where each node is represetned with a 1-hot vector.


def hotembd(A: np.ndarray) -> np.ndarray:
    """given an adjacency matrix A, generate an embedding matrix H where each node is represetned with a 1-hot vector.

    Args:
        A (np.ndarray): An adjacency matrix

    Returns:
        np.ndarray: An Embedding matrix
    """
    return one_hot_embedding_matrix(adjacency_matrix=A)


def one_hot_embedding_matrix(adjacency_matrix: np.ndarray) -> np.ndarray:
    # TODO: not sure I understand.
    # each node becomes a one-hot vector? How is that different than just the identity?
    return np.identity(adjacency_matrix.shape[0], dtype=bool)

# 4- (5 points) Write a function randwalkemb(A,k) which, given an adjacency matrix A, a transition matrix T, and one-hot encoding H,
# performs random walks on the graph from each node w times with lenght equal to l and generate an embedding matrix for each node based
# on the sum of 1-hot encodings of all nodes that are visited during the walks.


def randwalkembd(A, T, H, w, l):
    return random_walk_embedding(adjacency_matrix=A, transition_matrix=T, embedding_matrix=H, walk_times=w, walk_length=l)


def random_walk_embedding(adjacency_matrix: np.ndarray, transition_matrix: np.ndarray, embedding_matrix: np.ndarray, walk_times: int, walk_length: int) -> np.ndarray:
    """given an adjacency matrix A, a transition matrix T, and one-hot encoding H,
    performs random walks on the graph from each node w times with length equal to l and generate an embedding matrix for each node based
    on the sum of 1-hot encodings of all nodes that are visited during the walks.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix
        transition_matrix (np.ndarray): The transition matrix
        embedding_matrix (np.ndarray): The one-hot embedding matrix
        walk_times (int): The number of random_walks
        walk_length (int): The length of each of the random walks

    Returns:
        np.ndarray: An embedding matrix for each node, based on the sum of the embeddings of all the nodes visited during the walks.
    """
    n = adjacency_matrix.shape[0]
    visited_nodes = np.zeros([n, walk_length * walk_times], dtype=int)

    initial_seed = 123
    for time in range(walk_times):
        # start from every node:
        state = np.arange(n, dtype=int)
        np.random.seed(initial_seed + time)

        for step in range(walk_length):
            # take a random step
            # TODO: figure out a way to vectorize this.
            for i in range(n):
                state[i] = np.random.choice(n, p=transition_matrix[state[i]])
            # store the visited node.
            visited_nodes[:, time * walk_times + step] = state
    
    print(visited_nodes)
    sum_of_embeddings = np.sum(embedding_matrix[visited_nodes], axis=1)
    return sum_of_embeddings

# 5- (5 points) Write a function hopeneighbormbd(A,H,k) which, given an adjacency matrix A, and one-hot node encoding matrix H, generates node embedding matrix which represents each node as sum of 1-hot encodings of k-hobs neighbors. 


adjacency_matrix = adjacency_matrix_without_disconnected_nodes(10, 0.3)
print(adjacency_matrix)
transition_matrix = transionmat(adjacency_matrix)
print(transition_matrix)
embedding_matrix = one_hot_embedding_matrix(adjacency_matrix)

walk_times = 2
walk_length = 2

embeddings = random_walk_embedding(
    adjacency_matrix,
    transition_matrix,
    embedding_matrix,
    walk_times,
    walk_length
)
print(embeddings)

