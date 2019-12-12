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
    return random_walk_embedding(adjacency_matrix=A, transition_matrix=T, embedding_matrix=H, num_walks=w, walk_length=l)


def random_walk_embedding(adjacency_matrix: np.ndarray, transition_matrix: np.ndarray, embedding_matrix: np.ndarray, num_walks: int, walk_length: int) -> np.ndarray:
    """given an adjacency matrix A, a transition matrix T, and one-hot encoding H,
    performs random walks on the graph from each node w times with length equal to l and generate an embedding matrix for each node based
    on the sum of 1-hot encodings of all nodes that are visited during the walks.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix
        transition_matrix (np.ndarray): The transition matrix
        embedding_matrix (np.ndarray): The one-hot embedding matrix
        num_walks (int): The number of random_walks
        walk_length (int): The length of each of the random walks

    Returns:
        np.ndarray: An embedding matrix for each node, based on the sum of the embeddings of all the nodes visited during the walks.
    """
    n = adjacency_matrix.shape[0]
    visited_nodes = np.zeros([walk_length * num_walks, n], dtype=int)

    initial_seed = 123
    for walk in range(num_walks):
        # reset the state to be at the starting nodes:
        state = np.arange(n, dtype=int)
        # use a different seed everytime, otherwise we'll get the same results on each walk.
        np.random.seed(initial_seed + walk)

        for step in range(walk_length):
            # take a random step
            # TODO: figure out a way to vectorize this.
            for i in range(n):
                state[i] = np.random.choice(n, p=transition_matrix[state[i]])
            # store the visited node.
            visited_nodes[walk * num_walks + step, :] = state
    
    # print(visited_nodes)
    sum_of_embeddings = np.sum(embedding_matrix[visited_nodes], axis=0)
    return sum_of_embeddings

# 5- (5 points) Write a function hopeneighbormbd(A,H,k) which, given an adjacency matrix A,
# and one-hot node encoding matrix H, generates node embedding matrix which represents each
# node as sum of 1-hot encodings of k-hops neighbors. 

def hopeneighbormbd(A, H, k):
    return k_hop_neighbours_embeddings(adjacency_matrix=A, embedding_matrix=H, k=k)



def get_k_hop_neighbours(node: int, adjacency_matrix: np.ndarray, k: int, verbose=False) -> np.ndarray:
    """Return the K-hop neighbours of a given node in the graph represented by the given adjacency matrix.
    This is implemented as a variant of a recursive breath-first-search. Not recommended to use with a very dense graph or large k.

    Args:
        node (int): The starting node
        adjacency_matrix (np.ndarray): The adjacency matrix
        k (int): The depth parameter. when set to 1, the node's immediate neighbours are returned.
    
    Returns:
        np.ndarray: The set of K-hop node indices.
    """
    def _k_hop_neighbours(node: int, adjacency_matrix: np.ndarray, depth: int, previous_node: Optional[int] = None) -> np.ndarray:
        """Recursively find the K-hop neighbours of a given node a the graph.
        
        Args:
            node (int): The starting node
            adjacency_matrix (np.ndarray): The adjacency matrix
            depth (int): The depth parameter. when set to 1, the node's immediate neighbours are returned.
            previous_node (Optional[int], optional): The previous node which invoked the present recursive call. Do not set. Defaults to None.
        
        Returns:
            np.ndarray: The set of K-hop node indices.
        """
        n = adjacency_matrix.shape[0]
        neighbours = adjacency_matrix[node]
        neighbouring_nodes = np.nonzero(neighbours)[0]

        prefix = "\t"* (k - depth)

        if depth == 1:
            if verbose:
                print(prefix + f"node {node} has 1-hop neighbours: {neighbouring_nodes}")
            return neighbouring_nodes
        else:
            if verbose:
                print(prefix + f"Recursing into node {node}'s immediate neighbours: {neighbouring_nodes}")
            if previous_node is not None:
                neighbours = np.copy(neighbours)
                neighbours[previous_node] = False
            
            k_minus_one_neighbours = np.zeros(n, dtype=bool)

            for neighbour_node in neighbouring_nodes:
                neighbours_of_neighbour = _k_hop_neighbours(neighbour_node, adjacency_matrix, depth-1, previous_node=node)
                # print(f"{depth}: neighbouring node {neighbour_node} has {depth-1}-hop neighbours: {neighbours_of_neighbour}")

                # add the neighbours of the neighbour
                k_minus_one_neighbours[neighbours_of_neighbour] = True

            # Remove the neighbouring nodes themselves from the k-1 hop neighbours
            k_minus_one_neighbours[neighbouring_nodes] = False
            k_minus_one_neighbours[node] = False
            if verbose:
                print(prefix + f"node {node} has {depth}-hop neighbours: {np.nonzero(k_minus_one_neighbours)[0]}")
            return np.nonzero(k_minus_one_neighbours)[0]
    return _k_hop_neighbours(node, adjacency_matrix, depth=k, previous_node=None) 


def k_hop_neighbours_embeddings(adjacency_matrix: np.ndarray, embedding_matrix: np.ndarray, k: int) -> np.ndarray:
    """given an adjacency matrix A, and one-hot node encoding matrix H,
    generates node embedding matrix which represents each node as sum of 1-hot encodings of k-hops neighbors. 
    
    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix
        embedding_matrix (np.ndarray): The one-hot encoding matrix
        k (int): The number of hops
    
    Returns:
        np.ndarray: The embedding matrix.
    """
    n = adjacency_matrix.shape[0]
    if n >= 100 and k > 4:
        raise Warning(f"N or K are too large, this is probably never going to finish. (n={n},k={k})")

    embeddings = np.zeros_like(embedding_matrix)
    k_hop_neighbours = np.zeros([n,n], dtype=bool)
    for node in range(n):
        k_hop_neighbour_indices = get_k_hop_neighbours(node, adjacency_matrix, k)
        k_hop_neighbours[node][k_hop_neighbour_indices] = True
        # print(f"node {node} has {k}-hop neighbours: {k_hop_neighbour_indices}")

        neighbour_embeddings = embedding_matrix[k_hop_neighbour_indices]
        # print("neighbour embeddings:\n", neighbour_embeddings.astype(int))
        embeddings[node] = np.sum(neighbour_embeddings, axis=0)
    
    # print(k_hop_neighbours.astype(int))
    # k_hop_neighbours_2 = np.linalg.matrix_power(adjacency_matrix, k)
    # print(k_hop_neighbours_2.astype(int))
    # assert np.array_equal(k_hop_neighbours, k_hop_neighbours_2), "wtf"
    return embeddings

adjacency_matrix = adjacency_matrix_without_disconnected_nodes(100, 0.2)
# print(adjacency_matrix)
transition_matrix = transionmat(adjacency_matrix)
# print(transition_matrix)
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
# print(embeddings)
# adjacency_matrix = np.array([
#     [0, 0, 1],
#     [0, 0, 1],
#     [1, 1, 0],
# ]).astype(bool)
# embedding_matrix = np.identity(3, dtype=bool)

k = 2
k_hop_embeddings = k_hop_neighbours_embeddings(adjacency_matrix, embedding_matrix, k=k)
print(k_hop_embeddings)

