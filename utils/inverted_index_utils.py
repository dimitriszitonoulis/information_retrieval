from typing import List

import numpy as np
from numpy import float32, int64
from numpy.typing import NDArray

from model.inverted_index import InvertedIndex


def build_inverted_indexes(
    centroids: NDArray[float32],
    labels: NDArray[int64],
    n_clusters: int,
) -> List[InvertedIndex]:
    """
    Each inverted index corresponds to one cluster and is represented
    by that cluster's centroid. The index contains the indices of the
    vectors in `matrix` that belong to the cluster.

    The i-th element of the inverted index list is the inverted index that
    corresponds to the i-th element of `centroids`

    Args:
        centroids (NDArray[float32]): An array of vectors where the i-th
            vector is the centroid of the i-th cluster.
        labels (NDArray[np.int32]): Index of the cluster each sample belongs to.
            ex labels[i] = 2 -> the element at position i belongs in cluster 2.
        n_clusters (int): The total number of clusters.
        matrix (NDArray[float32]): The dataset which contains all the vectors.

    Returns:
        List[InvertedIndex]: A list of all the inverted indexes.
    """
    clusters: List[List] = [[] for _ in range(n_clusters)]

    for vector_idx in range(len(labels)):
        cluster_idx = labels[vector_idx]
        clusters[cluster_idx].append(vector_idx)

    inverted_indexes: List[InvertedIndex] = []

    for cluster_idx in range(n_clusters):
        centroid = centroids[cluster_idx]
        cluster_member_indices = np.array(clusters[cluster_idx], dtype=int64)

        inv_ind = InvertedIndex(centroid, cluster_member_indices)
        inverted_indexes.append(inv_ind)

    return inverted_indexes
