from typing import List

import numpy as np
from numpy import float32, int64
from numpy.typing import NDArray

from model.inverted_index import InvertedIndex
from utils.distance import euclidian_distance


def build_indexes(
    centroids: NDArray[float32],
    labels: NDArray[np.int32],
    n_clusters: int,
    matrix: NDArray[float32],
) -> List[InvertedIndex]:
    """
    Each inverted index corresponds to one cluster and is represented
    by that cluster's centroid. The index contains the indices of the
    vectors in `matrix` that belong to the cluster, sorted in ascending
    order by their euclidian distance from the centroid.

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
    buckets: List[List] = [[] for _ in range(n_clusters)]

    for idx in range(len(labels)):
        label = labels[idx]
        buckets[label].append(idx)

    inverted_indexes: List[InvertedIndex] = []

    for i in range(n_clusters):
        centroid = centroids[i]
        cluster_member_indices = np.array(buckets[i], dtype=int64)

        inv_ind = InvertedIndex(centroid, cluster_member_indices, matrix)
        inverted_indexes.append(inv_ind)

    return inverted_indexes


def _sort_indexes(
    centroid: NDArray[float32],
    cluster_member_indexes: NDArray[int64],
    matrix: NDArray[float32],
) -> NDArray[int64]:
    """
    Sorts `cluster_member_indexes` items in descending order
    based on the distance of each item from the centroid.

    Args:
        centroid (NDArray[float32]): The centroid of the cluster
        cluster_member_indices (NDArray[int64]): Array containing the indexes
            of the each cluster member based on the original matrix.
        matrix (NDArray[float32]): The matrix containing all the vectors
            of the dataset

    Returns:
        NDArray[int64]: The `cluster_member_indexes` array sorted
            based on each member's distance from the centroid
    """
    distances = _dist(centroid, cluster_member_indexes, matrix)

    return cluster_member_indexes[np.argsort(distances)]


def _dist(
    centroid: NDArray[float32],
    cluster_member_indices: NDArray[int64],
    matrix: NDArray[float32],
):
    return euclidian_distance(
        centroid=centroid, cluster_member_indices=cluster_member_indices, matrix=matrix
    )
