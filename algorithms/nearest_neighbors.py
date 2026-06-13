from typing import List, Tuple

from numpy import float32, int64
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from model.inverted_index import InvertedIndex


def find_precise_nn(
    queries: NDArray[float32],
    n_nearest_neighbors: int,
    dataset: NDArray[float32],
) -> Tuple[NDArray[int64], NDArray[int64]]:
    """
    Finds the nearest neighbors to the vectors in`queries`
    and returns their indices to `dataset`.

    Args:
        queries (NDArray[float32]): 2D array containing query vectors.
        n_nearest_neighbors (int): The number of nearest neighbors
            to search for each query.
        dataset (NDArray[float32]): The dataset used to find the
            nearest neighbors.

    Returns:
        Tuple[NDArray[int64], NDArray[int64]]:
            - 2D array of shape (n_queries, n_nearest_neigbors)
            containing the indices of the nearest neighbors to `queries`.
            - 1D array of shape (n_nearest_neighbors) containing the number
            of distances used to find the i-th neighbor.
    """

    n_queries = queries.shape[0]
    nearest_neighbors_indices = np.empty((n_queries, n_nearest_neighbors), dtype=int64)
    n_dist_per_query = np.empty((n_queries), dtype=int64)
    vector_indices: NDArray[int64] = np.arange(0, dataset.shape[0], 1, dtype=int64)

    for q_idx in range(n_queries):
        q_vector = queries[q_idx]

        neighbors, n_distances_calculated = _find_nn(
            q_vector,
            n_nearest_neighbors=n_nearest_neighbors,
            vector_indices=vector_indices,
            dataset=dataset,
        )
        nearest_neighbors_indices[q_idx] = neighbors
        n_dist_per_query[q_idx] = n_distances_calculated

    return nearest_neighbors_indices, n_dist_per_query


def find_approximate_nn(
    queries: NDArray[float32],
    n_nearest_neighbors: int,
    n_nearest_centroids: int,
    centroids: NDArray[float32],
    inverted_indexes: List[InvertedIndex],
    dataset: NDArray[float32],
) -> Tuple[NDArray[int64], NDArray[int64]]:
    """
    Finds the approximate nearest neighbors for
    each query vector in `queries`.
    This is done by finding the centroids that are the
    closest to the query and checking their cluster members
    for the nearest neighbors.

    Args:
        queries (NDArray[float32]): An array containing
            queries for which to find nearest neighbors.
        n_nearest_neighbors (int): The number of nearest neighbors
            to find.
        n_nearest_centroids (int): The number of nearest centroids
            to check.
        centroids (NDArray[float32]): 2D array of the centroid vectors.
        inverted_indexes (List[InvertedIndex]): Inverted indexes
            for all the clusters.
            The i-th element of `inverted_indexes` must contain
            the inverted index for the i-th element of `centroids`.
        dataset (NDArray[float32]): The original dataset.

    Returns:
        Tuple[NDArray[int64], NDArray[int64]]:
            - 2D array of shape (n_queries, n_nearest_neigbors)
            containing the indices of the nearest neighbors to `queries`.
            - 1D array of shape (n_nearest_neighbors) containing the number
            of distances used to find the i-th neighbor.
    """

    n_queries = queries.shape[0]
    nearest_neighbors_indices = np.empty((n_queries, n_nearest_neighbors), dtype=int64)
    n_dist_per_query = np.empty((n_queries), dtype=int64)

    knn = NearestNeighbors(n_neighbors=n_nearest_centroids)
    knn.fit(centroids)
    _, nearest_centroids_indices = knn.kneighbors(queries)

    for q_idx in range(n_queries):
        query_vector: NDArray[float32] = queries[q_idx]
        centroids_indices: NDArray[int64] = nearest_centroids_indices[q_idx]

        vector_indices_list: List[NDArray[int64]] = []

        for centroid_idx in centroids_indices:
            inverted_index: InvertedIndex = inverted_indexes[centroid_idx]
            member_indices: NDArray[int64] = inverted_index.cluster_member_indices

            vector_indices_list.append(member_indices)

        vector_indices: NDArray[int64] = np.concatenate(vector_indices_list)

        neighbors, n_distances_calculated = _find_nn(
            query_vector=query_vector,
            n_nearest_neighbors=n_nearest_neighbors,
            vector_indices=vector_indices,
            dataset=dataset,
        )

        nearest_neighbors_indices[q_idx] = neighbors
        n_dist_per_query[q_idx] = n_distances_calculated

    return nearest_neighbors_indices, n_dist_per_query


def _find_nn(
    query_vector: NDArray[float32],
    n_nearest_neighbors: int,
    vector_indices: NDArray[int64],
    dataset: NDArray[float32],
) -> Tuple[NDArray[int64], int]:
    """
    Finds the nearest neighbors of `query_vector` and returns
    their indices to `dataset`.

    Args:
        query_vector (NDArray[float32]):
            The query for which to find the nearest neighbors.
        n_nearest_neighbors (int):
            The number of nearest neighbors.
        vector_indices (NDArray[int64]):
            1D array of the indices to the dataset
            of the vectors to check.
        dataset (NDArray[float32]):
            2D array that contains all the vectors
            used for the nearest neighbor search.
            It can contain more vectors that those used

    Returns:
        Tuple[NDArray[int64], int]:
        - 1D array containing the indices (to `dataset`)
        of the nearest neighbors.
        - the number of distances calculated
        in order to find the nearest neighbors.
    """

    vectors: NDArray[float32] = dataset[vector_indices]
    distances: NDArray[float32] = np.linalg.norm(vectors - query_vector, axis=1)
    n_distances: int = distances.shape[0]

    nn_indices = np.argpartition(distances, n_nearest_neighbors)[:n_nearest_neighbors]

    return vector_indices[nn_indices], n_distances
