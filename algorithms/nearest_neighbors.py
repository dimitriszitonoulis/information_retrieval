from typing import List, Tuple, cast

from numpy import float32, int64
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from model.inverted_index import InvertedIndex
from model.max_heap import MaxHeap


def find_precise_nearest_neighbors(
    queries: NDArray[float32], n_nearest_neighbors: int, dataset: NDArray[float32]
) -> NDArray[int64]:
    """
    Finds the nearest neighbors to queries and returns their indices.

    Args:
        queries (NDArray[float32]): 2D array containing queries.
        n_nearest_neighbors (int): The number of nearest neighbors
            to search for each query.
        dataset (NDArray[float32]): The dataset used to find the
            nearest neighbors.

    Returns:
        NDarray[int64]: 2D array of shape (n_queries, n_nearest_neigbors)
            containing the indices of the nearest neighbors to `queries`.
    """

    knn = NearestNeighbors(
        n_neighbors=n_nearest_neighbors, algorithm="brute", metric="euclidean"
    )
    knn.fit(dataset)

    nearest_neighbors_indices = knn.kneighbors(queries, return_distance=False)

    return cast(NDArray[int64], nearest_neighbors_indices)


def find_approximate_nearest_neighbors(
    queries: NDArray[float32],
    n_nearest_neighbors: int,
    n_nearest_centroids: int,
    centroids: NDArray[float32],
    inverted_indexes: List[InvertedIndex],
    dataset: NDArray[float32],
) -> Tuple[NDArray[int64], NDArray[int64]]:
    """
    Finds the elements that are the closest to the query.
    This is done by finding the centroids that are the
    closest to the query and then checking only the elements
    of the cluster they represent.

    The i-th element of `inverted_indexes` must contain
    the inverted index for the i-th element of `centroids`.

    Args:
        queries (NDArray[float32]): An array containing
            queries for which to find nearest neighbors.
        n_nearest_neighbors (int): The number of nearest neighbors
            to find.
        centroids (NDArray[float32]): The centroids of the clusters.
        inverted_indexes (List[InvertedIndex]): Inverted indexes
            for all the clusters.
        matrix (NDArray[float32]): The original dataset.

    Returns:
        List: The list of the nearest neighbors to the query.
    """

    n_queries = queries.shape[0]

    knn = NearestNeighbors(n_neighbors=n_nearest_centroids)
    knn.fit(centroids)

    nearest_neighbors_indices = np.empty((n_queries, n_nearest_neighbors), dtype=int64)
    n_dist_per_query = np.empty((n_queries), dtype=int64)

    for q_idx in range(n_queries):
        query_vector: NDArray[float32] = queries[q_idx]

        _, neighboring_centroids_indices = find_nearest_centroids(
            query_vector=query_vector,
            knn=knn,
        )
        neighboring_centroids_indices = neighboring_centroids_indices[0]

        neighbors, n_distances_calculated_ = _find_approximate_nearest_neighbors(
            query_vector=query_vector,
            n_nearest_neighbors=n_nearest_neighbors,
            neighboring_centroids_indices=neighboring_centroids_indices,
            inverted_indexes=inverted_indexes,
            dataset=dataset,
        )

        nearest_neighbors_indices[q_idx] = neighbors
        n_dist_per_query[q_idx] = n_distances_calculated_

    return nearest_neighbors_indices, n_dist_per_query


def _find_approximate_nearest_neighbors(
    query_vector: NDArray[float32],
    n_nearest_neighbors: int,
    neighboring_centroids_indices: NDArray[float32],
    inverted_indexes: List[InvertedIndex],
    dataset: NDArray[float32],
) -> Tuple:
    n_distances_calculated: int = 0
    neighbors: MaxHeap = MaxHeap()

    for centroid_idx in neighboring_centroids_indices:
        inverted_index: InvertedIndex = inverted_indexes[centroid_idx]

        members_indices: NDArray[int64] = inverted_index.cluster_member_indices
        members: NDArray[float32] = dataset[members_indices]
        candidate_distances: NDArray[float32] = np.linalg.norm(
            members - query_vector, axis=1
        )

        n_distances_calculated += members_indices.shape[0]

        # add to heap
        _add_vectors_to_heap(
            vectors=members_indices,
            distances=candidate_distances,
            vec_heap=neighbors,
            n_nearest_neighbors=n_nearest_neighbors,
        )

    return neighbors.get_all_values(), n_distances_calculated


def _add_vectors_to_heap(
    vectors: NDArray[int64],
    distances: NDArray[float32],
    vec_heap: MaxHeap,
    n_nearest_neighbors: int,
):
    """
    Adds the vectors with the least distance from the query to `vec_heap`.
    The heap contains at most `n_nearest_neighbors` number of vectors.

    The i-th element of `distances` must be the distance
    of the i-th element of `vectors` from the query.

    Args:
        vectors (NDArray[float32]): The vectors to check.
        distances (NDArray[float32]): The distance of each vector
            from the query.
        vec_heap (MaxHeap): A heap that contains the vectors
            with the least distance from the query.
        n_nearest_neighbors (int): The number of vectors
            that the heap should have.
    """

    # print(vectors)
    # print(f"{vectors.shape=}")

    # TODO remove else, use continue
    for i in range(vectors.shape[0]):
        vec_idx = vectors[i]
        distance = distances[i]

        if len(vec_heap) < n_nearest_neighbors:
            vec_heap.push(distance, vec_idx)
        else:
            root_q_distance, _ = vec_heap.peek()
            if distance < root_q_distance:
                vec_heap.replace(distance, vec_idx)


def find_nearest_centroids(
    query_vector: NDArray[float32],
    knn: NearestNeighbors,
):
    """
    Finds the of centroids that are the closest to `query_vector`.

    Args:
        query_vector (NDArray[float32]): The query for which
            the centroids are found.
        knn (NearestNeighbors): A NearestNeighbors object used
            to get the k nearest centroids of `query_vector`.

    Returns:
        Tuple: A tuple containing the nearest centroids as vectors.
    """
    q = query_vector.copy().reshape(1, -1)

    return knn.kneighbors(q)
