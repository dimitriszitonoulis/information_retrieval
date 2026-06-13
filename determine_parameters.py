import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import float32, int64
from numpy.typing import NDArray
from algorithms.nearest_neighbors import find_approximate_nn
from config import (
    CLUSTER_NUMBER,
    CLUSTERER_PATH,
    SIFT_BASE,
    SIFT_GROUNDTRUTH,
    SIFT_QUERY,
)
from model.inverted_index import InvertedIndex
from utils.clustering import get_cluster_info
from utils.dataset_loader import read_fvecs, read_ivecs
from utils.evaluation_utils import queries_per_second, recall
from utils.inverted_index_utils import build_inverted_indexes


def approximate_nn(
    n_nearest_centroids: int,
    n_nearest_neighbors: int,
    dataset: NDArray[float32],
    queries: NDArray[float32],
    groundtruth: NDArray[int64],
) -> Tuple[float, float, int]:

    centroids, labels = get_cluster_info(
        path=CLUSTERER_PATH,
        dataset=dataset,
        n_clusters=CLUSTER_NUMBER,
        n_init="auto",
        max_iter=300,
    )

    inverted_indexes: List[InvertedIndex] = build_inverted_indexes(
        centroids, labels, CLUSTER_NUMBER
    )

    ann_start_time = time.perf_counter()
    neighbors, n_distances_calculated = find_approximate_nn(
        queries=queries,
        n_nearest_neighbors=n_nearest_neighbors,
        n_nearest_centroids=n_nearest_centroids,
        centroids=centroids,
        inverted_indexes=inverted_indexes,
        dataset=dataset,
    )
    ann_end_time = time.perf_counter()

    ann_total_time = ann_end_time - ann_start_time

    rec: float = recall(neighbors, groundtruth, neighbors.shape[1])
    qps: float = queries_per_second(queries.shape[0], ann_total_time)

    return rec, qps, sum(n_distances_calculated)


def main():
    dataset = read_fvecs(SIFT_BASE)
    queries = read_fvecs(SIFT_QUERY)
    groundtruth = read_ivecs(SIFT_GROUNDTRUTH)

    max_nearest_centroids = CLUSTER_NUMBER - 1
    max_nearest_neighbors = groundtruth.shape[1] - 1

    n_nearest_centroids_static = 100
    n_nearest_neighbors_static = 100

    centroid_step = 50
    nn_step = 5

    # Pick a random sample of queries
    subset_size = 100
    test_indices = np.random.randint(0, queries.shape[0] - 1, size=subset_size)
    queries = queries[test_indices]
    groundtruth = groundtruth[test_indices]

    # get altering values for centroids
    n_nearest_centroid_values = list(range(0, max_nearest_centroids, centroid_step))
    qps_centroids = np.empty((len(n_nearest_centroid_values),), dtype=float32)
    recall_centroids = np.empty((len(n_nearest_centroid_values),), dtype=float32)
    n_distances_centroids = np.empty((len(n_nearest_centroid_values),), dtype=int64)

    # get altering values for neighbors
    n_nearest_neighbor_values = list(range(1, max_nearest_neighbors, nn_step))
    qps_neighbors = np.empty((len(n_nearest_neighbor_values),), dtype=float32)
    recall_neighbors = np.empty((len(n_nearest_neighbor_values),), dtype=float32)
    n_distances_neighbors = np.empty((len(n_nearest_neighbor_values),), dtype=int64)

    for idx, n_nearest_centroids in enumerate(n_nearest_centroid_values):
        if n_nearest_centroids == 0:
            n_nearest_centroids = 1

        print(f"{idx=}, {n_nearest_centroids=}")
        rec, qps, n_distances = approximate_nn(
            n_nearest_centroids,
            n_nearest_neighbors=n_nearest_neighbors_static,
            dataset=dataset,
            queries=queries,
            groundtruth=groundtruth,
        )
        qps_centroids[idx] = qps
        recall_centroids[idx] = rec
        n_distances_centroids[idx] = n_distances

    for idx, n_nearest_neighbors in enumerate(n_nearest_neighbor_values):
        if n_nearest_neighbors == 0:
            n_nearest_neighbors = 1 

        rec, qps, n_distances = approximate_nn(
            n_nearest_centroids=n_nearest_centroids_static,
            n_nearest_neighbors=n_nearest_neighbors,
            dataset=dataset,
            queries=queries,
            groundtruth=groundtruth,
        )
        qps_neighbors[idx] = qps
        recall_neighbors[idx] = rec
        n_distances_neighbors[idx] = n_distances

    
    _, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    axes[0, 0].plot(n_nearest_centroid_values, qps_centroids, marker="o")
    axes[0, 0].set_title("QPS (Centroids)")
    axes[0, 0].set_xlabel("Number of nearest centroids")
    axes[0, 0].set_ylabel("Queries per second")

    axes[0, 1].plot(n_nearest_centroid_values[1:], recall_centroids[1:], marker="o")
    axes[0, 1].set_title("Recall (Centroids)")
    axes[0, 1].set_xlabel("Number of nearest centroids")
    axes[0, 1].set_ylabel("Recall")

    axes[0, 2].plot(n_nearest_centroid_values, n_distances_centroids, marker="o")
    axes[0, 2].set_title("Distances (Centroids)")
    axes[0, 2].set_xlabel("Number of nearest centroids")
    axes[0, 2].set_ylabel("Distances calculated")

    axes[1, 0].plot(n_nearest_neighbor_values, qps_neighbors, marker="o")
    axes[1, 0].set_title("QPS (Neighbors)")
    axes[1, 0].set_xlabel("Number of nearest neighbors")
    axes[1, 0].set_ylabel("Queries per second")


    axes[1, 1].plot(n_nearest_neighbor_values, recall_neighbors, marker="o")
    axes[1, 1].set_title("Recall (Neighbors)")
    axes[1, 1].set_xlabel("Number of nearest neighbors")
    axes[1, 1].set_ylabel("Recall")

    axes[1, 2].plot(n_nearest_neighbor_values, n_distances_neighbors, marker="o")
    axes[1, 2].set_title("Distances (Neighbors)")
    axes[1, 2].set_xlabel("Number of nearest neighbors")
    axes[1, 2].set_ylabel("Distances calculated")


    for ax in axes.flat:
        ax.grid(True)
    
    pad = 0.001
    axes[0, 1].set_ylim(recall_centroids[1:].min() - pad, recall_centroids.max() + pad)
    axes[1, 1].set_ylim(recall_neighbors.min() - pad, recall_neighbors.max() + pad)

    plt.show()


if __name__ == "__main__":
    main()
