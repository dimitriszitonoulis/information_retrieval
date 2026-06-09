import time
from typing import List

from numpy import float32, int64
import numpy as np
from numpy.typing import NDArray

from algorithms.nearest_neighbors import (
    find_approximate_nn,
    find_precise_nn,
)
from config import (
    CLUSTER_NUMBER,
    CLUSTERER_PATH,
    N_NEAREST_CENTROIDS,
    N_NEAREST_NEIGHBORS,
    SIFT_BASE,
    SIFT_GROUNDTRUTH,
    SIFT_QUERY,
)
from model.inverted_index import InvertedIndex
from model.inverted_index import InvertedIndex
from utils.clustering import get_cluster_info
from utils.dataset_loader import read_fvecs, read_ivecs
from utils.evaluation_utils import queries_per_second, recall
from utils.inverted_index_utils import build_inverted_indexes


def print_info(
    total_time: float,
    n_distances_calculated: int,
    rec: float,
    qps: float,
    dataset_len: int,
):
    if dataset_len == 100000:
        print("Using Sift Learn")
    else:
        print("Using Sift Base")

    print(f"Recall: {rec}")
    print(f"Queries per second: {qps}")

    print(f"Finished finding neighbors in: {total_time} sec")
    print(f"Number of distances calculated: {n_distances_calculated}\n")


def precise_nn(
    dataset: NDArray[float32], queries: NDArray[float32], groundtruth: NDArray[int64]
):
    start_time = time.perf_counter()
    # n_distances_calculated = queries.shape[0] * dataset.shape[0]
    vector_indices: NDArray[int64] = np.arange(0, dataset.shape[0], 1, dtype=int64)

    neighbors, n_distances_calculated = find_precise_nn(
        queries,
        N_NEAREST_NEIGHBORS,
        vector_indices,
        dataset,
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    rec = recall(
        nearest_neighbors=neighbors,
        groundtruth=groundtruth,
        n_neighbors=neighbors.shape[1],
    )
    qps = queries_per_second(n_queries=queries.shape[0], total_time=total_time)

    print_info(
        total_time=total_time,
        n_distances_calculated=sum(n_distances_calculated),
        rec=rec,
        qps=qps,
        dataset_len=dataset.shape[0],
    )


def approximate_nn(
    dataset: NDArray[float32], queries: NDArray[float32], groundtruth: NDArray[int64]
):

    kmeans_start_time = time.perf_counter()
    centroids, labels = get_cluster_info(
        path=CLUSTERER_PATH,
        dataset=dataset,
        n_clusters=CLUSTER_NUMBER,
        n_init="auto",
        max_iter=300,
    )
    print(f"Finished kmeans after {time.perf_counter() - kmeans_start_time} seconds")

    index_start_time = time.perf_counter()
    inverted_indexes: List[InvertedIndex] = build_inverted_indexes(
        centroids, labels, CLUSTER_NUMBER
    )
    print(f"Finished building index in {time.perf_counter() - index_start_time}")

    ann_start_time = time.perf_counter()
    neighbors, n_distances_calculated = find_approximate_nn(
        queries=queries,
        n_nearest_neighbors=N_NEAREST_NEIGHBORS,
        n_nearest_centroids=N_NEAREST_CENTROIDS,
        centroids=centroids,
        inverted_indexes=inverted_indexes,
        dataset=dataset,
    )
    ann_end_time = time.perf_counter()

    ann_total_time = ann_end_time - ann_start_time

    rec: float = recall(neighbors, groundtruth, neighbors.shape[1])
    qps: float = queries_per_second(queries.shape[0], ann_total_time)

    print_info(
        total_time=ann_total_time,
        n_distances_calculated=sum(n_distances_calculated),
        rec=rec,
        qps=qps,
        dataset_len=dataset.shape[0],
    )


def main():
    dataset = read_fvecs(SIFT_BASE)
    queries = read_fvecs(SIFT_QUERY)
    groundtruth = read_ivecs(SIFT_GROUNDTRUTH)

    precise_nn(dataset, queries, groundtruth)

    approximate_nn(dataset, queries, groundtruth)


if __name__ == "__main__":
    main()
