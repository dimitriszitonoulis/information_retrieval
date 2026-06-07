import time
from typing import List

from numpy import float32, int64
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from algorithms.nearest_neighbors import (
    find_approximate_nearest_neighbors,
    find_precise_nearest_neighbors,
)
from config import (
    CLUSTER_NUMBER,
    N_NEAREST_CENTROIDS,
    N_NEAREST_NEIGHBORS,
    PCA_DIMENSIONS,
    SIFT_BASE,
    SIFT_GROUNDTRUTH,
    SIFT_LEARN,
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
    n_distances_calculated = queries.shape[0] * dataset.shape[0]

    neighbors = find_precise_nearest_neighbors(
        queries=queries, n_nearest_neighbors=N_NEAREST_NEIGHBORS, dataset=dataset
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
        n_distances_calculated=n_distances_calculated,
        rec=rec,
        qps=qps,
        dataset_len=dataset.shape[0],
    )


def approximate_nn(
    dataset: NDArray[float32], queries: NDArray[float32], groundtruth: NDArray[int64]
):

    kmeans_start_time = time.perf_counter()
    centroids, labels = get_cluster_info(
        dataset=dataset,
        n_clusters=CLUSTER_NUMBER,
        n_init="auto",
        max_iter=300,
    )
    print(f"Finished kmeans after {time.perf_counter() - kmeans_start_time} seconds")

    index_start_time = time.perf_counter()
    inverted_indexes: List[InvertedIndex] = build_inverted_indexes(
        centroids, labels, CLUSTER_NUMBER, dataset
    )
    print(f"Finished building index in {time.perf_counter() - index_start_time}")

    ann_start_time = time.perf_counter()
    neighbors, n_distances_calculated = find_approximate_nearest_neighbors(
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
    # dataset = read_fvecs(SIFT_LEARN)
    dataset = read_fvecs(SIFT_BASE)
    queries = read_fvecs(SIFT_QUERY)

    groundtruth = read_ivecs(SIFT_GROUNDTRUTH)

    pca_start_time = time.perf_counter()
    pca = PCA(n_components=PCA_DIMENSIONS)
    dataset = pca.fit_transform(dataset)
    queries = pca.transform(queries)
    print(f"Finished PCA after {time.perf_counter() - pca_start_time} seconds")

    precise_nn(dataset, queries, groundtruth)

    approximate_nn(dataset, queries, groundtruth)


if __name__ == "__main__":
    main()
