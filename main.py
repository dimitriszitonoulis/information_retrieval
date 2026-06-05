import time
from typing import List

from sklearn.decomposition import PCA

from algorithms.nearest_neighbors import find_approximate_nearest_neighbors
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
from utils.clustering import get_cluster_info
from utils.dataset_loader import read_fvecs, read_ivecs
from utils.evaluation_utils import queries_per_second, recall
from utils.inverted_index_utils import build_inverted_indexes


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
        queries=queries[:20],
        n_nearest_neighbors=N_NEAREST_NEIGHBORS,
        n_nearest_centroids=N_NEAREST_CENTROIDS,
        centroids=centroids,
        inverted_indexes=inverted_indexes,
    )
    ann_end_time = time.perf_counter()

    recall_1: float = recall(
        neighbors[0].reshape(1, -1), groundtruth[0].reshape(1, -1), neighbors.shape[1]
    )
    qps = queries_per_second(1, ann_end_time - ann_start_time)

    print(f"{qps=}")

    print(f"Recall for the 1st query is: {recall_1}")

    print(f"Finished finding neighbors {ann_end_time - ann_start_time}")
    print(
        f"Number of distances calculated for the 1st query: {n_distances_calculated[0]}\n"
    )
    print(f"Sift dataset size: {len(dataset)}")
    if len(dataset) == 100000:
        print("Using Sift Learn")
    else:
        print("Using Sift Base")


if __name__ == "__main__":
    main()
