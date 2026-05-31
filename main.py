import time
from typing import List

from algorithms.nearest_neighbors import find_nearest_neighbors
from config import (
    CLUSTER_NUMBER,
    N_NEAREST_CENTROIDS,
    N_NEAREST_NEIGHBORS,
    SIFT_BASE,
    SIFT_LEARN,
    SIFT_QUERY,
)
from model.inverted_index import InvertedIndex
from utils.clustering import get_cluster_info
from utils.dataset_loader import read_fvecs
from utils.inverted_index_utils import build_indexes

# sift_base = read_fvecs(SIFT_BASE)
# queries = read_fvecs(SIFT_QUERY)
# sift_gt = read_ivecs(SIFT_GROUNDTRUTH)

PCA_DIMENSIONS = 64


def main():
    dataset = read_fvecs(SIFT_LEARN)
    # dataset = read_fvecs(SIFT_BASE)
    queries = read_fvecs(SIFT_QUERY)

    kmeans_start_time = time.time()
    centroids, labels = get_cluster_info(
        dataset=dataset,
        n_clusters=CLUSTER_NUMBER,
        n_init="auto",
        max_iter=300,
    )

    print(f"Finished kmeans after {time.time() - kmeans_start_time} seconds")

    index_start_time = time.perf_counter()
    inverted_indexes: List[InvertedIndex] = build_indexes(
        centroids, labels, CLUSTER_NUMBER, dataset
    )
    print(f"Finished building index in {time.perf_counter() - index_start_time}")

    neighbors_start_time = time.perf_counter()
    neighbors, n_distances_calculated = find_nearest_neighbors(
        queries=queries[:20],
        n_nearest_neighbors=N_NEAREST_NEIGHBORS,
        n_nearest_centroids=N_NEAREST_CENTROIDS,
        centroids=centroids,
        inverted_indexes=inverted_indexes,
        matrix=dataset,
    )

    print(f"Finished finding neighbors {time.perf_counter() - neighbors_start_time}")
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
