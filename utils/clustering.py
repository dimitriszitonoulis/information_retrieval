from typing import Literal, Tuple

from numpy import float32, int64
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def get_cluster_info(
    dataset: NDArray[np.float32],
    n_clusters: int,
    max_iter: int,
    n_init: int | Literal["auto"] = "auto",
    random_state: int = 42,
) -> Tuple[NDArray[float32], NDArray[int64]]:
    """
    Calculates the centroids and members of each cluster.

    Args:
        dataset NDArray[np.float32]: 2D array containing the dataset vectors.
        n_clusters (int): The number of clusters to split the `dataset` into.
        max_iter (int): Maximum number of iterations of the k-means algorithm for a single run.
        n_init (int | Literal["auto"]):
            Defaults to auto.
            Number of times the k-means algorithm is run with different centroid seeds.
            The final results is the best output of n_init consecutive runs in terms of inertia.
            Several runs are recommended for sparse high-dimensional problems (see Clustering sparse data with k-means).
        random_state (int):
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic.
            Defaults to 42.

    Returns:
        A tuple of 2 NDArrays
        - NDrrray[float32]: A 2D array that stores the centroids
            of each cluster as vectors.
        - NDArray[int64]: A 2D array of shape (n_samples,).
            Index of the cluster each sample belongs to

    """
    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,  # pyright: ignore
        max_iter=max_iter,
        random_state=random_state,
    )

    # get array like (n_samples)
    # each element is the index of the cluster the instance belongs to
    labels: NDArray = model.fit_predict(dataset)
    centroids: NDArray = model.cluster_centers_

    return centroids, labels
