from typing import List, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def get_cluster_info(
    dataset: NDArray[np.float32],
    n_clusters: int,
    max_iter: int,
    n_init: int | Literal["auto"] = "auto",
    random_state=42,
):
    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )

    # get array like (n_samples)
    # each element is the index of the cluster the instance belongs to
    labels: NDArray = model.fit_predict(dataset)
    centroids: NDArray = model.cluster_centers_

    return centroids, labels


def get_distances(
    centroid: NDArray[np.float32],
    members_indexes: NDArray[np.int64],
    matrix: NDArray[np.float32],
) -> List[np.float32]:
    return np.linalg.norm(matrix[members_indexes] - centroid, axis=1)
