import numpy as np
from numpy import float32, int64
from numpy.typing import NDArray


def euclidian_distance(
    centroid: NDArray[float32],
    cluster_member_indices: NDArray[int64],
    matrix: NDArray[float32],
):
    return np.linalg.norm(matrix[cluster_member_indices] - centroid, axis=1)


def cosine_distance(
    centroid: NDArray[float32],
    cluster_member_indices: NDArray[int64],
    matrix: NDArray[float32],
) -> NDArray[float32]:

    cluster_members = matrix[cluster_member_indices]

    centroid_normalized = _normalize(centroid)
    cluster_members_normalized = _normalize(cluster_members)

    return cluster_members_normalized @ centroid_normalized


def _normalize(vector: NDArray, axis: int = 1, keepdims: bool = True) -> NDArray:
    """
    Normalize a vector or a series of vectors

    If a matrix is passed (number of dimensions > 1)
    it is considered to be an array of vectors and each vector
    is normalized.

    Args:
        vector (NDArray): The vector or array of vectors to be normalized
        axis (int, optional): The axis to along which to calculate the norms.
            Defaults to 1.
        keepdims (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if vector.ndim == 1:
        return vector / np.linalg.norm(vector)

    return vector / np.linalg.norm(vector, axis=axis, keepdims=keepdims)
