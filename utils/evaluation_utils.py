from typing import List

from numpy import float32, int64
from numpy.typing import NDArray


def recall(nearest_neighbors: NDArray[int64], groundtruth: NDArray[int64]) -> float:
    """
    Returns the calculated recall of the nearest neighbor algorithm.
    The recall is calculated based on the number of neighbors that
    were correctly guessed for each query.

    The i-th element of `nearest_neighbors` and the
    i-th element of `groundtruth` must refer to the same query.

    For both arrays the indices of the nearest neighbor for each query
    must be order based on ascending distance between the neighbor
    and the query.

    Args:
        nearest_neighbors (NDArray[int64]):
            A 2D array where the i-th element are the indices
            of the nearest neighbors of the i-th query as calculated
            by the algorithm.

        groundtruth (NDArray[int64]):
            A 2D array where the i-th element are the indices
            of the nearest neighbors of the i-th query.

    Returns:
        float: The calculated recall of the nearest neighbor algorithm.
    """

    gd_n_queries, gd_n_neighbors = groundtruth.shape
    nn_n_queries, nn_n_neighbors = nearest_neighbors.shape

    if nn_n_queries > gd_n_queries:
        raise ValueError(
            f"Nearest neighbors array is has more queries than groundtruth array ({nn_n_queries} > {gd_n_queries})"
        )

    if nn_n_neighbors > gd_n_neighbors:
        raise ValueError(
            f"Nearest neighbors array has more neighbors per query than groundtruth array ({nn_n_neighbors} > {gd_n_neighbors})"
        )

    correct_guesses: int = 0
    for predicted, true in zip(nearest_neighbors, groundtruth):
        true_set = set(true)

        for neighbor in predicted:
            if neighbor in true_set:
                correct_guesses += 1

    return correct_guesses / (nn_n_queries * nn_n_neighbors)


def recall_at_k(pred, gt, k=100):
    hits = 0
    for p, g in zip(pred, gt):
        if g[0] in p[:k]:
            hits += 1
    return hits / len(pred)


def queries_per_second(queries: NDArray[float32], time_per_query: List[float]) -> float:
    """
    Calculates the queries per second for which the nearest neighbors are found.

    Args:
        queries (NDArray): Contain
        time (List[float]): Each float is the time it took
            to find the nearest neighbors i-th element of `queries`
    Returns:
        float: Queries per second
    """
    n_queries = len(queries)
    total_time = sum(time_per_query)

    return n_queries / total_time
