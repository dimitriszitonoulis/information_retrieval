from numpy import int64
import numpy as np
from numpy.typing import NDArray


def recall(
    nearest_neighbors: NDArray[int64], groundtruth: NDArray[int64], n_neighbors: int
) -> float:
    """
    Returns the calculated recall of the nearest neighbor algorithm.

    The recall is calculated as:
    (The sum of all the correctly guessed nearest neighbors per query) /
    (the number of queries * number of nn per query)

    The order of the calculated and the actual nearest neighbors
    does not matter.

    The i-th element of `nearest_neighbors` and the
    i-th element of `groundtruth` must refer to the same query.

    Args:
        nearest_neighbors (NDArray[int64]):
            A 2D array where the i-th row has the indices
            of the nearest neighbors of the i-th query as calculated
            by the algorithm.
        groundtruth (NDArray[int64]):
            A 2D array where the i-th row has the indices
            of the actual nearest neighbors of the i-th query.
        n_neighbors (int): The number of neighbors that were calculated.

    Returns:
        float: The calculated recall of the nearest neighbor algorithm.
    """

    gt_n_queries, gd_n_neighbors = groundtruth.shape
    nn_n_queries, nn_n_neighbors = nearest_neighbors.shape

    if nn_n_queries > gt_n_queries:
        raise ValueError(
            f"Nearest neighbors array is has more queries than groundtruth array ({nn_n_queries} > {gt_n_queries})"
        )

    if nn_n_neighbors > gd_n_neighbors:
        raise ValueError(
            f"Nearest neighbors array has more neighbors per query than groundtruth array ({nn_n_neighbors} > {gd_n_neighbors})"
        )

    # loop over every query and find the number of common elements
    # between the predicted nearest neighbors and the actual nn.
    # Calculate the sum of correct elements over all queries.
    correct_guesses = sum(
        np.isin(predicted, gt).sum()
        for predicted, gt in zip(nearest_neighbors, groundtruth)
    )

    return correct_guesses / (gt_n_queries * n_neighbors)


def queries_per_second(n_queries: int, total_time: float) -> float:
    """
    Calculates the number of queries for which the nearest neighbors were found
    in a second.

    Args:
        n_queries (int): Number of queries used.
        total_time (List[float]): The time it took to finish
            the calculations for all the queries.
    Returns:
        float: Queries per second
    """
    return n_queries / total_time
