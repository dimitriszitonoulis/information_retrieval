import os

import numpy as np
from numpy.typing import NDArray


def read_fvecs(filename: str) -> NDArray:
    """
    Reads a .fvecs file and returns a NumPy array of shape (n_vectors, dim).
    Each vector is stored as: [dim(int32), values(float32)...]
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        data = np.fromfile(filename, dtype=np.float32)
        if data.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        # First int32 is dimension
        dim = int(data[0].view(np.int32))
        if dim <= 0:
            raise ValueError(f"Invalid dimension {dim} in file {filename}")

        # Add 1 for the dimension field
        data = data.reshape(-1, dim + 1)

        # Drop the dimension column
        return data[:, 1:]
    except Exception as e:
        raise RuntimeError(f"Error reading {filename}: {e}")


def read_ivecs(filename) -> NDArray:
    """
    Reads a .ivecs file and returns a NumPy array of shape (n_vectors, dim).
    Each vector is stored as: [dim(int32), values(int32)...]
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        data = np.fromfile(filename, dtype=np.int32)
        if data.size == 0:
            return np.empty((0, 0), dtype=np.int32)

        dim = data[0]
        if dim <= 0:
            raise ValueError(f"Invalid dimension {dim} in file {filename}")

        data = data.reshape(-1, dim + 1)

        # Drop the dimension column
        return data[:, 1:]
    except Exception as e:
        raise RuntimeError(f"Error reading {filename}: {e}")
