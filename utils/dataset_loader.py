import os

import numpy as np
from numpy.typing import NDArray

"""
The .fvecs and .ivecs file formats are binary vector formats commonly used in information retrieval datasets like SIFT1M and BIGANN.

.fvecs → stores float32 vectors
.ivecs → stores int32 vectors
Both formats store the vector dimension as the first 4 bytes (int32), followed by the vector values.

Python Code to Read .fvecs and .ivecs
"""

"""
How It Works

Binary Structure:

First int32 = vector dimension.
Followed by dim values (float32 for .fvecs, int32 for .ivecs).

Reshape:
We reshape into (n_vectors, dim+1) to separate the dimension field from the actual vector values.


Drop the First Column:
The first column is just the dimension (same for all vectors), so we remove it.




Advantages of this approach:
Works for large datasets without loading line-by-line.
Handles empty files and invalid dimensions.
Uses NumPy for speed.


If you want, I can also give you a memory-mapped version so you can read .fvecs and .ivecs without loading the entire file into RAM, which is useful for datasets like BIGANN that are hundreds of GB.
Do you want me to prepare that optimized version?
"""


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
        return data[:, 1:]  # Drop the dimension column
    except Exception as e:
        raise RuntimeError(f"Error reading {filename}: {e}")
