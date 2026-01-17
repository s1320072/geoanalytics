# OneNNDTW (1-Nearest Neighbor Dynamic Time Warping) is an algorithm for computing shape similarity between time series patterns using Dynamic Time Warping (DTW) distance.
#
# This implementation provides three execution modes: sequential CPU processing, parallel multi-threaded CPU processing, and GPU-accelerated CUDA processing.
#
# The algorithm supports both original and differenced time series data.
#
# **Importing this algorithm into a python program**
#
#             from geoanalytics.oneClassClassifiers import OneNNDTW
#
#             training = pd.read_csv('training_data.csv')
#             testing = pd.read_csv('testing_data.csv')
#
#             obj = OneNNDTW()
#
#             results, top_similar = obj.run(training, testing, topK=10, mode='parallel', algorithm='DTW')
#
#             print("Total results with DTW distances:", len(results))
#             print("Top similar patterns:", top_similar.head())
#

__copyright__ = """
Copyright (C)  2022 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np
import time
import os
import psutil
from tqdm import tqdm
from numba import njit, prange, cuda


# =================== Numba-compatible Functions OUTSIDE the class ===================

@njit
def dtw_numba(A, B):
    """
    Compute Dynamic Time Warping distance between two sequences.

    :param A: First time series sequence
    :type A: np.ndarray
    :param B: Second time series sequence
    :type B: np.ndarray
    :return: DTW distance between sequences
    :rtype: float
    """
    N, M = len(A), len(B)
    d = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            d[n][m] = (A[n] - B[m]) ** 2
    D = np.zeros((N, M))
    D[0][0] = d[0][0]
    for n in range(1, N):
        D[n][0] = d[n][0] + D[n - 1][0]
    for m in range(1, M):
        D[0][m] = d[0][m] + D[0][m - 1]
    for n in range(1, N):
        for m in range(1, M):
            D[n][m] = d[n][m] + min(D[n - 1][m], D[n - 1][m - 1], D[n][m - 1])
    return D[N - 1][M - 1]


@njit(parallel=True)
def compute_dtw_parallel(testing_data, training_data):
    """
    Compute DTW distances using parallel CPU processing.

    :param testing_data: Testing sequences as numpy array
    :type testing_data: np.ndarray
    :param training_data: Training sequences as numpy array
    :type training_data: np.ndarray
    :return: Array of minimum DTW distances for each testing sequence
    :rtype: np.ndarray
    """
    num_test = testing_data.shape[0]
    num_train = training_data.shape[0]
    distances = np.empty(num_test)
    for i in prange(num_test):
        min_dist = 1e10
        for j in range(num_train):
            dist = dtw_numba(testing_data[i], training_data[j])
            if dist < min_dist:
                min_dist = dist
        distances[i] = min_dist
    return distances


@njit
def dtw_manual(A, B):
    """
    Manually computes DTW distance using explicit formula (optimized version).

    :param A: First time series sequence
    :type A: np.ndarray
    :param B: Second time series sequence
    :type B: np.ndarray
    :return: DTW distance between sequences
    :rtype: float
    """
    N, M = len(A), len(B)
    d = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            d[n, m] = (A[n] - B[m]) ** 2
    D = np.zeros((N, M))
    D[0, 0] = d[0, 0]
    for n in range(1, N):
        D[n, 0] = d[n, 0] + D[n - 1, 0]
    for m in range(1, M):
        D[0, m] = d[0, m] + D[0, m - 1]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = d[n, m] + min(D[n - 1, m], D[n - 1, m - 1], D[n, m - 1])
    return D[N - 1, M - 1]


@njit(parallel=True)
def compute_dtw_sequential_numba(testing_np, training_np):
    """
    Optimized sequential-threaded DTW computation using explicit formula.

    :param testing_np: Testing sequences as numpy array
    :type testing_np: np.ndarray
    :param training_np: Training sequences as numpy array
    :type training_np: np.ndarray
    :return: Array of minimum DTW distances for each testing sequence
    :rtype: np.ndarray
    """
    num_test = testing_np.shape[0]
    num_train = training_np.shape[0]
    distances = np.full(num_test, np.inf)  # Preallocate with large values

    for i in prange(num_test):  # Parallel loop with Numba
        for j in range(num_train):
            dist = dtw_manual(testing_np[i], training_np[j])  # Using optimized DTW
            if dist < distances[i]:
                distances[i] = dist

    return distances


# =================== Main Class ===================

class OneNNDTW:
    """
    **About this algorithm**

    :**Description**: OneNNDTW (1-Nearest Neighbor Dynamic Time Warping) computes shape similarity
                      between time series patterns using Dynamic Time Warping distance.
                      For each test pattern, it finds the minimum DTW distance to any training pattern.
                      DTW allows comparison of time series with different temporal lengths and speeds.

    :**Reference**:  Berndt, D. J., & Clifford, J. (1994). Using dynamic time warping to find patterns in time series.
                     In KDD workshop (Vol. 10, No. 16, pp. 359-370).

    :**Parameters**:    - **training** (*pd.DataFrame*) -- *Training dataset containing reference patterns.*
                        - **testing** (*pd.DataFrame*) -- *Testing dataset to evaluate for similarity.*
                        - **topK** (*int*) -- *Number of most similar patterns to return (default: -1 returns all).*
                        - **mode** (*str*) -- *Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential').*
                        - **algorithm** (*str*) -- *Algorithm variant: 'DTW' or 'difDTW' for differenced data (default: 'DTW').*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the execution process.*
                        - **endTime** (*float*) -- *To record the completion time of the execution process.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.timeSeriesSimilarity import OneNNDTW

            # Load your time series data
            training = pd.read_csv('training_patterns.csv')
            testing = pd.read_csv('testing_patterns.csv')

            # Initialize the similarity finder
            obj = OneNNDTW()

            # Run the algorithm with parallel processing
            results, top_similar = obj.run(training, testing, topK=10, mode='parallel', algorithm='DTW')

            print("Total patterns with DTW distances:", len(results))
            print("Top 5 most similar patterns:")
            print(top_similar.head(5))

            # Access statistics
            print("Memory consumption in KB:", process.memory_info().rss / 1024)

    **Example Usage**

    .. code-block:: python

            # Example 1: Basic usage with sequential processing
            obj = OneNNDTW()
            results, top_similar = obj.run(training_data, testing_data, topK=5, mode='sequential')

            # Example 2: Using differenced data with parallel processing
            results, top_similar = obj.run(training_data, testing_data, topK=20, mode='parallel', algorithm='difDTW')

            # Example 3: GPU-accelerated processing
            results, top_similar = obj.run(training_data, testing_data, topK=15, mode='cuda')

    **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

    """

    @staticmethod
    def get_statistics(start_time):
        """
        Display execution statistics after algorithm completion.

        :param start_time: Start time of the execution (from time.time())
        :type start_time: float

        Prints execution time and memory consumption in KB.
        """
        print("Total Execution Time:", time.time() - start_time)
        process = psutil.Process(os.getpid())
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    @staticmethod
    @cuda.jit
    def dtw_cuda_kernel(testing_data, training_data, result):
        """
        CUDA kernel for DTW distance computation on GPU.

        :param testing_data: Testing sequences in GPU memory
        :type testing_data: cuda.devicearray
        :param training_data: Training sequences in GPU memory
        :type training_data: cuda.devicearray
        :param result: Output array for distances in GPU memory
        :type result: cuda.devicearray
        """
        i = cuda.grid(1)
        if i < testing_data.shape[0]:
            min_dist = 1e10
            for j in range(training_data.shape[0]):
                dist = 0.0
                for k in range(testing_data.shape[1]):
                    val = testing_data[i][k] - training_data[j][k]
                    dist += val * val
                if dist < min_dist:
                    min_dist = dist
            result[i] = min_dist

    @staticmethod
    def compute_dtw_cuda(testing, training):
        """
        Compute DTW distances using CUDA GPU acceleration.

        :param testing: Testing DataFrame
        :type testing: pd.DataFrame
        :param training: Training DataFrame
        :type training: pd.DataFrame
        :return: Array of minimum DTW distances for each testing sequence
        :rtype: np.ndarray
        """
        test_np = testing.to_numpy().astype(np.float32)
        train_np = training.to_numpy().astype(np.float32)
        result = cuda.device_array(test_np.shape[0], dtype=np.float32)
        threads_per_block = 128
        blocks_per_grid = (test_np.shape[0] + threads_per_block - 1) // threads_per_block
        OneNNDTW.dtw_cuda_kernel[blocks_per_grid, threads_per_block](test_np, train_np, result)
        return result.copy_to_host()

    def compute_dtw_sequential(self, testing, training):
        """
        Compute DTW distances using sequential CPU processing.

        :param testing: Testing DataFrame
        :type testing: pd.DataFrame
        :param training: Training DataFrame
        :type training: pd.DataFrame
        :return: List of minimum DTW distances for each testing sequence
        :rtype: list
        """
        testing_np = testing.to_numpy()
        training_np = training.to_numpy()
        distances = compute_dtw_sequential_numba(testing_np, training_np)
        return distances.tolist()

    def run(self, training, testing, topK=-1, mode="sequential", algorithm="DTW"):
        """
        Main execution method for the OneNNDTW algorithm.

        :param training: Training dataset containing reference patterns
        :type training: pd.DataFrame
        :param testing: Testing dataset to evaluate for similarity
        :type testing: pd.DataFrame
        :param topK: Number of most similar patterns to return (default: -1 returns all)
        :type topK: int
        :param mode: Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential')
        :type mode: str
        :param algorithm: Algorithm variant: 'DTW' or 'difDTW' for differenced data (default: 'DTW')
        :type algorithm: str
        :return: Tuple containing (complete_results_with_DTW_distances, top_K_most_similar_patterns)
        :rtype: (pd.DataFrame, pd.DataFrame)
        :raises ValueError: If an invalid mode is specified
        :raises RuntimeError: If CUDA mode is selected but CUDA is not available
        """
        start_time = time.time()

        if algorithm == "difDTW":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        if mode == "sequential":
            distances = self.compute_dtw_sequential(testing, training)
        elif mode == "parallel":
            distances = compute_dtw_parallel(testing.to_numpy(), training.to_numpy())
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_dtw_cuda(testing, training)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'")

        testing['1NNDTW'] = distances
        sorted_df = testing.sort_values('1NNDTW').head(topK)
        self.get_statistics(start_time)
        return testing, sorted_df