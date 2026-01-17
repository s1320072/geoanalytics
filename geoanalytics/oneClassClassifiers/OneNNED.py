# OneNNED (1-Nearest Neighbor Euclidean Distance) is an algorithm for computing similarity between time series patterns using Euclidean distance.
#
# This implementation provides three execution modes: sequential CPU processing with NumPy vectorization, parallel multi-threaded CPU processing, and GPU-accelerated CUDA processing.
#
# The algorithm supports both original and differenced time series data.
#
# **Importing this algorithm into a python program**
#
#             from geoanalytics.oneClassClassifiers import OneNNED
#
#             training = pd.read_csv('training_data.csv')
#             testing = pd.read_csv('testing_data.csv')
#
#             obj = OneNNED()
#
#             results, top_similar = obj.run(training, testing, topK=10, mode='parallel', algorithm='ED')
#
#             print("Total results with Euclidean distances:", len(results))
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
import math
import os
import psutil
from tqdm import tqdm
from numba import njit, prange, cuda


class OneNNED:
    """
    **About this algorithm**

    :**Description**: OneNNED (1-Nearest Neighbor Euclidean Distance) computes similarity between
                      time series patterns using Euclidean distance. For each test pattern,
                      it finds the minimum Euclidean distance to any training pattern.
                      Euclidean distance measures the straight-line distance between points
                      in multidimensional space.

    :**Reference**:  Based on the classical Euclidean distance formula from geometry and
                     pattern recognition literature. Widely used in k-NN algorithms and
                     time series classification.

    :**Parameters**:    - **training** (*pd.DataFrame*) -- *Training dataset containing reference patterns.*
                        - **testing** (*pd.DataFrame*) -- *Testing dataset to evaluate for similarity.*
                        - **topK** (*int*) -- *Number of most similar patterns to return (default: -1 returns all).*
                        - **mode** (*str*) -- *Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential').*
                        - **algorithm** (*str*) -- *Algorithm variant: 'ED' or 'difED' for differenced data (default: 'ED').*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the execution process.*
                        - **endTime** (*float*) -- *To record the completion time of the execution process.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.timeSeriesSimilarity import OneNNED

            # Load your time series data
            training = pd.read_csv('training_patterns.csv')
            testing = pd.read_csv('testing_patterns.csv')

            # Initialize the similarity finder
            obj = OneNNED()

            # Run the algorithm with parallel processing
            results, top_similar = obj.run(training, testing, topK=10, mode='parallel', algorithm='ED')

            print("Total patterns with Euclidean distances:", len(results))
            print("Top 5 most similar patterns:")
            print(top_similar.head(5))

            # Access statistics
            print("Memory consumption in KB:", process.memory_info().rss / 1024)

    **Example Usage**

    .. code-block:: python

            # Example 1: Basic usage with sequential processing
            obj = OneNNED()
            results, top_similar = obj.run(training_data, testing_data, topK=5, mode='sequential')

            # Example 2: Using differenced data with parallel processing
            results, top_similar = obj.run(training_data, testing_data, topK=20, mode='parallel', algorithm='difED')

            # Example 3: GPU-accelerated processing
            results, top_similar = obj.run(training_data, testing_data, topK=15, mode='cuda')

    **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

    """

    def __init__(self):
        """
        Initialize the OneNNED similarity finder.

        This constructor sets up the similarity finder without any required parameters.
        All configuration is passed to the run() method.
        """
        pass

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

    # ---------------------- Optimized sequential Thread (NumPy Vectorization) ----------------------
    @staticmethod
    def compute_ed_sequential(testing, training):
        """
        Compute Euclidean distances using sequential CPU processing with NumPy vectorization.

        :param testing: Testing DataFrame
        :type testing: pd.DataFrame
        :param training: Training DataFrame
        :type training: pd.DataFrame
        :return: List of minimum Euclidean distances for each testing sequence
        :rtype: list
        """
        test_np = testing.to_numpy()
        train_np = training.to_numpy()

        # Ensure both have the same number of columns (features)
        if test_np.shape[1] != train_np.shape[1]:
            min_features = min(test_np.shape[1], train_np.shape[1])
            test_np = test_np[:, :min_features]
            train_np = train_np[:, :min_features]

        # Compute pairwise squared differences
        dists = np.sqrt(((test_np[:, np.newaxis, :] - train_np[np.newaxis, :, :]) ** 2).sum(axis=2))

        # Take the minimum distance for each test sample
        return np.min(dists, axis=1).tolist()

    # ---------------------- Parallel Version with Numba ----------------------
    @staticmethod
    @njit(parallel=True)
    def compute_ed_parallel(test_np, train_np):
        """
        Compute Euclidean distances using parallel CPU processing with Numba.

        :param test_np: Testing sequences as numpy array
        :type test_np: np.ndarray
        :param train_np: Training sequences as numpy array
        :type train_np: np.ndarray
        :return: Array of minimum Euclidean distances for each testing sequence
        :rtype: np.ndarray
        """
        num_test = test_np.shape[0]
        num_train = train_np.shape[0]
        num_features = train_np.shape[1]
        distances = np.empty(num_test)

        for i in prange(num_test):
            min_dist = 1e10
            for j in range(num_train):
                sq = 0.0
                for k in range(num_features):
                    diff = test_np[i, k] - train_np[j, k]
                    sq += diff * diff
                dist = math.sqrt(sq)
                if dist < min_dist:
                    min_dist = dist
            distances[i] = min_dist

        return distances

    # ---------------------- CUDA Kernel ----------------------
    @staticmethod
    @cuda.jit
    def compute_ed_cuda_kernel(test_data, train_data, result):
        """
        CUDA kernel for Euclidean distance computation on GPU.

        :param test_data: Testing sequences in GPU memory
        :type test_data: cuda.devicearray
        :param train_data: Training sequences in GPU memory
        :type train_data: cuda.devicearray
        :param result: Output array for distances in GPU memory
        :type result: cuda.devicearray
        """
        i = cuda.grid(1)
        if i < test_data.shape[0]:
            min_dist = 1e10
            for j in range(train_data.shape[0]):
                sq = 0.0
                for k in range(test_data.shape[1]):
                    diff = test_data[i][k] - train_data[j][k]
                    sq += diff * diff
                dist = math.sqrt(sq)
                if dist < min_dist:
                    min_dist = dist
            result[i] = min_dist

    @staticmethod
    def compute_ed_cuda(testing, training):
        """
        Compute Euclidean distances using CUDA GPU acceleration.

        :param testing: Testing DataFrame
        :type testing: pd.DataFrame
        :param training: Training DataFrame
        :type training: pd.DataFrame
        :return: Array of minimum Euclidean distances for each testing sequence
        :rtype: np.ndarray
        """
        test_np = testing.to_numpy().astype(np.float32)
        train_np = training.to_numpy().astype(np.float32)
        result = cuda.device_array(test_np.shape[0], dtype=np.float32)

        threads_per_block = 128
        blocks_per_grid = (test_np.shape[0] + threads_per_block - 1) // threads_per_block

        OneNNED.compute_ed_cuda_kernel[blocks_per_grid, threads_per_block](test_np, train_np, result)
        return result.copy_to_host()

    # ---------------------- Main Function ----------------------
    def run(self, training, testing, topK=-1, mode="sequential", algorithm='ED'):
        """
        Main execution method for the OneNNED algorithm.

        :param training: Training dataset containing reference patterns
        :type training: pd.DataFrame
        :param testing: Testing dataset to evaluate for similarity
        :type testing: pd.DataFrame
        :param topK: Number of most similar patterns to return (default: -1 returns all)
        :type topK: int
        :param mode: Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential')
        :type mode: str
        :param algorithm: Algorithm variant: 'ED' or 'difED' for differenced data (default: 'ED')
        :type algorithm: str
        :return: Tuple containing (complete_results_with_Euclidean_distances, top_K_most_similar_patterns)
        :rtype: (pd.DataFrame, pd.DataFrame)
        :raises ValueError: If an invalid mode is specified
        :raises RuntimeError: If CUDA mode is selected but CUDA is not available
        """
        start_time = time.time()

        if algorithm == "difED":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        if mode == "sequential":
            distances = self.compute_ed_sequential(testing, training)
        elif mode == "parallel":
            distances = self.compute_ed_parallel(testing.to_numpy(), training.to_numpy())
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_ed_cuda(testing, training)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'")

        testing['1NNED'] = distances
        sorted_df = testing.sort_values('1NNED').head(topK)
        self.get_statistics(start_time)

        return testing, sorted_df