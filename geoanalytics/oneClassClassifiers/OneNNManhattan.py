# OneNNManhattan (1-Nearest Neighbor Manhattan Distance) is an algorithm for computing similarity between time series patterns using Manhattan (L1) distance.
#
# This implementation provides three execution modes: sequential CPU processing with NumPy vectorization, parallel multi-threaded CPU processing, and GPU-accelerated CUDA processing.
#
# The algorithm supports both original and differenced time series data.
#
# **Importing this algorithm into a python program**
#
#             from geoanalytics.oneClassClassifiers import OneNNManhattan
#
#             training = pd.read_csv('training_data.csv')
#             testing = pd.read_csv('testing_data.csv')
#
#             obj = OneNNManhattan()
#
#             results, top_similar = obj.run(training, testing, topK=10, mode='parallel', algorithm='Manhattan')
#
#             print("Total results with Manhattan distances:", len(results))
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
import psutil
from tqdm import tqdm
from numba import njit, prange, cuda


class OneNNManhattan:
    """
    **About this algorithm**

    :**Description**: OneNNManhattan (1-Nearest Neighbor Manhattan Distance) computes similarity between
                      time series patterns using Manhattan (L1) distance. For each test pattern,
                      it finds the minimum Manhattan distance to any training pattern.
                      Manhattan distance measures the sum of absolute differences along coordinate axes,
                      making it suitable for high-dimensional data and sparse patterns.

    :**Reference**:  Based on the Manhattan distance (also known as taxicab or L1 distance) from geometry.
                     Commonly used in pattern recognition, computer vision, and data mining applications.

    :**Parameters**:    - **training** (*pd.DataFrame*) -- *Training dataset containing reference patterns.*
                        - **testing** (*pd.DataFrame*) -- *Testing dataset to evaluate for similarity.*
                        - **topK** (*int*) -- *Number of most similar patterns to return (default: -1 returns all).*
                        - **mode** (*str*) -- *Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential').*
                        - **algorithm** (*str*) -- *Algorithm variant: 'Manhattan' or 'difManhattan' for differenced data (default: 'Manhattan').*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the execution process.*
                        - **endTime** (*float*) -- *To record the completion time of the execution process.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float**) -- *To store the total amount of RSS memory consumed by the program.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.timeSeriesSimilarity import OneNNManhattan

            # Load your time series data
            training = pd.read_csv('training_patterns.csv')
            testing = pd.read_csv('testing_patterns.csv')

            # Initialize the similarity finder
            obj = OneNNManhattan()

            # Run the algorithm with parallel processing
            results, top_similar = obj.run(training, testing, topK=10, mode='parallel', algorithm='Manhattan')

            print("Total patterns with Manhattan distances:", len(results))
            print("Top 5 most similar patterns:")
            print(top_similar.head(5))

            # Access statistics
            print("Memory consumption in KB:", process.memory_info().rss / 1024)

    **Example Usage**

    .. code-block:: python

            # Example 1: Basic usage with sequential processing
            obj = OneNNManhattan()
            results, top_similar = obj.run(training_data, testing_data, topK=5, mode='sequential')

            # Example 2: Using differenced data with parallel processing
            results, top_similar = obj.run(training_data, testing_data, topK=20, mode='parallel', algorithm='difManhattan')

            # Example 3: GPU-accelerated processing
            results, top_similar = obj.run(training_data, testing_data, topK=15, mode='cuda')

    **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

    """

    def __init__(self):
        """
        Initialize the OneNNManhattan similarity finder.

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
        process = psutil.Process()
        memory_kb = process.memory_full_info().uss / 1024
        print("Memory Usage (KB):", memory_kb)

    @staticmethod
    def compute_manhattan_sequential(testing, training):
        """
        Compute Manhattan distances using sequential CPU processing with NumPy vectorization.

        :param testing: Testing DataFrame
        :type testing: pd.DataFrame
        :param training: Training DataFrame
        :type training: pd.DataFrame
        :return: List of minimum Manhattan distances for each testing sequence
        :rtype: list
        """
        test_np = testing.to_numpy()
        train_np = training.to_numpy()

        # Fix feature size mismatch
        if test_np.shape[1] != train_np.shape[1]:
            min_features = min(test_np.shape[1], train_np.shape[1])
            test_np = test_np[:, :min_features]
            train_np = train_np[:, :min_features]

        distances = np.min(np.sum(np.abs(test_np[:, None] - train_np), axis=2), axis=1)
        return distances.tolist()

    @staticmethod
    @njit(parallel=True)
    def compute_manhattan_parallel(test_np, train_np):
        """
        Compute Manhattan distances using parallel CPU processing with Numba.

        :param test_np: Testing sequences as numpy array
        :type test_np: np.ndarray
        :param train_np: Training sequences as numpy array
        :type train_np: np.ndarray
        :return: Array of minimum Manhattan distances for each testing sequence
        :rtype: np.ndarray
        """
        num_test = test_np.shape[0]
        num_train = train_np.shape[0]
        distances = np.empty(num_test)

        for i in prange(num_test):
            min_dist = 1e10
            for j in range(num_train):
                dist = np.sum(np.abs(test_np[i] - train_np[j]))
                if dist < min_dist:
                    min_dist = dist
            distances[i] = min_dist
        return distances

    @staticmethod
    @cuda.jit
    def compute_manhattan_cuda_kernel(testing, training, result):
        """
        CUDA kernel for Manhattan distance computation on GPU.

        :param testing: Testing sequences in GPU memory
        :type testing: cuda.devicearray
        :param training: Training sequences in GPU memory
        :type training: cuda.devicearray
        :param result: Output array for distances in GPU memory
        :type result: cuda.devicearray
        """
        i = cuda.grid(1)
        if i >= testing.shape[0]:
            return

        num_features = testing.shape[1]
        min_dist = 1e10
        for j in range(training.shape[0]):
            dist = 0.0
            for k in range(num_features):
                dist += abs(testing[i][k] - training[j][k])
            if dist < min_dist:
                min_dist = dist
        result[i] = min_dist

    @staticmethod
    def compute_manhattan_cuda(testing, training):
        """
        Compute Manhattan distances using CUDA GPU acceleration.

        :param testing: Testing DataFrame
        :type testing: pd.DataFrame
        :param training: Training DataFrame
        :type training: pd.DataFrame
        :return: Array of minimum Manhattan distances for each testing sequence
        :rtype: np.ndarray
        """
        test_np = testing.to_numpy().astype(np.float32)
        train_np = training.to_numpy().astype(np.float32)

        # Fix feature size mismatch
        if test_np.shape[1] != train_np.shape[1]:
            min_features = min(test_np.shape[1], train_np.shape[1])
            test_np = test_np[:, :min_features]
            train_np = train_np[:, :min_features]

        result = cuda.device_array(test_np.shape[0], dtype=np.float32)
        threads_per_block = 128
        blocks_per_grid = (test_np.shape[0] + threads_per_block - 1) // threads_per_block
        OneNNManhattan.compute_manhattan_cuda_kernel[blocks_per_grid, threads_per_block](test_np, train_np,
                                                                                         result)
        return result.copy_to_host()

    def run(self, training, testing, topK=-1, mode="sequential", algorithm="Manhattan"):
        """
        Main execution method for the OneNNManhattan algorithm.

        :param training: Training dataset containing reference patterns
        :type training: pd.DataFrame
        :param testing: Testing dataset to evaluate for similarity
        :type testing: pd.DataFrame
        :param topK: Number of most similar patterns to return (default: -1 returns all)
        :type topK: int
        :param mode: Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential')
        :type mode: str
        :param algorithm: Algorithm variant: 'Manhattan' or 'difManhattan' for differenced data (default: 'Manhattan')
        :type algorithm: str
        :return: Tuple containing (complete_results_with_Manhattan_distances, top_K_most_similar_patterns)
        :rtype: (pd.DataFrame, pd.DataFrame)
        :raises ValueError: If an invalid mode is specified
        :raises RuntimeError: If CUDA mode is selected but CUDA is not available
        """
        start_time = time.time()

        if algorithm == "difManhattan":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        if mode == "sequential":
            distances = self.compute_manhattan_sequential(testing, training)
        elif mode == "parallel":
            distances = self.compute_manhattan_parallel(testing.to_numpy(), training.to_numpy())
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_manhattan_cuda(testing, training)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'")

        testing['1NNManhattan'] = distances
        sorted_df = testing.sort_values('1NNManhattan').head(topK)
        self.get_statistics(start_time)
        return testing, sorted_df