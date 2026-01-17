# OneNNMaxNorm (1-Nearest Neighbor Maximum Norm) is an algorithm for computing similarity between time series patterns using Maximum Norm (L-infinity) distance.
#
# This implementation provides three execution modes: sequential CPU processing with NumPy vectorization, parallel multi-threaded CPU processing, and GPU-accelerated CUDA processing.
#
# The algorithm supports both original and differenced time series data.
#
# **Importing this algorithm into a python program**
#
#             from geoanalytics.oneClassClassifiers import OneNNMaxNorm
#
#             training = pd.read_csv('training_data.csv')
#             testing = pd.read_csv('testing_data.csv')
#
#             obj = OneNNMaxNorm()
#
#             results, top_similar = obj.run(training, testing, topK=10, mode='parallel', algorithm='MaxNorm')
#
#             print("Total results with Maximum Norm distances:", len(results))
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


class OneNNMaxNorm:
    """
    **About this algorithm**

    :**Description**: OneNNMaxNorm (1-Nearest Neighbor Maximum Norm) computes similarity between
                      time series patterns using Maximum Norm (L-infinity) distance. For each test pattern,
                      it finds the minimum Maximum Norm distance to any training pattern.
                      Maximum Norm measures the maximum absolute difference along any coordinate axis,
                      making it sensitive to the largest deviation between patterns.

    :**Reference**:  Based on the Maximum Norm (also known as Chebyshev distance or L-infinity norm) from geometry.
                     Useful for applications where the worst-case deviation is critical, such as quality control
                     and outlier detection.

    :**Parameters**:    - **training** (*pd.DataFrame*) -- *Training dataset containing reference patterns.*
                        - **testing** (*pd.DataFrame*) -- *Testing dataset to evaluate for similarity.*
                        - **topK** (*int*) -- *Number of most similar patterns to return (default: -1 returns all).*
                        - **mode** (*str*) -- *Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential').*
                        - **algorithm** (*str**) -- *Algorithm variant: 'MaxNorm' or 'difMaxNorm' for differenced data (default: 'MaxNorm').*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the execution process.*
                        - **endTime** (*float*) -- *To record the completion time of the execution process.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float**) -- *To store the total amount of RSS memory consumed by the program.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.timeSeriesSimilarity import OneNNMaxNorm

            # Load your time series data
            training = pd.read_csv('training_patterns.csv')
            testing = pd.read_csv('testing_patterns.csv')

            # Initialize the similarity finder
            obj = OneNNMaxNorm()

            # Run the algorithm with parallel processing
            results, top_similar = obj.run(training, testing, topK=10, mode='parallel', algorithm='MaxNorm')

            print("Total patterns with Maximum Norm distances:", len(results))
            print("Top 5 most similar patterns:")
            print(top_similar.head(5))

            # Access statistics
            print("Memory consumption in KB:", process.memory_info().rss / 1024)

    **Example Usage**

    .. code-block:: python

            # Example 1: Basic usage with sequential processing
            obj = OneNNMaxNorm()
            results, top_similar = obj.run(training_data, testing_data, topK=5, mode='sequential')

            # Example 2: Using differenced data with parallel processing
            results, top_similar = obj.run(training_data, testing_data, topK=20, mode='parallel', algorithm='difMaxNorm')

            # Example 3: GPU-accelerated processing
            results, top_similar = obj.run(training_data, testing_data, topK=15, mode='cuda')

    **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

    """

    def __init__(self):
        """
        Initialize the OneNNMaxNorm similarity finder.

        This constructor sets up the similarity finder without any required parameters.
        All configuration is passed to the run() method.
        """
        pass

    def getStatistics(self, start_time):
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

    # ----------- Sequential Mode (Optimized) -----------
    def compute_maxnorm_sequential(self, testing, training):
        """
        Compute Maximum Norm distances using sequential CPU processing with NumPy broadcasting.

        :param testing: Testing DataFrame
        :type testing: pd.DataFrame
        :param training: Training DataFrame
        :type training: pd.DataFrame
        :return: List of minimum Maximum Norm distances for each testing sequence
        :rtype: list
        """
        test_np = testing.to_numpy()
        train_np = training.to_numpy()

        if test_np.shape[1] != train_np.shape[1]:
            min_features = min(test_np.shape[1], train_np.shape[1])
            test_np = test_np[:, :min_features]
            train_np = train_np[:, :min_features]

        # Broadcasting: shape (num_test, 1, num_features) - (1, num_train, num_features)
        diff = np.abs(test_np[:, None, :] - train_np)
        max_diff = np.max(diff, axis=2)  # shape: (num_test, num_train)
        distances = np.min(max_diff, axis=1)
        return distances.tolist()

    # ----------- Parallel Mode -----------
    @staticmethod
    @njit(parallel=True)
    def compute_maxnorm_parallel(test_np, train_np):
        """
        Compute Maximum Norm distances using parallel CPU processing with Numba.

        :param test_np: Testing sequences as numpy array
        :type test_np: np.ndarray
        :param train_np: Training sequences as numpy array
        :type train_np: np.ndarray
        :return: Array of minimum Maximum Norm distances for each testing sequence
        :rtype: np.ndarray
        """
        num_test = test_np.shape[0]
        num_train = train_np.shape[0]
        num_features = test_np.shape[1]
        distances = np.empty(num_test)

        for i in prange(num_test):
            min_dist = 1e10
            for j in range(num_train):
                max_diff = 0.0
                for k in range(num_features):
                    diff = abs(test_np[i, k] - train_np[j, k])
                    if diff > max_diff:
                        max_diff = diff
                if max_diff < min_dist:
                    min_dist = max_diff
            distances[i] = min_dist

        return distances

    # ----------- CUDA Mode -----------
    @staticmethod
    @cuda.jit
    def compute_maxnorm_cuda_kernel(testing, training, result):
        """
        CUDA kernel for Maximum Norm distance computation on GPU.

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
            max_diff = 0.0
            for k in range(num_features):
                diff = abs(testing[i][k] - training[j][k])
                if diff > max_diff:
                    max_diff = diff
            if max_diff < min_dist:
                min_dist = max_diff

        result[i] = min_dist

    def compute_maxnorm_cuda(self, testing, training):
        """
        Compute Maximum Norm distances using CUDA GPU acceleration.

        :param testing: Testing DataFrame
        :type testing: pd.DataFrame
        :param training: Training DataFrame
        :type training: pd.DataFrame
        :return: Array of minimum Maximum Norm distances for each testing sequence
        :rtype: np.ndarray
        """
        test_np = testing.to_numpy().astype(np.float32)
        train_np = training.to_numpy().astype(np.float32)

        if test_np.shape[1] != train_np.shape[1]:
            min_features = min(test_np.shape[1], train_np.shape[1])
            test_np = test_np[:, :min_features]
            train_np = train_np[:, :min_features]

        result = cuda.device_array(test_np.shape[0], dtype=np.float32)
        threads_per_block = 128
        blocks_per_grid = (test_np.shape[0] + threads_per_block - 1) // threads_per_block
        self.compute_maxnorm_cuda_kernel[blocks_per_grid, threads_per_block](test_np, train_np, result)

        return result.copy_to_host()

    # ----------- Entry Point -----------
    def run(self, training, testing, topK=-1, mode="sequential", algorithm="MaxNorm"):
        """
        Main execution method for the OneNNMaxNorm algorithm.

        :param training: Training dataset containing reference patterns
        :type training: pd.DataFrame
        :param testing: Testing dataset to evaluate for similarity
        :type testing: pd.DataFrame
        :param topK: Number of most similar patterns to return (default: -1 returns all)
        :type topK: int
        :param mode: Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential')
        :type mode: str
        :param algorithm: Algorithm variant: 'MaxNorm' or 'difMaxNorm' for differenced data (default: 'MaxNorm')
        :type algorithm: str
        :return: Tuple containing (complete_results_with_Maximum_Norm_distances, top_K_most_similar_patterns)
        :rtype: (pd.DataFrame, pd.DataFrame)
        :raises ValueError: If an invalid mode is specified
        :raises RuntimeError: If CUDA mode is selected but CUDA is not available
        """
        start_time = time.time()

        if algorithm == "difMaxNorm":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        if mode == "sequential":
            distances = self.compute_maxnorm_sequential(testing, training)
        elif mode == "parallel":
            distances = self.compute_maxnorm_parallel(testing.to_numpy(), training.to_numpy())
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_maxnorm_cuda(testing, training)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'")

        testing = testing.copy()
        testing["1NNmaxNorm"] = distances
        sorted_df = testing.sort_values("1NNmaxNorm").head(topK)
        self.getStatistics(start_time)
        return testing, sorted_df