# OneNNHausdorff (1-Nearest Neighbor Hausdorff Distance) is an algorithm for computing shape similarity between time series patterns using Hausdorff distance.
#
# This implementation provides three execution modes: sequential CPU processing, parallel multi-threaded CPU processing, and GPU-accelerated CUDA processing.
#
# Hausdorff distance measures the maximum distance from one set to the nearest point in the other set, making it useful for comparing shapes and contours.
#
# **Importing this algorithm into a python program**
#
#             from geoanalytics.oneClassClassifiers import OneNNHausdorff
#
#             training = pd.read_csv('training_data.csv')
#             testing = pd.read_csv('testing_data.csv')
#
#             obj = OneNNHausdorff()
#
#             results, top_similar = obj.run(training, testing, topK=10, mode='parallel', algorithm='Hausdorff')
#
#             print("Total results with Hausdorff distances:", len(results))
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


# ---------------------- Numba Hausdorff (Optimized) ----------------------
@njit
def hausdorff_numba(u, v):
    """
    Compute directional Hausdorff distance from set u to set v.

    :param u: First time series sequence
    :type u: np.ndarray
    :param v: Second time series sequence
    :type v: np.ndarray
    :return: Directional Hausdorff distance from u to v
    :rtype: float
    """
    row = u.shape[0]
    lea_distance = 0.0
    for i in range(row):
        diffs = np.empty(row - 1)
        for k in range(row - 1):
            diffs[k] = abs((u[i] - v[k + 1]) - (u[i] - v[k]))
        distance1 = np.min(diffs)
        if distance1 > lea_distance:
            lea_distance = distance1
    return lea_distance


@njit
def hausdorff_distance(u, v):
    """
    Compute symmetric Hausdorff distance between two sequences.

    :param u: First time series sequence
    :type u: np.ndarray
    :param v: Second time series sequence
    :type v: np.ndarray
    :return: Symmetric Hausdorff distance between sequences
    :rtype: float
    """
    return max(hausdorff_numba(u, v), hausdorff_numba(v, u))


@njit
def compute_hausdorff_sequential(test_np, train_np):
    """
    Compute Hausdorff distances using sequential CPU processing.

    :param test_np: Testing sequences as numpy array
    :type test_np: np.ndarray
    :param train_np: Training sequences as numpy array
    :type train_np: np.ndarray
    :return: Array of minimum Hausdorff distances for each testing sequence
    :rtype: np.ndarray
    """
    num_test = test_np.shape[0]
    num_train = train_np.shape[0]
    distances = np.empty(num_test)
    for i in range(num_test):
        min_dist = 1e10
        for j in range(num_train):
            dist = hausdorff_distance(test_np[i], train_np[j])
            if dist < min_dist:
                min_dist = dist
        distances[i] = min_dist
    return distances


@njit(parallel=True)
def compute_hausdorff_parallel(test_np, train_np):
    """
    Compute Hausdorff distances using parallel CPU processing with Numba.

    :param test_np: Testing sequences as numpy array
    :type test_np: np.ndarray
    :param train_np: Training sequences as numpy array
    :type train_np: np.ndarray
    :return: Array of minimum Hausdorff distances for each testing sequence
    :rtype: np.ndarray
    """
    num_test = test_np.shape[0]
    num_train = train_np.shape[0]
    distances = np.empty(num_test)
    for i in prange(num_test):
        min_dist = 1e10
        for j in range(num_train):
            dist = hausdorff_distance(test_np[i], train_np[j])
            if dist < min_dist:
                min_dist = dist
        distances[i] = min_dist
    return distances


# ---------------------- Class Wrapper ----------------------
class OneNNHausdorff:
    """
    **About this algorithm**

    :**Description**: OneNNHausdorff (1-Nearest Neighbor Hausdorff Distance) computes shape similarity
                      between time series patterns using Hausdorff distance. For each test pattern,
                      it finds the minimum Hausdorff distance to any training pattern.
                      Hausdorff distance measures the maximum distance from one set to the nearest point
                      in the other set, making it particularly useful for comparing shapes and contours.

    :**Reference**:  Huttenlocher, D. P., Klanderman, G. A., & Rucklidge, W. J. (1993). Comparing images using the Hausdorff distance.
                     IEEE Transactions on pattern analysis and machine intelligence, 15(9), 850-863.

    :**Parameters**:    - **training** (*pd.DataFrame*) -- *Training dataset containing reference patterns.*
                        - **testing** (*pd.DataFrame*) -- *Testing dataset to evaluate for similarity.*
                        - **topK** (*int*) -- *Number of most similar patterns to return (default: -1 returns all).*
                        - **mode** (*str*) -- *Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential').*
                        - **algorithm** (*str*) -- *Algorithm variant: 'Hausdorff' or 'difHausdorff' for differenced data (default: 'Hausdorff').*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the execution process.*
                        - **endTime** (*float*) -- *To record the completion time of the execution process.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.shapeSimilarity import OneNNHausdorff

            # Load your time series data
            training = pd.read_csv('training_patterns.csv')
            testing = pd.read_csv('testing_patterns.csv')

            # Initialize the similarity finder
            obj = OneNNHausdorff()

            # Run the algorithm with parallel processing
            results, top_similar = obj.run(training, testing, topK=10, mode='parallel', algorithm='Hausdorff')

            print("Total patterns with Hausdorff distances:", len(results))
            print("Top 5 most similar patterns:")
            print(top_similar.head(5))

            # Access statistics
            print("Memory consumption in KB:", process.memory_info().rss / 1024)

    **Example Usage**

    .. code-block:: python

            # Example 1: Basic usage with sequential processing
            obj = OneNNHausdorff()
            results, top_similar = obj.run(training_data, testing_data, topK=5, mode='sequential')

            # Example 2: Using differenced data with parallel processing
            results, top_similar = obj.run(training_data, testing_data, topK=20, mode='parallel', algorithm='difHausdorff')

            # Example 3: GPU-accelerated processing
            results, top_similar = obj.run(training_data, testing_data, topK=15, mode='cuda')

    **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

    """

    def get_statistics(self, start_time):
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
    @cuda.jit
    def custom_hausdorff_kernel(testing, training, result):
        """
        CUDA kernel for Hausdorff distance computation on GPU.

        :param testing: Testing sequences in GPU memory
        :type testing: cuda.devicearray
        :param training: Training sequences in GPU memory
        :type training: cuda.devicearray
        :param result: Output array for distances in GPU memory
        :type result: cuda.devicearray
        """
        test_idx = cuda.grid(1)
        if test_idx >= testing.shape[0]:
            return

        num_features = testing.shape[1]
        min_total_dist = 1e10

        for train_idx in range(training.shape[0]):
            max_min1 = 0.0
            for i in range(num_features):
                min_diff = 1e10
                for k in range(num_features - 1):
                    diff = abs((testing[test_idx][i] - training[train_idx][k + 1]) -
                               (testing[test_idx][i] - training[train_idx][k]))
                    if diff < min_diff:
                        min_diff = diff
                if min_diff > max_min1:
                    max_min1 = min_diff

            max_min2 = 0.0
            for i in range(num_features):
                min_diff = 1e10
                for k in range(num_features - 1):
                    diff = abs((training[train_idx][i] - testing[test_idx][k + 1]) -
                               (training[train_idx][i] - testing[test_idx][k]))
                    if diff < min_diff:
                        min_diff = diff
                if min_diff > max_min2:
                    max_min2 = min_diff

            final_dist = max(max_min1, max_min2)
            if final_dist < min_total_dist:
                min_total_dist = final_dist

        result[test_idx] = min_total_dist

    def compute_hausdorff_cuda(self, testing, training):
        """
        Compute Hausdorff distances using CUDA GPU acceleration.

        :param testing: Testing DataFrame
        :type testing: pd.DataFrame
        :param training: Training DataFrame
        :type training: pd.DataFrame
        :return: Array of minimum Hausdorff distances for each testing sequence
        :rtype: np.ndarray
        """
        test_np = testing.to_numpy().astype(np.float64)
        train_np = training.to_numpy().astype(np.float64)
        result = cuda.device_array(test_np.shape[0], dtype=np.float64)

        threads_per_block = 128
        blocks_per_grid = (test_np.shape[0] + threads_per_block - 1) // threads_per_block
        self.custom_hausdorff_kernel[blocks_per_grid, threads_per_block](test_np, train_np, result)

        return result.copy_to_host()

    def run(self, training, testing, topK=-1, mode="sequential", algorithm='Hausdorff'):
        """
        Main execution method for the OneNNHausdorff algorithm.

        :param training: Training dataset containing reference patterns
        :type training: pd.DataFrame
        :param testing: Testing dataset to evaluate for similarity
        :type testing: pd.DataFrame
        :param topK: Number of most similar patterns to return (default: -1 returns all)
        :type topK: int
        :param mode: Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential')
        :type mode: str
        :param algorithm: Algorithm variant: 'Hausdorff' or 'difHausdorff' for differenced data (default: 'Hausdorff')
        :type algorithm: str
        :return: Tuple containing (complete_results_with_Hausdorff_distances, top_K_most_similar_patterns)
        :rtype: (pd.DataFrame, pd.DataFrame)
        :raises ValueError: If an invalid mode is specified
        :raises RuntimeError: If CUDA mode is selected but CUDA is not available
        """
        start_time = time.time()

        if algorithm == "difHausdorff":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        test_np = testing.to_numpy()
        train_np = training.to_numpy()

        if mode == "sequential":
            distances = compute_hausdorff_sequential(test_np, train_np)
        elif mode == "parallel":
            distances = compute_hausdorff_parallel(test_np, train_np)
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_hausdorff_cuda(testing, training)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'")

        testing['1NNHausdorff'] = distances
        sorted_df = testing.sort_values('1NNHausdorff').head(topK)
        self.get_statistics(start_time)
        return testing, sorted_df