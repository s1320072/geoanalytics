# Correlation Distance Similarity Finder is an algorithm for computing shape similarity  between time series patterns using correlation distance (1 - Pearson correlation).
#
# This implementation provides three execution modes: sequential CPU processing, parallel multi-threaded CPU processing, and GPU-accelerated CUDA processing.
#
# The algorithm normalizes data using z-score normalization and finds the minimum correlation distance between each test pattern and all training patterns.
#
# **Importing this algorithm into a python program**
#
#             from geoanalytics.oneClassClassifiers import Correlation
#
#             training = pd.read_csv('training_data.csv')
#             testing = pd.read_csv('testing_data.csv')
#
#             obj = Correlation()
#
#             results, top_similar = obj.run(training, testing, topK=10, mode='parallel')
#
#             print("Total results with distances:", len(results))
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
from tqdm import tqdm
from numba import njit, prange, cuda
import numpy as np
import time
import gc
import os
import psutil


class Correlation:
    """
    **About this algorithm**

    :**Description**: Correlation Distance Similarity Finder computes shape similarity between time series patterns
                      using correlation distance (1 - Pearson correlation coefficient). For each test pattern,
                      it finds the minimum distance to any training pattern after z-score normalization.
                      The algorithm measures shape similarity independent of amplitude and offset differences.

    :**Reference**:  Based on Pearson correlation coefficient for measuring linear relationship between variables.
                     Commonly used in time series analysis and pattern recognition.

    :**Parameters**:    - **training** (*pd.DataFrame*) -- *Training dataset containing reference patterns.*
                        - **testing** (*pd.DataFrame*) -- *Testing dataset to evaluate for similarity.*
                        - **topK** (*int*) -- *Number of most similar patterns to return (default: 10).*
                        - **mode** (*str*) -- *Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential').*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the execution process.*
                        - **endTime** (*float*) -- *To record the completion time of the execution process.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.shapeSimilarity import Correlation

            # Load your time series data
            training = pd.read_csv('training_patterns.csv')
            testing = pd.read_csv('testing_patterns.csv')

            # Initialize the similarity finder
            obj = Correlation()

            # Run the algorithm with parallel processing
            results, top_similar = obj.run(training, testing, topK=10, mode='parallel')

            print("Total patterns with distances:", len(results))
            print("Top 5 most similar patterns:")
            print(top_similar.head(5))

            # Access statistics
            print("Memory consumption in KB:", process.memory_info().rss / 1024)

    **Example Usage**

    .. code-block:: python

            # Example 1: Basic usage with sequential processing
            obj = Correlation()
            results, top_similar = obj.run(training_data, testing_data, topK=5, mode='sequential')

            # Example 2: Using parallel processing for larger datasets
            results, top_similar = obj.run(training_data, testing_data, topK=20, mode='parallel')

            # Example 3: GPU-accelerated processing
            results, top_similar = obj.run(training_data, testing_data, topK=15, mode='cuda')

    **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

    """

    def __init__(self):
        """
        Initialize the Correlation similarity finder.

        This constructor sets up the similarity finder without any required parameters.
        All configuration is passed to the run() method.
        """
        pass

    def getStatistics(self, startTime):
        """
        Display execution statistics after algorithm completion.

        :param startTime: Start time of the execution (from time.time())
        :type startTime: float

        Prints execution time and memory consumption in KB.
        """
        print("Total Execution time:", time.time() - startTime)
        process = psutil.Process(os.getpid())
        memory = process.memory_full_info().uss / 1024
        print("Memory in KB:", memory)

    @staticmethod
    @njit
    def _normalize_row(row):
        """
        Normalize a single row using z-score normalization.

        :param row: Input row of data
        :type row: np.ndarray
        :return: Z-score normalized row
        :rtype: np.ndarray
        """
        eps = 1e-10
        mean_val = np.mean(row)
        std_val = np.std(row) + eps
        if std_val == eps:
            return np.zeros_like(row)
        return (row - mean_val) / std_val

    @staticmethod
    @njit
    def _correlation_distance(row1, row2):
        """
        Compute correlation distance (1 - Pearson correlation).

        :param row1: First normalized row
        :type row1: np.ndarray
        :param row2: Second normalized row
        :type row2: np.ndarray
        :return: Correlation distance (0-2 range, where 0 means perfect correlation)
        :rtype: float
        """
        n = len(row1)
        mean1, mean2 = np.mean(row1), np.mean(row2)
        std1, std2 = np.std(row1), np.std(row2)

        if std1 < 1e-10 or std2 < 1e-10:
            return 1.0

        cov = 0.0
        for i in range(n):
            cov += (row1[i] - mean1) * (row2[i] - mean2)

        corr = cov / (n * std1 * std2)
        return 1.0 - corr

    @staticmethod
    def compute_similarity_sequential(testing_array, train_normalized):
        """
        Compute similarity distances using sequential CPU processing.

        :param testing_array: Testing data as numpy array
        :type testing_array: np.ndarray
        :param train_normalized: Pre-normalized training data
        :type train_normalized: np.ndarray
        :return: Array of minimum distances for each testing instance
        :rtype: np.ndarray
        """
        num_rows = testing_array.shape[0]
        distances = np.zeros(num_rows, dtype=np.float64)
        num_train = train_normalized.shape[0]

        for i in tqdm(range(num_rows)):
            test_row = testing_array[i]
            test_norm = Correlation._normalize_row(test_row)
            min_dist = np.inf

            for j in range(num_train):
                dist = Correlation._correlation_distance(test_norm, train_normalized[j])
                if dist < min_dist:
                    min_dist = dist

            distances[i] = min_dist

        return distances

    @staticmethod
    @njit(parallel=True)
    def compute_similarity_parallel(testing_array, train_normalized):
        """
        Compute similarity distances using parallel CPU processing with Numba.

        :param testing_array: Testing data as numpy array
        :type testing_array: np.ndarray
        :param train_normalized: Pre-normalized training data
        :type train_normalized: np.ndarray
        :return: Array of minimum distances for each testing instance
        :rtype: np.ndarray
        """
        num_rows = testing_array.shape[0]
        distances = np.zeros(num_rows)
        num_train = train_normalized.shape[0]

        for i in prange(num_rows):
            test_row = testing_array[i]

            # Normalize current test row
            eps = 1e-10
            mean_val = 0.0
            for k in range(len(test_row)):
                mean_val += test_row[k]
            mean_val /= len(test_row)

            var_val = 0.0
            for k in range(len(test_row)):
                diff = test_row[k] - mean_val
                var_val += diff * diff
            std_val = np.sqrt(var_val / len(test_row)) + eps

            test_norm = np.empty_like(test_row)
            if std_val == eps:
                for k in range(len(test_norm)):
                    test_norm[k] = 0.0
            else:
                for k in range(len(test_norm)):
                    test_norm[k] = (test_row[k] - mean_val) / std_val

            min_dist = 1.0

            # Compare with all training rows
            for j in range(num_train):
                train_row = train_normalized[j]

                # Compute correlation
                mean1, mean2 = 0.0, 0.0
                for k in range(len(test_norm)):
                    mean1 += test_norm[k]
                    mean2 += train_row[k]
                mean1 /= len(test_norm)
                mean2 /= len(test_norm)

                var1, var2 = 0.0, 0.0
                cov = 0.0
                for k in range(len(test_norm)):
                    diff1 = test_norm[k] - mean1
                    diff2 = train_row[k] - mean2
                    var1 += diff1 * diff1
                    var2 += diff2 * diff2
                    cov += diff1 * diff2

                std1 = np.sqrt(var1 / len(test_norm))
                std2 = np.sqrt(var2 / len(test_norm))

                if std1 < eps or std2 < eps:
                    dist = 1.0
                else:
                    corr = cov / (len(test_norm) * std1 * std2)
                    dist = 1.0 - corr

                if dist < min_dist:
                    min_dist = dist

            distances[i] = min_dist

        return distances

    @staticmethod
    @cuda.jit
    def compute_similarity_kernel(testing_array, train_normalized, distances, num_train):
        """
        CUDA kernel for similarity computation on GPU.

        :param testing_array: Testing data in GPU memory
        :type testing_array: cuda.devicearray
        :param train_normalized: Training data in GPU memory
        :type train_normalized: cuda.devicearray
        :param distances: Output array for distances in GPU memory
        :type distances: cuda.devicearray
        :param num_train: Number of training samples
        :type num_train: int
        """
        i = cuda.grid(1)
        if i < testing_array.shape[0]:
            eps = 1e-10

            # Normalize test row
            test_row = testing_array[i]

            # Calculate mean
            mean_val = 0.0
            for k in range(len(test_row)):
                mean_val += test_row[k]
            mean_val /= len(test_row)

            # Calculate std
            var_val = 0.0
            for k in range(len(test_row)):
                diff = test_row[k] - mean_val
                var_val += diff * diff
            std_val = (var_val / len(test_row)) ** 0.5 + eps

            # Normalize
            test_norm = cuda.local.array(8, dtype=np.float32)
            if std_val == eps:
                for k in range(len(test_row)):
                    test_norm[k] = 0.0
            else:
                for k in range(len(test_row)):
                    test_norm[k] = (test_row[k] - mean_val) / std_val

            # Find minimum distance
            min_dist = 1.0

            for j in range(num_train):
                train_row = train_normalized[j]

                # Compute means
                mean1, mean2 = 0.0, 0.0
                for k in range(len(test_norm)):
                    mean1 += test_norm[k]
                    mean2 += train_row[k]
                mean1 /= len(test_norm)
                mean2 /= len(test_norm)

                # Compute covariance and variances
                var1, var2 = 0.0, 0.0
                cov = 0.0
                for k in range(len(test_norm)):
                    diff1 = test_norm[k] - mean1
                    diff2 = train_row[k] - mean2
                    var1 += diff1 * diff1
                    var2 += diff2 * diff2
                    cov += diff1 * diff2

                std1 = (var1 / len(test_norm)) ** 0.5
                std2 = (var2 / len(test_norm)) ** 0.5

                if std1 < eps or std2 < eps:
                    dist = 1.0
                else:
                    corr = cov / (len(test_norm) * std1 * std2)
                    dist = 1.0 - corr

                if dist < min_dist:
                    min_dist = dist

            distances[i] = min_dist

    def compute_similarity_cuda(self, testing_array, train_normalized):
        """
        Compute similarity distances using CUDA GPU acceleration.

        :param testing_array: Testing data as numpy array
        :type testing_array: np.ndarray
        :param train_normalized: Pre-normalized training data
        :type train_normalized: np.ndarray
        :return: Array of minimum distances for each testing instance
        :rtype: np.ndarray
        :raises RuntimeError: If CUDA is not available on the system
        """
        num_rows = testing_array.shape[0]
        num_train = train_normalized.shape[0]

        # Convert to float32 for GPU (better performance)
        testing_array_f32 = testing_array.astype(np.float32)
        train_normalized_f32 = train_normalized.astype(np.float32)

        # Allocate GPU memory
        d_testing = cuda.to_device(testing_array_f32)
        d_train = cuda.to_device(train_normalized_f32)
        d_distances = cuda.device_array(num_rows, dtype=np.float32)

        # Kernel configuration
        threads_per_block = 256
        blocks_per_grid = (num_rows + threads_per_block - 1) // threads_per_block

        # Launch kernel
        self.compute_similarity_kernel[blocks_per_grid, threads_per_block](
            d_testing, d_train, d_distances, num_train
        )

        return d_distances.copy_to_host()

    def run(self, training, testing, topK=10, mode="sequential"):
        """
        Main execution method for the Correlation similarity finder.

        :param training: Training dataset containing reference patterns
        :type training: pd.DataFrame
        :param testing: Testing dataset to evaluate for similarity
        :type testing: pd.DataFrame
        :param topK: Number of most similar patterns to return (default: 10)
        :type topK: int
        :param mode: Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential')
        :type mode: str
        :return: Tuple containing (complete_results_with_distances, top_K_most_similar_patterns)
        :rtype: (pd.DataFrame, pd.DataFrame)
        :raises ValueError: If an invalid mode is specified
        :raises RuntimeError: If CUDA mode is selected but CUDA is not available
        """
        start_time = time.time()

        # Convert to numpy arrays with consistent dtype
        train_array = training.values.astype(np.float64)
        test_array = testing.values.astype(np.float64)

        # Normalize training data (once) - using list comprehension
        train_normalized = np.array([self._normalize_row(row) for row in train_array], dtype=np.float64)

        # Clean up
        del train_array
        gc.collect()

        # Compute distances based on mode
        if mode == "sequential":
            distances = self.compute_similarity_sequential(test_array, train_normalized)
        elif mode == "parallel":
            distances = self.compute_similarity_parallel(test_array, train_normalized)
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            distances = self.compute_similarity_cuda(test_array, train_normalized)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'.")

        # Add distances to testing DataFrame
        testing = testing.copy()  # Avoid modifying original
        testing['Distance'] = distances

        # Get topK rows
        sorted_df = testing.sort_values('Distance').head(topK)

        # Statistics
        self.getStatistics(start_time)

        return testing, sorted_df