# FuzzyTSC (Fuzzy Time Series Classifier) is an algorithm for computing relative deviation (RD) scores to identify anomalous patterns in time series data. 
#
# This implementation provides three execution modes: sequential CPU processing, parallel multi-threaded CPU processing, and GPU-accelerated CUDA processing.
#
# The algorithm supports both original and differenced time series data with optional Z-score normalization.
#
# **Importing this algorithm into a python program**
#
#             from geoanalytics.oneClassClassifiers import FuzzyTSC
#
#             training = pd.read_csv('training_data.csv')
#             testing = pd.read_csv('testing_data.csv')
#
#             obj = FuzzyTSC()
#
#             results, top_results = obj.run(training, testing, topK=100, mode='parallel', algorithm='FuzzyTSC', apply_zscore=True)
#
#             print("Total results with RD scores:", len(results))
#             print("Top anomalous patterns:", top_results.head())
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
import sys


class FuzzyTSC:
    """
    **About this algorithm**

    :**Description**: FuzzyTSC (Fuzzy Time Series Classifier) computes relative deviation (RD) scores to identify
                      anomalous patterns in time series data. The algorithm measures how far each data point deviates
                      from the training distribution using fuzzy logic principles. It provides three execution modes:
                      sequential CPU, parallel multi-threaded CPU, and GPU-accelerated CUDA processing.

    :**Reference**:  Time series anomaly detection using fuzzy logic and statistical methods.
                     Based on principles of fuzzy set theory applied to time series classification.

    :**Parameters**:    - **training** (*pd.DataFrame*) -- *Training dataset for establishing baseline statistics.*
                        - **testing** (*pd.DataFrame*) -- *Testing dataset to evaluate for anomalies.*
                        - **topK** (*int*) -- *Number of top anomalous patterns to return (default: -1 returns all).*
                        - **mode** (*str*) -- *Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential').*
                        - **algorithm** (*str*) -- *Algorithm variant: 'FuzzyTSC' or 'difFuzzyTSC' for differenced data (default: 'FuzzyTSC').*
                        - **apply_zscore** (*bool*) -- *Whether to apply Z-score normalization (default: True).*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the execution process.*
                        - **endTime** (*float*) -- *To record the completion time of the execution process.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.oneClassClassifiers import FuzzyTSC

            # Load your time series data
            training = pd.read_csv('training_data.csv')
            testing = pd.read_csv('testing_data.csv')

            # Initialize the classifier
            obj = FuzzyTSC()

            # Run the algorithm with parallel processing
            results, top_results = obj.run(training, testing, topK=100, mode='parallel', algorithm='FuzzyTSC', apply_zscore=True)

            print("Total results with RD scores:", len(results))
            print("Top 10 anomalous patterns:")
            print(top_results.head(10))

            # Access statistics
            print("Memory consumption in KB:", process.memory_info().rss / 1024)

    **Example Usage**

    .. code-block:: python

            # Example 1: Basic usage with sequential processing
            obj = FuzzyTSC()
            results, topK = obj.run(training_data, testing_data, topK=50, mode='sequential')

            # Example 2: Using differenced data with parallel processing
            results, topK = obj.run(training_data, testing_data, topK=100, mode='parallel', algorithm='difFuzzyTSC')

            # Example 3: GPU-accelerated processing without normalization
            results, topK = obj.run(training_data, testing_data, mode='cuda', apply_zscore=False)

    **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

    """

    def __init__(self):
        """
        Initialize the FuzzyTSC classifier.

        This constructor sets up the classifier without any required parameters.
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
        print("Total Execution time of proposedAlgo:", time.time() - startTime)
        process = psutil.Process(os.getpid())
        memory = process.memory_full_info().uss
        memory_in_KB = memory / 1024
        print("Memory of proposedAlgo in KB:", memory_in_KB)

    @staticmethod
    def compute_rd_sequential(testing_array, mean_data, min_data, max_data, num_columns):
        """
        Compute RD scores for sequential-threaded execution.

        :param testing_array: Testing data as numpy array
        :type testing_array: np.ndarray
        :param mean_data: Mean values for each feature from training data
        :type mean_data: np.ndarray
        :param min_data: Minimum values for each feature from training data
        :type min_data: np.ndarray
        :param max_data: Maximum values for each feature from training data
        :type max_data: np.ndarray
        :param num_columns: Number of columns/features in the data
        :type num_columns: int
        :return: Array of RD scores for each testing instance
        :rtype: np.ndarray
        """
        num_rows = testing_array.shape[0]
        rd_scores = np.zeros(num_rows)

        for i in tqdm(range(num_rows)):
            counter = 0.0
            for j in range(1, num_columns):  # Skip the first column
                value = testing_array[i, j]
                if value >= mean_data[j]:
                    if value <= max_data[j]:
                        counter += 0.5 * (value - mean_data[j]) / (max_data[j] - mean_data[j])
                    else:
                        counter += 1
                else:
                    if value >= min_data[j]:
                        counter += 0.5 * (mean_data[j] - value) / (mean_data[j] - min_data[j])
                    else:
                        counter += 1
            rd_scores[i] = counter / num_columns

        return rd_scores

    @staticmethod
    @njit(parallel=True)
    def compute_rd_parallel(testing_array, mean_data, min_data, max_data, num_columns):
        """
        Compute RD scores using multi-threaded parallel processing with Numba.

        :param testing_array: Testing data as numpy array
        :type testing_array: np.ndarray
        :param mean_data: Mean values for each feature from training data
        :type mean_data: np.ndarray
        :param min_data: Minimum values for each feature from training data
        :type min_data: np.ndarray
        :param max_data: Maximum values for each feature from training data
        :type max_data: np.ndarray
        :param num_columns: Number of columns/features in the data
        :type num_columns: int
        :return: Array of RD scores for each testing instance
        :rtype: np.ndarray
        """
        num_rows = testing_array.shape[0]
        rd_scores = np.zeros(num_rows)

        for i in prange(num_rows):
            counter = 0.0
            for j in range(1, num_columns):  # Skip the first column
                value = testing_array[i, j]
                if value >= mean_data[j]:
                    if value <= max_data[j]:
                        counter += 0.5 * (value - mean_data[j]) / (max_data[j] - mean_data[j])
                    else:
                        counter += 1
                else:
                    if value >= min_data[j]:
                        counter += 0.5 * (mean_data[j] - value) / (mean_data[j] - min_data[j])
                    else:
                        counter += 1
            rd_scores[i] = counter / num_columns

        return rd_scores

    @staticmethod
    @cuda.jit
    def compute_rd_kernel(testing_array, mean_data, min_data, max_data, rd_scores, num_columns):
        """
        CUDA kernel for RD score computation on GPU.

        :param testing_array: Testing data in GPU memory
        :type testing_array: cuda.devicearray
        :param mean_data: Mean values in GPU memory
        :type mean_data: cuda.devicearray
        :param min_data: Minimum values in GPU memory
        :type min_data: cuda.devicearray
        :param max_data: Maximum values in GPU memory
        :type max_data: cuda.devicearray
        :param rd_scores: Output array for RD scores in GPU memory
        :type rd_scores: cuda.devicearray
        :param num_columns: Number of columns/features in the data
        :type num_columns: int
        """
        i = cuda.grid(1)
        if i < testing_array.shape[0]:  # Ensure index is within bounds
            counter = 0.0
            for j in range(1, num_columns):  # Skip the first column
                value = testing_array[i, j]
                if value >= mean_data[j]:
                    if value <= max_data[j]:
                        counter += 0.5 * (value - mean_data[j]) / (max_data[j] - mean_data[j])
                    else:
                        counter += 1
                else:
                    if value >= min_data[j]:
                        counter += 0.5 * (mean_data[j] - value) / (mean_data[j] - min_data[j])
                    else:
                        counter += 1
            rd_scores[i] = counter / num_columns

    def compute_rd_cuda(self, testing_array, mean_data, min_data, max_data, num_columns):
        """
        Compute RD scores using CUDA for GPU acceleration.

        :param testing_array: Testing data as numpy array
        :type testing_array: np.ndarray
        :param mean_data: Mean values for each feature from training data
        :type mean_data: np.ndarray
        :param min_data: Minimum values for each feature from training data
        :type min_data: np.ndarray
        :param max_data: Maximum values for each feature from training data
        :type max_data: np.ndarray
        :param num_columns: Number of columns/features in the data
        :type num_columns: int
        :return: Array of RD scores for each testing instance
        :rtype: np.ndarray
        :raises RuntimeError: If CUDA is not available on the system
        """
        num_rows = testing_array.shape[0]

        # Allocate GPU memory
        d_testing_array = cuda.to_device(testing_array)
        d_mean_data = cuda.to_device(mean_data)
        d_min_data = cuda.to_device(min_data)
        d_max_data = cuda.to_device(max_data)
        d_rd_scores = cuda.device_array(num_rows, dtype=np.float32)

        # Define thread and block configuration
        threads_per_block = 256
        blocks_per_grid = (num_rows + (threads_per_block - 1)) // threads_per_block

        # Launch CUDA kernel
        self.compute_rd_kernel[blocks_per_grid, threads_per_block](
            d_testing_array, d_mean_data, d_min_data, d_max_data, d_rd_scores, num_columns
        )

        # Copy RD scores back to host
        return d_rd_scores.copy_to_host()

    def run(self, training, testing, topK=-1, mode="sequential", algorithm="FuzzyTSC", apply_zscore=True):
        """
        Main execution method for the FuzzyTSC algorithm.

        :param training: Training dataset for establishing baseline statistics
        :type training: pd.DataFrame
        :param testing: Testing dataset to evaluate for anomalies
        :type testing: pd.DataFrame
        :param topK: Number of top anomalous patterns to return (default: -1 returns all)
        :type topK: int
        :param mode: Execution mode: 'sequential', 'parallel', or 'cuda' (default: 'sequential')
        :type mode: str
        :param algorithm: Algorithm variant: 'FuzzyTSC' or 'difFuzzyTSC' for differenced data (default: 'FuzzyTSC')
        :type algorithm: str
        :param apply_zscore: Whether to apply Z-score normalization (default: True)
        :type apply_zscore: bool
        :return: Tuple containing (complete_results_with_RD_scores, top_K_anomalous_patterns)
        :rtype: (pd.DataFrame, pd.DataFrame)
        :raises ValueError: If an invalid mode is specified
        :raises RuntimeError: If CUDA mode is selected but CUDA is not available
        """
        start_time = time.time()

        # Store original testing data BEFORE any modification
        original_testing = testing.copy()

        if algorithm == "difFuzzyTSC":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]
            original_testing = original_testing.diff(axis=1).iloc[:, 1:]

        # Store normalized versions if needed
        training_normalized = training.copy()
        testing_normalized = testing.copy()

        # Apply Z-score normalization if requested
        if apply_zscore:
            # Apply Z-score normalization to training data
            training_means = training.mean(axis=1)
            training_stds = training.std(axis=1) + 1e-10
            training_normalized = training.sub(training_means, axis=0).div(training_stds, axis=0)

            # Apply Z-score normalization to testing data
            testing_means = testing.mean(axis=1)
            testing_stds = testing.std(axis=1) + 1e-10
            testing_normalized = testing.sub(testing_means, axis=0).div(testing_stds, axis=0)

            # Use normalized data for calculations
            training_for_stats = training_normalized
            testing_for_calc = testing_normalized
        else:
            # Use original data for calculations
            training_for_stats = training
            testing_for_calc = testing

        # Calculate max, mean, and min values for each feature in the training dataset
        max_data = training_for_stats.max(axis=0).values
        mean_data = training_for_stats.mean(axis=0).values
        min_data = training_for_stats.min(axis=0).values

        # Clean up training data
        del training, training_for_stats, training_normalized
        gc.collect()

        # Convert testing dataset to numpy array for compatibility with Numba and CUDA
        testing_array = testing_for_calc.to_numpy()
        num_rows, num_columns = testing_array.shape

        if mode == "sequential":
            rd_scores = self.compute_rd_sequential(testing_array, mean_data, min_data, max_data, num_columns)
        elif mode == "parallel":
            rd_scores = self.compute_rd_parallel(testing_array, mean_data, min_data, max_data, num_columns)
        elif mode == "cuda":
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            rd_scores = self.compute_rd_cuda(testing_array, mean_data, min_data, max_data, num_columns)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', 'parallel', or 'cuda'.")

        # Clean up
        del testing_array, testing_for_calc, testing_normalized
        gc.collect()

        # Add RD scores to the ORIGINAL testing DataFrame
        original_testing['RD'] = rd_scores

        # Retrieve top elements based on RD scores
        sorted_df = original_testing.sort_values('RD').head(topK)

        # Log statistics
        self.getStatistics(start_time)

        # Add normalization info to output
        print("Z-score normalization applied:", apply_zscore)

        return original_testing, sorted_df