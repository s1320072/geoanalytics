# OneNNSVM (1-Nearest Neighbor Support Vector Machine) is an anomaly detection algorithm that identifies anomalous patterns using One-Class Support Vector Machines.
#
# This implementation provides two execution modes: sequential sample-by-sample processing and parallel batch processing using scikit-learn's OneClassSVM.
#
# The algorithm supports both original and differenced time series data.
#
# **Importing this algorithm into a python program**
#
#             from geoanalytics.oneClassClassifiers import OneNNSVM
#
#             training = pd.read_csv('training_data.csv')
#             testing = pd.read_csv('testing_data.csv')
#
#             obj = OneNNSVM()
#
#             results, top_anomalies = obj.run(training, testing, topK=10, mode='parallel', algorithm='OneClassSVM')
#
#             print("Total results with SVM scores:", len(results))
#             print("Top anomalous patterns:", top_anomalies.head())
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

import numpy as np
import pandas as pd
import time
import psutil
from tqdm import tqdm
from sklearn.svm import OneClassSVM


class OneNNSVM:
    """
    **About this algorithm**

    :**Description**: OneNNSVM (1-Nearest Neighbor Support Vector Machine) is an anomaly detection
                      algorithm that identifies anomalous patterns using One-Class Support Vector Machines.
                      One-Class SVM learns a decision boundary that separates normal data from anomalies
                      by mapping data into a high-dimensional feature space using the kernel trick
                      and finding a hyperplane that maximally separates the data from the origin.

    :**Reference**:  Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001).
                     Estimating the support of a high-dimensional distribution.
                     Neural computation, 13(7), 1443-1471.

    :**Parameters**:    - **training** (*pd.DataFrame*) -- *Training dataset for building the One-Class SVM model.*
                        - **testing** (*pd.DataFrame*) -- *Testing dataset to evaluate for anomalies.*
                        - **topK** (*int*) -- *Number of top anomalous patterns to return (default: -1 returns all).*
                        - **mode** (*str*) -- *Execution mode: 'sequential' or 'parallel' (default: 'sequential').*
                        - **algorithm** (*str*) -- *Algorithm variant: 'OneClassSVM' or 'difOneClassSVM'
                                                    for differenced data (default: 'OneClassSVM').*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the execution process.*
                        - **endTime** (*float*) -- *To record the completion time of the execution process.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.anomalyDetection import OneNNSVM

            # Load your time series data
            training = pd.read_csv('training_data.csv')
            testing = pd.read_csv('testing_data.csv')

            # Initialize the anomaly detector
            obj = OneNNSVM()

            # Run the algorithm with parallel processing
            results, top_anomalies = obj.run(training, testing, topK=10, mode='parallel', algorithm='OneClassSVM')

            print("Total patterns with SVM scores:", len(results))
            print("Top 5 most anomalous patterns:")
            print(top_anomalies.head(5))

            # Access statistics
            print("Memory consumption in KB:", process.memory_info().rss / 1024)

    **Example Usage**

    .. code-block:: python

            # Example 1: Basic usage with sequential processing
            obj = OneNNSVM()
            results, top_anomalies = obj.run(training_data, testing_data, topK=5, mode='sequential')

            # Example 2: Using differenced data with parallel processing
            results, top_anomalies = obj.run(training_data, testing_data, topK=20, mode='parallel', algorithm='difOneClassSVM')

            # Example 3: Returning all results sorted by anomaly score
            results, top_anomalies = obj.run(training_data, testing_data, topK=-1, mode='parallel')

    **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

    """

    def __init__(self):
        """
        Initialize the OneNNSVM anomaly detector.

        This constructor sets up the anomaly detector without any required parameters.
        All configuration is passed to the run() method.
        """
        pass

    # ---------------------- Stats Logger ----------------------
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

    # ---------------------- One-Class SVM (sequential Sample Loop) ----------------------
    def compute_ocsvm_sequential(self, training_np, testing_np):
        """
        Compute anomaly scores using sequential sample-by-sample processing.

        :param training_np: Training data as numpy array for model fitting
        :type training_np: np.ndarray
        :param testing_np: Testing data as numpy array for scoring
        :type testing_np: np.ndarray
        :return: Array of anomaly scores for each testing instance (higher scores = more normal)
        :rtype: np.ndarray
        """
        print("Training One-Class SVM (sequential)...")
        model = OneClassSVM(kernel='rbf', gamma='scale')
        model.fit(training_np)

        print("Scoring test samples (loop)...")
        scores = np.array([model.decision_function([x])[0] for x in tqdm(testing_np, desc="Scoring (sequential)")])
        return scores

    # ---------------------- One-Class SVM (Vectorized Batch) ----------------------
    def compute_ocsvm_parallel(self, training_np, testing_np):
        """
        Compute anomaly scores using parallel batch processing.

        :param training_np: Training data as numpy array for model fitting
        :type training_np: np.ndarray
        :param testing_np: Testing data as numpy array for scoring
        :type testing_np: np.ndarray
        :return: Array of anomaly scores for each testing instance (higher scores = more normal)
        :rtype: np.ndarray
        """
        print("Training One-Class SVM (parallel)...")
        model = OneClassSVM(kernel='rbf', gamma='scale')
        model.fit(training_np)

        print("Scoring test samples (batch)...")
        scores = model.decision_function(testing_np)
        return scores

    # ---------------------- Main Entry Point ----------------------
    def run(self, training, testing, topK=-1, mode="sequential", algorithm='OneClassSVM'):
        """
        Main execution method for the OneNNSVM algorithm.

        :param training: Training dataset for building the One-Class SVM model
        :type training: pd.DataFrame
        :param testing: Testing dataset to evaluate for anomalies
        :type testing: pd.DataFrame
        :param topK: Number of top anomalous patterns to return (default: -1 returns all)
        :type topK: int
        :param mode: Execution mode: 'sequential' or 'parallel' (default: 'sequential')
        :type mode: str
        :param algorithm: Algorithm variant: 'OneClassSVM' or 'difOneClassSVM'
                          for differenced data (default: 'OneClassSVM')
        :type algorithm: str
        :return: Tuple containing (complete_results_with_SVM_scores, top_K_most_anomalous_patterns)
        :rtype: (pd.DataFrame, pd.DataFrame)
        :raises ValueError: If an invalid mode is specified
        """
        start_time = time.time()

        # Optional preprocessing
        if algorithm == "difOneClassSVM":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        # Convert to NumPy for faster processing
        training_np = training.to_numpy()
        testing_np = testing.to_numpy()

        # Scoring
        if mode == "sequential":
            scores = self.compute_ocsvm_sequential(training_np, testing_np)
        elif mode == "parallel":
            scores = self.compute_ocsvm_parallel(training_np, testing_np)
        else:
            raise ValueError("Invalid mode. Choose 'sequential' or 'parallel'")

        # Return DataFrame with scores and sorted top elements
        testing = testing.copy()
        testing['OCSVM_Score'] = scores
        sorted_df = testing.sort_values('OCSVM_Score', ascending=False).head(topK)

        self.getStatistics(start_time)
        return testing, sorted_df