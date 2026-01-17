# OneNNIsolationForest (1-Nearest Neighbor Isolation Forest) is an anomaly detection algorithm that identifies anomalous patterns using the Isolation Forest ensemble method.
#
# This implementation provides two execution modes: sequential CPU processing and parallel multi-core CPU processing using scikit-learn's Isolation Forest.
#
# The algorithm supports both original and differenced time series data.
#
# **Importing this algorithm into a python program**
#
#             from geoanalytics.oneClassClassifiers import OneNNIsolationForest
#
#             training = pd.read_csv('training_data.csv')
#             testing = pd.read_csv('testing_data.csv')
#
#             obj = OneNNIsolationForest()
#
#             results, top_anomalies = obj.run(training, testing, topK=10, mode='parallel', algorithm='rasterIsolationForest')
#
#             print("Total results with anomaly scores:", len(results))
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

import pandas as pd
import numpy as np
import time
import psutil
from tqdm import tqdm
from sklearn.ensemble import IsolationForest


class OneNNIsolationForest:
    """
    **About this algorithm**

    :**Description**: OneNNIsolationForest (1-Nearest Neighbor Isolation Forest) is an anomaly detection
                      algorithm that identifies anomalous patterns using the Isolation Forest ensemble method.
                      Isolation Forest isolates observations by randomly selecting features and then
                      randomly selecting a split value between the maximum and minimum values of the selected feature.
                      Anomalies are identified as observations that are easier to isolate (require fewer splits).

    :**Reference**:  Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
                     In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.

    :**Parameters**:    - **training** (*pd.DataFrame*) -- *Training dataset for building the Isolation Forest model.*
                        - **testing** (*pd.DataFrame*) -- *Testing dataset to evaluate for anomalies.*
                        - **topK** (*int*) -- *Number of top anomalous patterns to return (default: -1 returns all).*
                        - **mode** (*str*) -- *Execution mode: 'sequential' or 'parallel' (default: 'sequential').*
                        - **algorithm** (*str*) -- *Algorithm variant: 'rasterIsolationForest' or 'difrasterIsolationForest'
                                                    for differenced data (default: 'rasterIsolationForest').*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the execution process.*
                        - **endTime** (*float*) -- *To record the completion time of the execution process.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.anomalyDetection import OneNNIsolationForest

            # Load your time series data
            training = pd.read_csv('training_data.csv')
            testing = pd.read_csv('testing_data.csv')

            # Initialize the anomaly detector
            obj = OneNNIsolationForest()

            # Run the algorithm with parallel processing
            results, top_anomalies = obj.run(training, testing, topK=10, mode='parallel', algorithm='rasterIsolationForest')

            print("Total patterns with anomaly scores:", len(results))
            print("Top 5 most anomalous patterns:")
            print(top_anomalies.head(5))

            # Access statistics
            print("Memory consumption in KB:", process.memory_info().rss / 1024)

    **Example Usage**

    .. code-block:: python

            # Example 1: Basic usage with sequential processing
            obj = OneNNIsolationForest()
            results, top_anomalies = obj.run(training_data, testing_data, topK=5, mode='sequential')

            # Example 2: Using differenced data with parallel processing
            results, top_anomalies = obj.run(training_data, testing_data, topK=20, mode='parallel', algorithm='difrasterIsolationForest')

            # Example 3: Returning all results sorted by anomaly score
            results, top_anomalies = obj.run(training_data, testing_data, topK=-1, mode='parallel')

    **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

    """

    def __init__(self):
        """
        Initialize the OneNNIsolationForest anomaly detector.

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

    # ---------------------- Isolation Forest Modes ----------------------
    def compute_iforest_sequential(self, training, testing):
        """
        Compute anomaly scores using sequential CPU processing.

        :param training: Training DataFrame for model fitting
        :type training: pd.DataFrame
        :param testing: Testing DataFrame for scoring
        :type testing: pd.DataFrame
        :return: Array of anomaly scores for each testing instance (higher scores = more normal)
        :rtype: np.ndarray
        """
        print("Training Isolation Forest (sequential)...")
        model = IsolationForest(n_estimators=100, contamination='auto', n_jobs=1, random_state=42)
        model.fit(training)

        print("Scoring test samples...")
        scores = model.decision_function(testing)
        return scores

    def compute_iforest_parallel(self, training, testing):
        """
        Compute anomaly scores using parallel CPU processing.

        :param training: Training DataFrame for model fitting
        :type training: pd.DataFrame
        :param testing: Testing DataFrame for scoring
        :type testing: pd.DataFrame
        :return: Array of anomaly scores for each testing instance (higher scores = more normal)
        :rtype: np.ndarray
        """
        print("Training Isolation Forest (parallel)...")
        model = IsolationForest(n_estimators=100, contamination='auto', n_jobs=-1, random_state=42)
        model.fit(training)

        print("Scoring test samples in parallel...")
        scores = model.decision_function(testing)
        return scores

    # ---------------------- Main Function ----------------------
    def run(self, training, testing, topK=-1, mode="sequential", algorithm='rasterIsolationForest'):
        """
        Main execution method for the OneNNIsolationForest algorithm.

        :param training: Training dataset for building the Isolation Forest model
        :type training: pd.DataFrame
        :param testing: Testing dataset to evaluate for anomalies
        :type testing: pd.DataFrame
        :param topK: Number of top anomalous patterns to return (default: -1 returns all)
        :type topK: int
        :param mode: Execution mode: 'sequential' or 'parallel' (default: 'sequential')
        :type mode: str
        :param algorithm: Algorithm variant: 'rasterIsolationForest' or 'difrasterIsolationForest'
                          for differenced data (default: 'rasterIsolationForest')
        :type algorithm: str
        :return: Tuple containing (complete_results_with_anomaly_scores, top_K_most_anomalous_patterns)
        :rtype: (pd.DataFrame, pd.DataFrame)
        :raises ValueError: If an invalid mode is specified
        """
        start_time = time.time()

        if algorithm == "difrasterIsolationForest":
            training = training.diff(axis=1).iloc[:, 1:]
            testing = testing.diff(axis=1).iloc[:, 1:]

        if mode == "sequential":
            scores = self.compute_iforest_sequential(training, testing)
        elif mode == "parallel":
            scores = self.compute_iforest_parallel(training, testing)
        else:
            raise ValueError("Invalid mode. Choose 'sequential', or 'parallel'")

        testing['IForest_Score'] = scores
        sorted_df = testing.sort_values('IForest_Score', ascending=False).head(topK)
        self.getStatistics(start_time)
        return testing, sorted_df