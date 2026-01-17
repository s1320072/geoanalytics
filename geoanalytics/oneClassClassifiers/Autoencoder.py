# Autoencoder is an anomaly detection algorithm that identifies anomalous patterns using an Autoencoder neural network. This implementation uses PyTorch for training and reconstruction error computation, with GPU acceleration support.
#
# The algorithm learns a compressed representation of normal patterns and detects anomalies based on high reconstruction error.
#
# **Importing this algorithm into a python program**
#
#             from geoanalytics.oneClassClassifiers import rasterOneClassAutoencoder
#
#             training = pd.read_csv('training_data.csv')
#             testing = pd.read_csv('testing_data.csv')
#
#             results, top_anomalies = rasterOneClassAutoencoder(training, testing, topElements=10)
#
#             print("Total results with reconstruction errors:", len(results))
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
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getStatistics(start_time):
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

# ---------------------- Improved Autoencoder Model ----------------------
class Autoencoder(nn.Module):
    """
    Autoencoder neural network for anomaly detection.

    :**Architecture**:
        - Encoder: Input → 128 → 64 → 16 (bottleneck)
        - Decoder: 16 → 64 → 128 → Input (with sigmoid activation)
        - Activation: ReLU for hidden layers, Sigmoid for output
    """
    def __init__(self, input_dim):
        """
        Initialize the Autoencoder model.

        :param input_dim: Number of input features/dimensions
        :type input_dim: int
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        :param x: Input tensor
        :type x: torch.Tensor
        :return: Reconstructed tensor
        :rtype: torch.Tensor
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ---------------------- Autoencoder Modes ----------------------
def compute_autoencoder(training, testing, mode="cuda", epochs=100, batch_size=64):
    """
    Compute reconstruction errors using Autoencoder.

    :param training: Training DataFrame for model fitting
    :type training: pd.DataFrame
    :param testing: Testing DataFrame for anomaly scoring
    :type testing: pd.DataFrame
    :param mode: Execution mode: 'cuda' for GPU or 'cpu' (default: 'cuda')
    :type mode: str
    :param epochs: Number of training epochs (default: 100)
    :type epochs: int
    :param batch_size: Batch size for training and inference (default: 64)
    :type batch_size: int
    :return: Array of reconstruction errors for each testing instance
    :rtype: np.ndarray
    """
    device = torch.device("cuda" if mode == "cuda" and torch.cuda.is_available() else "cpu")

    # Standardize data
    scaler = StandardScaler()
    training_np = scaler.fit_transform(training.to_numpy())
    testing_np = scaler.transform(testing.to_numpy())

    # Convert to PyTorch tensors
    x_train = torch.tensor(training_np, dtype=torch.float32)
    x_test = torch.tensor(testing_np, dtype=torch.float32)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=batch_size, shuffle=False)

    # Initialize model
    model = Autoencoder(x_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_data.size(0)
        avg_loss = total_loss / len(x_train)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    # Inference and scoring
    print("Scoring test samples in batches...")
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            batch_data = batch[0].to(device)
            output = model(batch_data)
            loss = torch.mean((batch_data - output) ** 2, dim=1)
            errors.extend(loss.cpu().numpy())

    return np.array(errors)

# ---------------------- Main Entry ----------------------
def rasterOneClassAutoencoder(training, testing, topElements=-1):
    """
    Main execution function for the Autoencoder-based anomaly detection.

    :**Description**: Trains an Autoencoder neural network on normal training data and computes
                      reconstruction errors for testing samples. Higher reconstruction errors
                      indicate more anomalous patterns.

    :**Reference**:  Hawkins, S., He, H., Williams, G., & Baxter, R. (2002).
                     Outlier detection using replicator neural networks.
                     In International Conference on Data Warehousing and Knowledge Discovery (pp. 170-180).

    :param training: Training dataset containing normal patterns
    :type training: pd.DataFrame
    :param testing: Testing dataset to evaluate for anomalies
    :type testing: pd.DataFrame
    :param topElements: Number of top anomalous patterns to return (default: -1 returns all)
    :type topElements: int
    :return: Tuple containing (complete_results_with_reconstruction_errors, top_K_most_anomalous_patterns)
    :rtype: (pd.DataFrame, pd.DataFrame)

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.anomalyDetection import rasterOneClassAutoencoder

            # Load your time series data
            training = pd.read_csv('normal_training_data.csv')
            testing = pd.read_csv('testing_data.csv')

            # Run the autoencoder anomaly detection
            results, top_anomalies = rasterOneClassAutoencoder(training, testing, topElements=10)

            print("Total patterns with reconstruction errors:", len(results))
            print("Top 5 most anomalous patterns:")
            print(top_anomalies.head(5))

            # Access statistics
            print("Memory consumption in KB:", process.memory_info().rss / 1024)

    **Example Usage**

    .. code-block:: python

            # Example 1: Basic usage with default parameters
            results, top_anomalies = rasterOneClassAutoencoder(training_data, testing_data, topElements=5)

            # Example 2: Returning all results sorted by reconstruction error
            results, top_anomalies = rasterOneClassAutoencoder(training_data, testing_data, topElements=-1)

            # Example 3: Using with custom parameters (through compute_autoencoder directly)
            errors = compute_autoencoder(training_data, testing_data, mode='cuda', epochs=50, batch_size=32)

    **Credits**

     The complete program was written by M. Charan Teja under the supervision of Professor Rage Uday Kiran.

    """
    start_time = time.time()
    distances = compute_autoencoder(training, testing)
    testing['AE_ReconError'] = distances
    sorted_df = testing.sort_values('AE_ReconError').head(topElements)
    getStatistics(start_time)
    return testing, sorted_df