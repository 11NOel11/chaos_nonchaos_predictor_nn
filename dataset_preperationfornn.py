import numpy as np
import pandas as pd
from double_pendulum_chaoticsystem import generate_double_pendulum_data  # Import function

# Generate Double Pendulum Data
t_dp,t_theta = generate_double_pendulum_data()  # Extract time data
theta_dp=t_theta
# SHM Parameters
omega = 2 * np.pi  # Angular frequency (rad/s)
A = 1.0  # Amplitude (m)

# Generate Double Pendulum Data


# Generate SHM Data
t_shm = np.linspace(0, 20, len(t_dp))  # Match time length with double pendulum
x_shm = A * np.cos(omega * t_shm)  # Position
v_shm = -A * omega * np.sin(omega * t_shm)  # Velocity
a_shm = -A * omega**2 * np.cos(omega * t_shm)  # Acceleration

# Feature Extraction for SHM (Non-Chaotic)
shm_features = np.column_stack((x_shm, v_shm, a_shm))
shm_labels = np.zeros(len(t_shm))  # Label: 0 for non-chaotic
shm_features = np.hstack((shm_features, np.zeros((shm_features.shape[0], 1))))

# Feature Extraction for Double Pendulum (Chaotic)
# Using angles and angular velocities as features
dp_features = np.column_stack((theta_dp[0], theta_dp[1], theta_dp[2], theta_dp[3]))
dp_labels = np.ones(len(t_dp))  # Label: 1 for chaotic


# Create Dataset
features = np.vstack((shm_features, dp_features))
labels = np.hstack((shm_labels, dp_labels))

# Convert to Pandas DataFrame
dataset = pd.DataFrame(features, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
dataset['Label'] = labels

# Save Dataset
dataset.to_csv("chaotic_system_dataset.csv", index=False)
print("Dataset saved as chaotic_system_dataset.csv")
