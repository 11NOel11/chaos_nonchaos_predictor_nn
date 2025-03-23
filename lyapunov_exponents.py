from sklearn.linear_model import LinearRegression
import numpy as np
from double_pendulum_chaoticsystem import generate_double_pendulum_data  # Import function
t_dp,theta_dp = generate_double_pendulum_data()  # Extract time data
from shm_nonchaotic import x_shm # Import function

# Function to compute Lyapunov exponent from a time series
def compute_lyapunov_exponent(time_series):
    # Compute differences between successive points (local divergence)
    diffs = np.abs(np.diff(time_series))
    diffs = diffs[diffs > 0]  # Remove zero differences to avoid log issues
    
    # Take logarithm of divergence
    log_diffs = np.log(diffs)
    
    # Fit a linear model to estimate the slope (Lyapunov exponent)
    time_steps = np.arange(1, len(log_diffs) + 1).reshape(-1, 1)
    model = LinearRegression().fit(time_steps, log_diffs)
    return model.coef_[0]  # Slope is the Lyapunov exponent

# Compute Lyapunov exponent for Double Pendulum (chaotic)
lyapunov_dp = compute_lyapunov_exponent(theta_dp[0])  # Using theta1 of double pendulum

# Compute Lyapunov exponent for SHM (non-chaotic)
lyapunov_shm = compute_lyapunov_exponent(x_shm)  # Using SHM position

print(lyapunov_dp, lyapunov_shm)
#A positive Lyapunov exponent confirms that the double pendulum is chaotic (small changes grow exponentially).
#A zero/negative exponent confirms that SHM is predictable (small changes don’t grow)
#now clearly double pendulum is chaotic system and shm non chaotic both lyapunov exponent and shannon entropy supoort this.