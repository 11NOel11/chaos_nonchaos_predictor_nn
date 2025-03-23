import matplotlib.pyplot as plt
import numpy as np
from double_pendulum_chaoticsystem import generate_double_pendulum_data  # Import function

# Generate double pendulum time data to match SHM time length
t_dp,t_theta = generate_double_pendulum_data()  # Extract time data

# SHM Parameters
omega = 2 * np.pi  # Angular frequency (rad/s)
A = 1.0  # Amplitude (m)

# Generate SHM data
t_shm = np.linspace(0, 20, len(t_dp))  # Match time length with double pendulum
x_shm = A * np.cos(omega * t_shm)  # Position as a function of time
v_shm = -A * omega * np.sin(omega * t_shm)  # Velocity as a function of time
if __name__ == "__main__":
    # Plot SHM motion
    plt.figure(figsize=(6, 4))
    plt.plot(t_shm, x_shm, label="Position (x)", color="green")
    plt.plot(t_shm, v_shm, label="Velocity (v)", color="red", linestyle="dashed")
    plt.xlabel("Time")
    plt.ylabel("Displacement / Velocity")
    plt.title("Simple Harmonic Motion (Non-Chaotic)")
    plt.legend()
    plt.show()
