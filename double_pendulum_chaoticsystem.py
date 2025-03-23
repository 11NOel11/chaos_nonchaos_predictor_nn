import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants for double pendulum
g = 9.81  # Gravity (m/s^2)
L1, L2 = 1.0, 1.0  # Lengths of pendulums (m)
m1, m2 = 1.0, 1.0  # Masses of pendulums (kg)

# Equations of motion for double pendulum
def double_pendulum(t, state):
    theta1, z1, theta2, z2 = state
    delta = theta2 - theta1

    denominator1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    denominator2 = (L2 / L1) * denominator1

    dz1 = (
        m2 * L1 * z1 ** 2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(theta2) * np.cos(delta)
        + m2 * L2 * z2 ** 2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta1)
    ) / denominator1

    dz2 = (
        -L2 * z2 ** 2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2) * (g * np.sin(theta1) * np.cos(delta) - L1 * z1 ** 2 * np.sin(delta) - g * np.sin(theta2))
    ) / denominator2

    return [z1, dz1, z2, dz2]

# Generate chaotic double pendulum data
def generate_double_pendulum_data(time_span=(0, 20), dt=0.01, initial_state=(np.pi/2, 0, np.pi/2, 0)):
    t_eval = np.arange(time_span[0], time_span[1], dt)
    sol = solve_ivp(double_pendulum, time_span, initial_state, t_eval=t_eval, method='RK45')
    return sol.t, sol.y

# Only run this block if script is executed directly
if __name__ == "__main__":
    # Generate data
    t_dp, theta_dp = generate_double_pendulum_data()

    # Plot Double Pendulum motion
    plt.figure(figsize=(6, 4))
    plt.plot(t_dp, theta_dp[0], label="Theta1 (Pendulum 1)", color="blue")
    plt.plot(t_dp, theta_dp[2], label="Theta2 (Pendulum 2)", color="orange", linestyle="dashed")
    plt.xlabel("Time")
    plt.ylabel("Angle (radians)")
    plt.title("Double Pendulum Motion (Chaotic)")
    plt.legend()
    plt.show()
