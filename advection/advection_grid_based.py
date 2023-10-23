import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian function
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Forward Euler method for the advection equation
def forward_euler(u, dx, dt, c):
    u_new = np.empty_like(u)
    for i in range(1, len(u)):
        u_new[i] = u[i] - c * dt / dx * (u[i] - u[i-1])
    u_new[0] = u[0]  # Boundary conditions (can be modified if needed)
    return u_new

# Parameters
L = 100.0            # Length of domain
N = 400             # Number of grid points
dx = L / N          # Grid spacing
x = np.linspace(0, L, N) # x coordinates
mu = L / 4          # Mean of the Gaussian
sigma = L / 20      # Standard deviation of the Gaussian
c = 1.0             # Advection speed
dt = 0.025          # Time step
T = 5.0             # Total simulation time
num_steps = int(T/dt)

# Initial condition
u_initial = gaussian(x, mu, sigma)
u = u_initial.copy()

# Time integration using forward Euler
for n in range(num_steps):
    u = forward_euler(u, dx, dt, c)

# Plotting
plt.plot(x, u_initial, label="Original")
plt.plot(x, u, label="Evolved")
plt.plot(x, gaussian(x - c*T, mu, sigma), '--', label="Analytical")
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title("Advection of Gaussian using Forward Euler")
plt.show()
