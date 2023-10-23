import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian function
def gaussian(x, mu=0, sigma=0.1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Source term
def source(x, mu):
    return 0.02 * np.exp(mu * x)

# Forward Euler with upwind differencing for the given equation
def forward_euler(u, x, dx, dt, mu):
    u_new = np.empty_like(u)
    u_new[0] = u[0]  # Boundary condition
    for i in range(1, len(u)):
        advection_term = u[i] * (u[i] - u[i-1]) / dx
        u_new[i] = u[i] - dt * (advection_term - source(x[i], mu))
    return u_new

# Parameters
L = 10.0
N = 8000
dx = L / N
x = np.linspace(-L/2, L/2, N)
mu_val = -0.25
sigma = L / 20
dt = 0.0005  # Further reduced time step for stability
T = 10.0
num_steps = int(T/dt)

# Initial condition
u_initial = gaussian(x)
u = u_initial.copy()

# Plot initial condition to verify
plt.plot(x, u_initial, label="Initial Gaussian")
plt.legend()
plt.show()

# Proceed only if initial condition looks good
input("Press Enter to proceed with time evolution...")

# Time integration
for n in range(num_steps):
    print(f'Time step {n+1}/{num_steps}')
    u = forward_euler(u, x, dx, dt, mu_val)

# Plotting results
plt.plot(x, u_initial, label="Original", linestyle='--')
plt.plot(x, u, label="Evolved")
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title("Nonlinear Advection with Source Term")
plt.show()