import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Define the Gaussian function
def gaussian(x, mu, sigma):
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
L = 30.0
N = 4000
dx = L / N
x = np.linspace(0, L, N)
mu_val = -0.25
sigma = L / 20
dt = 0.0005
T = 100.0
num_steps = int(T/dt)

# Initial condition
u_initial = gaussian(x, L/5, sigma)
u = u_initial.copy()

# Plot initial condition to verify
plt.plot(x, u_initial, label="Initial Gaussian")
plt.legend()
plt.show()

input("Press Enter to proceed with time evolution...")

# Time integration and plotting
images = []
plot_interval = 500  # adjust this value to control how often you want to plot/save an image

fig, ax = plt.subplots()

for n in range(num_steps):
    if n % plot_interval == 0:
        print(f'Plotting step {n+1}/{num_steps}')
        ax.clear()
        ax.plot(x, u_initial, label="Original", linestyle='--')
        ax.plot(x, u, label="Evolved")
        ax.legend()
        ax.set_title(f"Step {n+1}/{num_steps}")
        ax.set_xlim(0, L)
        ax.set_ylim(0, 1.2)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.pause(0.01)
        
        # Convert the Matplotlib plot to a PIL Image and append to the images list
        plt.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        images.append(image)

    u = forward_euler(u, x, dx, dt, mu_val)

# Save as a GIF
images[0].save('evolution.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
