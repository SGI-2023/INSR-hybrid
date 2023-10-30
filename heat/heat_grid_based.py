import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Define the Gaussian function
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def laplacian_gaussian(x, mu, sigma):
    """Laplacian of the normalized gaussian distribution"""
    term1 = np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / sigma ** 2
    term2 = ((x - mu) ** 2 / sigma ** 2) - 1
    return term1 * term2

# Forward Euler method for the heat equation
def forward_euler_heat(u, dx, dt, alpha):
    u_new = np.empty_like(u)
    for i in range(1, len(u)-1):
        u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    u_new[0] = u[0]  
    u_new[-1] = u[-1]
    return u_new

# Parameters
L = 100.0             # Length of domain
N = 400               # Number of grid points
dx = L / N            # Grid spacing
x = np.linspace(0, L, N) # x coordinates
mu = L / 2            # Mean of the Gaussian
sigma = L / 20        # Standard deviation of the Gaussian
alpha = 0.5           # Thermal diffusivity (you may need to adjust this)
dt = 0.005           # Time step (you may need to adjust this for stability)
T = 600.0               # Total simulation time
num_steps = int(T/dt)

# Initial condition
u_initial = laplacian_gaussian(x, mu, sigma)
u = u_initial.copy()

# Plot initial condition to verify
plt.plot(x, u_initial, label="Initial Laplacian Gaussian")
plt.legend()
plt.show()

input("Press Enter to proceed with time evolution...")

images = []
plot_interval = 100  # adjust this value to control how often you want to plot/save an image

fig, ax = plt.subplots()
# Time integration using forward Euler
for n in range(num_steps):
    if n % plot_interval == 0:
        print(f'Plotting step {n+1}/{num_steps}')
        ax.clear()
        ax.plot(x, u_initial, label="Original", linestyle='--')
        ax.plot(x, u, label="Evolved")
        ax.legend()
        ax.set_title(f"Step {n+1}/{num_steps}")
        ax.set_xlim(0, L)
        ax.set_ylim(np.min(u_initial), np.max(u_initial))
        plt.xlabel("x")
        plt.ylabel("u")
        plt.pause(0.01)
        
        # Convert the Matplotlib plot to a PIL Image and append to the images list
        plt.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        images.append(image)

    u = forward_euler_heat(u, dx, dt, alpha)

# Save as a GIF
images[0].save('laplacian_heat_evolution.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
