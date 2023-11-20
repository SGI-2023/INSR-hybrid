import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def source(x, mu):
    return 0.02 * np.exp(mu * x)


def forward_euler(u, w, x, dx, dt, mu):
    u_new = np.empty_like(u)
    w_new = np.empty_like(w)

    u_new[0] = u[0]
    w_new[0] = w[0]

    for i in range(1, len(u)):
        u_new[i] = u[i] - dt * (u[i] * w[i] - source(x[i], mu))
        w_new[i] = w[i] - dt * \
            (w[i]*w[i] + u[i]*((w[i] - w[i-1]) / dx) - mu*source(x[i], mu))

    return u_new, w_new


L = 1.0
N = 600
dx = L / N
x = np.linspace(0, L, N)
mu_val = 0.25
sigma = L / 20
dt = 0.0005
T = 20.
num_steps = int(T / dt)

u_initial = gaussian(x, L / 5, sigma)
u = u_initial.copy()
w = np.gradient(u_initial, dx)

plt.plot(x, u_initial, label="Initial Gaussian")
plt.legend()
plt.show()

input("Press Enter to proceed with time evolution...")

images = []
plot_interval = 500

fig, ax = plt.subplots()

for n in range(num_steps):
    if n % plot_interval == 0:
        print(f'Plotting step {n+1}/{num_steps}')
        ax.clear()
        ax.plot(x, u_initial, label="Original", linestyle='--')
        ax.plot(x, u, label="Evolved u")
        ax.plot(x, w, label="Evolved w (Gradient of u)", linestyle=':')
        ax.legend()
        ax.set_title(f"Step {n+1}/{num_steps}")
        ax.set_xlim(0, L*2)
        ax.set_ylim(-1, 1.2)
        plt.xlabel("x")
        plt.ylabel("Values")
        plt.pause(0.01)

        plt.draw()
        image = Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        images.append(image)

    u, w = forward_euler(u, w, x, dx, dt, mu_val)

# Save as a GIF
images[0].save('evolution.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
