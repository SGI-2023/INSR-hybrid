import numpy as np
import torch
from base import gradient
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial

def gaussianND_like(x, sdim=2, mu=0, sigma=0.1):
    """normalized gaussian distribution"""
    return torch.exp(-0.5 * torch.sum((x - mu)**2, dim=-1, keepdim=True) / (sdim * sigma ** 2))

source_func = partial(gaussianND_like, sdim=2, mu=-1.5)


cmap = "cividis"

samples_path = "plots/samples.npy.npz"
samples = np.load(samples_path)['arr_0']

dense_values_path = "plots/dense.npy.npz"
dense_values = np.load(dense_values_path)['arr_0']

hash_values_path = "plots/hash.npy.npz"
hash_values = np.load(hash_values_path)['arr_0']

siren_values_path = "plots/siren.npy.npz"
siren_values = np.load(siren_values_path)['arr_0']

goodhash_values_path = "plots/goodhash.npy.npz"
goodhash_values = np.load(goodhash_values_path)['arr_0']

samples_tensor = torch.from_numpy(samples).requires_grad_(True)
#gt_values = torch.exp(-0.5 * torch.sum((samples_tensor + 1.5)**2, dim=-1, keepdim=True) / (2 * 0.1 ** 2))
gt_values = source_func(samples_tensor)

gt_grad = gradient(gt_values, samples_tensor).detach().numpy()
gt_mag = np.linalg.norm(gt_grad, axis=-1)

# Plot Ground Truth + Siren + Dense + Hash
fig, axs = plt.subplots(2,2,figsize=(5,5))

# Ground Truth
gt = axs[0,0].scatter(samples[:,0], samples[:,1], c=gt_mag, cmap=cmap, s=1)
axs[0,0].set_title('Ground Truth')
axs[0,0].set_aspect('equal')
divider_gt = make_axes_locatable(axs[0,0])
cax_gt = divider_gt.append_axes("right", size="5%", pad=0.05)
fig.colorbar(gt,cax=cax_gt)

# Siren
siren = axs[0,1].scatter(samples[:,0], samples[:,1], c=siren_values, cmap=cmap, s=1)
axs[0,1].set_title('SIREN')
axs[0,1].set_aspect('equal')
divider_siren = make_axes_locatable(axs[0,1])
cax_siren = divider_siren.append_axes("right", size="5%", pad=0.05)
fig.colorbar(siren,cax=cax_siren)

# Dense
dense = axs[1,0].scatter(samples[:,0], samples[:,1], c=dense_values, cmap=cmap, s=1)
axs[1,0].set_title('Dense grid')
axs[1,0].set_aspect('equal')
divider_dense = make_axes_locatable(axs[1,0])
cax_dense = divider_dense.append_axes("right", size="5%", pad=0.05)
fig.colorbar(dense,cax=cax_dense)

# Hash
hash = axs[1,1].scatter(samples[:,0], samples[:,1], c=hash_values, cmap=cmap, s=1)
axs[1,1].set_title('Hash grid')
axs[1,1].set_aspect('equal')
divider_hash = make_axes_locatable(axs[1,1])
cax_hash = divider_hash.append_axes("right", size="5%", pad=0.05)
fig.colorbar(hash,cax=cax_hash)

plt.savefig("first_comparison.png")
plt.show()

# Plot Ground Truth + Siren + Good Hash
fig, axs = plt.subplots(1,3,figsize=(9,3))

# Ground Truth
gt = axs[0].scatter(samples[:,0], samples[:,1], c=gt_mag, cmap=cmap, s=1)
axs[0].set_title('Ground Truth')
axs[0].set_aspect('equal')
divider_gt = make_axes_locatable(axs[0])
cax_gt = divider_gt.append_axes("right", size="5%", pad=0.05)
fig.colorbar(gt,cax=cax_gt)

# Siren
siren = axs[1].scatter(samples[:,0], samples[:,1], c=siren_values, cmap=cmap, s=1)
axs[1].set_title('SIREN')
axs[1].set_aspect('equal')
divider_siren = make_axes_locatable(axs[1])
cax_siren = divider_siren.append_axes("right", size="5%", pad=0.05)
fig.colorbar(siren,cax=cax_siren)

# Good Hash
goodhash = axs[2].scatter(samples[:,0], samples[:,1], c=goodhash_values, cmap=cmap, s=1)
axs[2].set_title('Hash grid')
axs[2].set_aspect('equal')
divider_goodhash = make_axes_locatable(axs[2])
cax_goodhash = divider_goodhash.append_axes("right", size="5%", pad=0.05)
fig.colorbar(goodhash,cax=cax_goodhash)

plt.savefig("second_comparison.png")
plt.show()