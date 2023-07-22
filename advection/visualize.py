import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def draw_vector_field2D(vel, coords):
    u, v = vel[..., 0], vel[..., 1]
    x, y = coords[..., 0], coords[..., 1]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.quiver(x, y, u, v, scale=u.shape[0], scale_units='width')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    return fig


def draw_scalar_field2D(arr, vmin=None, vmax=None, cmap=None):
    multi = max(arr.shape[0] // 512, 1)
    fig, ax = plt.subplots(figsize=(5 * multi, 5 * multi))
    cax1 = ax.matshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(cax1, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def draw_signal1D(x, y, y_max=None, y_gt=None):
    fig, ax = plt.subplots()
    if y_gt is not None:
        ax.plot(x, y_gt, color='red', alpha=0.2)
    ax.plot(x, y)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig

def scatter_signal1D(x, y, y_max=None, y_gt=None):
    fig, ax = plt.subplots()
    if y_gt is not None:
        ax.scatter(x, y_gt, color='red', alpha=0.2, s=0.2)
    ax.scatter(x, y)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig

def scatter_compare2D(arr, t_color, gt_color, e_color, scatter_radius = 1.0, cmap = 'cool'):
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,45))

    sc1 = ax1.scatter(arr[:,0], arr[:,1], c = t_color, cmap=cmap, s=scatter_radius)
    ax1.set_title('Result')
    ax1.set_aspect('equal')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(sc1,cax=cax1)

    sc2 = ax2.scatter(arr[:,0], arr[:,1], c = gt_color, cmap=cmap, s=scatter_radius)
    ax2.set_title('Ground truth')
    ax2.set_aspect('equal')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(sc2,cax=cax2)

    sc3 = ax3.scatter(arr[:,0], arr[:,1], c = e_color, cmap=cmap, s=scatter_radius)
    ax3.set_title('Error')
    ax3.set_aspect('equal')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(sc3,cax=cax3)
    
    #fig.set_aspect('equal')
    #fig.tight_layout()
    #fig.colorbar(sc)
    return fig

def scatter_signal2D(arr, vmin=None, vmax=None, color=None, scatter_radius = 1.0, cmap = 'cool'):
    fig, ax = plt.subplots()
    if color is None:
        sc = ax.scatter(arr[:,0], arr[:,1], c = arr[:,0]+arr[:,1], cmap=cmap, s=scatter_radius)
    else:
        sc = ax.scatter(arr[:,0], arr[:,1], c = color, cmap=cmap, s=scatter_radius)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.colorbar(sc)
    return fig

def save_figure(fig, save_path, close=True):
    plt.savefig(save_path, bbox_inches='tight')
    if close:
        plt.close(fig)
