import matplotlib.cm as cm
import matplotlib.colors as clr
import matplotlib.pyplot as plt

from ..palettes import SEABORN


def create_discrete_cmap(n):
    """Create an n-bin discrete colormap."""
    if n <= len(SEABORN):
        colors = list(SEABORN.values())[:n]
    else:
        base = plt.cm.get_cmap('Paired')
        color_list = base([(i + 1) / (n + 1) for i in range(n)])
        cmap_name = base.name + str(n)
        return base.from_list(cmap_name, color_list, n)
    return clr.ListedColormap(colors)


def add_color_bar(ax, cmap, labels):
    """Add a colorbar to an axis.

    Args:
        ax (AxesSubplot)
        cmap (Colormap): A prepaped colormap of size n.
        labels (list of str): A list of strings of size n.
    """
    norm = clr.BoundaryNorm(list(range(cmap.N+1)), cmap.N)
    smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    smap.set_array([])
    cbar = plt.colorbar(smap, ax=ax)
    cbar.set_ticks([i + 0.5 for i in range(cmap.N)])
    cbar.set_ticklabels(labels)
