#!/usr/bin/env python3
"""
V2: Script to run Hawkes analysis on bund2.npz (enlarged dataset) using hawkes_conditional_law_v2.py
Enhanced with numba optimizations for better performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from hawkes_conditional_law_v2 import HawkesConditionalLaw


def share_y(ax):
    """Manually share y axis on an array of axis
    
    Parameters
    ----------
    ax : `np.ndarray`
        2d array of axes that will share y
        
    Notes
    -----
    This utlity is useful as sharey kwarg of subplots cannot be applied only 
    on a subset of the axes
    """
    n_rows, n_cols = ax.shape
    get_ylim = np.vectorize(lambda axis: axis.get_ylim())
    y_min, y_max = get_ylim(ax)
    y_min_min = y_min.min()
    y_max_max = y_max.max()
    for i in range(n_rows):
        for j in range(n_cols):
            ax[i, j].set_ylim([y_min_min, y_max_max])
            if j != 0:
                ax[i, j].get_yaxis().set_ticks([])


def share_x(ax):
    """Manually share x axis on an array of axis

    Parameters
    ----------
    ax : `np.ndarray`
        2d array of axes that will share x

    Notes
    -----
    This utlity is useful as sharex kwarg of subplots cannot be applied only 
    on a subset of the axes
    """
    n_rows, n_cols = ax.shape
    get_xlim = np.vectorize(lambda axis: axis.get_xlim())
    x_min, x_max = get_xlim(ax)
    x_min_min = x_min.min()
    x_max_max = x_max.max()
    for i in range(n_rows):
        for j in range(n_cols):
            ax[i, j].set_xlim([x_min_min, x_max_max])
            if i != n_rows - 1:
                ax[i, j].get_xaxis().set_ticks([])




def plot_hawkes_kernel_norms(kernel_object, show=True, pcolor_kwargs=None,
                             node_names=None, rotate_x_labels=0.):
    """Generic function to plot Hawkes kernel norms.

    Parameters
    ----------
    kernel_object : `Object`
        An object that must have the following API :

        * `kernel_object.n_nodes` : a field that stores the number of nodes
          of the associated Hawkes process (thus the number of kernels is
          this number squared)
        * `kernel_object.get_kernel_norms()` : must return a 2d numpy
          array with the norm of each kernel

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.

    pcolor_kwargs : `dict`, default=`None`
        Extra pcolor kwargs such as cmap, vmin, vmax

    node_names : `list` of `str`, shape=(n_nodes, ), default=`None`
        node names that will be displayed on axis.
        If `None`, node index will be used.

    rotate_x_labels : `float`, default=`0.`
        Number of degrees to rotate the x-labels clockwise, to prevent 
        overlapping.

    Notes
    -----
    Kernels are displayed such that it shows norm of column influence's
    on row.
    """
    n_nodes = kernel_object.n_nodes

    if node_names is None:
        node_names = range(n_nodes)
    elif len(node_names) != n_nodes:
        ValueError('node_names must be a list of length {} but has length {}'
                   .format(n_nodes, len(node_names)))

    row_labels = ['${} \\rightarrow$'.format(i) for i in node_names]
    column_labels = ['$\\rightarrow {}$'.format(i) for i in node_names]

    norms = kernel_object.get_kernel_norms()
    fig, ax = plt.subplots()

    if rotate_x_labels != 0.:
        # we want clockwise rotation because x-axis is on top
        rotate_x_labels = -rotate_x_labels
        x_label_alignment = 'right'
    else:
        x_label_alignment = 'center'

    if pcolor_kwargs is None:
        pcolor_kwargs = {}

    if norms.min() >= 0:
        pcolor_kwargs.setdefault("cmap", plt.cm.Blues)
    else:
        # In this case we want a diverging colormap centered on 0
        pcolor_kwargs.setdefault("cmap", plt.cm.RdBu)
        max_abs_norm = np.max(np.abs(norms))
        pcolor_kwargs.setdefault("vmin", -max_abs_norm)
        pcolor_kwargs.setdefault("vmax", max_abs_norm)

    heatmap = ax.pcolor(norms, **pcolor_kwargs)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(norms.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(norms.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False, fontsize=17, 
                       rotation=rotate_x_labels, ha=x_label_alignment)
    ax.set_yticklabels(column_labels, minor=False, fontsize=17)

    fig.subplots_adjust(right=0.8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    fig.colorbar(heatmap, cax=cax)

    if show:
        plt.show()

    return fig



def main():
    # Load the enlarged bund2.npz file
    print("V2-BUND2: Loading bund2.npz file (enlarged dataset)...")
    data = np.load('bund2.npz', allow_pickle=True).items()
    
    events = [list(timestamps) for _, timestamps in data]

    print(f"V2-BUND2: Loaded {len(events)} time series with enlarged dataset")

    # Define kernel discretization
    kernel_discretization = np.hstack((0, np.logspace(-5, 0, 50)))
    print(f"V2-BUND2: Kernel discretization: {len(kernel_discretization)} points from {kernel_discretization[0]} to {kernel_discretization[-1]}")
    
    # Create and configure Hawkes learner V2
    print("V2-BUND2: Creating HawkesConditionalLaw learner with numba optimizations...")
    hawkes_learner = HawkesConditionalLaw(
        claw_method="log", 
        delta_lag=0.1, 
        min_lag=5e-4, 
        max_lag=500,
        quad_method="log", 
        n_quad=10,  # Original parameter
        min_support=1e-4, 
        max_support=1, 
        n_threads=1  # Single thread for pure Python implementation
    )
    
    # Fit the model
    print("V2-BUND2: Fitting Hawkes model on enlarged dataset...")
    hawkes_learner.fit(events)
    
    print("V2-BUND2: Model fitting completed!")
    print(f"V2-BUND2: Number of nodes: {hawkes_learner.n_nodes}")
    print(f"V2-BUND2: Baseline intensities: {hawkes_learner.baseline}")
    
    # Numerical verification outputs - focus on kernel norm matrix
    print("BUND2-V2: Computing kernel norm matrix for comparison...")
    kernels_norms = hawkes_learner.get_kernel_norms()
    eigvals = np.linalg.eigvals(kernels_norms)
    spectral_radius = np.max(np.abs(eigvals))
    
    print(f"BUND2-V2: Matrix shape: {kernels_norms.shape}")
    print(f"BUND2-V2: Matrix min/max: {kernels_norms.min():.6f} / {kernels_norms.max():.6f}")
    print(f"BUND2-V2: Matrix norm (Frobenius): {np.linalg.norm(kernels_norms, 'fro'):.6f}")
    print(f"BUND2-V2: Spectral radius: {spectral_radius:.6f}")
    print("BUND2-V2: First 5x5 submatrix:")
    print(kernels_norms[:5, :5])
    print("BUND2-V2: Diagonal elements:")
    print(np.diag(kernels_norms))
    
    # Save matrix for comparison
    np.save('kernels_norms_v2.npy', kernels_norms)
    print("BUND2-V2: Kernel matrix saved as kernels_norms_v2.npy")
    
    print("BUND2-V2: Analysis completed successfully!")

if __name__ == "__main__":
    main()
