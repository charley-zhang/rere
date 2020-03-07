r"""
Module: utils.visualize

Defines the API regarding common image, data visualization.
"""

import sys, os
import math

import matplotlib.pyplot as plt

from rere.utils import images, files


def plot(X, Y, ax, param_dict):
    pass


def visualize(imgobjs, cols=4, collated=True, size=None):
    r"""
    Displays image either individually or in a collated grid.
    
    Parameters:
        imgobjs - list (file/dir paths, 3d/4d np, 3d/4d tensor, pils), or 
                  single (file/dir, 3d/4d np, 3d/4d tensor, pil, df)
    """

    ## Separate into list of single instance image objects
    imgs = []
    if isinstance(imgobjs, list):
        for io in imgobjs:
            imgs += images._create_img_list(io)
    else:
        imgs = images._create_img_list(imgobjs)

    ## Grid layout settings. Sets N, N_rows, N_cols
    N = len(imgs)
    assert N > 0
    if not size:
        size = [0, 0]  # H, W
        for img in imgs:
            _, _, H, W = get_dimensions(img)
            size[0] += H
            size[1] += W
        size = [int(d/len(imgs)) for d in size]
    else:
        assert len(size) == 2

    N_cols = cols if cols else 4
    if N < 4:
        N_cols = N
    N_rows = math.ceil(N/N_cols)
    print(f"Cols: {N_cols}, Rows: {N_rows}")

    ## Display Figure
    figure = plt.figure(figsize=(15, 10))
    for i in range(N):
        dims = images.get_dimensions(imgs[i])[1:]
        title = f"[Image {i+1}/{N}]"
        if isinstance(imgs[i], str):
            title = f"[Image {i+1}/{N}] {files.get_filename(imgs[i])}"
        title += f"\n shape{dims}"
        img = images.to_np(imgs[i], size=size, color='rgb')
        subplt = figure.add_subplot(N_rows, N_cols, i+1)
        subplt.set_title(title, fontsize=10)
        subplt.axis('off')
        plt.imshow(img)
    figure.tight_layout()
    # plt.subplots_adjust(wspace=.25, hspace=.5)
    plt.show()