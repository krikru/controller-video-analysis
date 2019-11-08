

################################################################################
# IMPORTS
################################################################################


import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from math import log, floor, ceil


################################################################################
# OPTIONS
################################################################################


data_dimensionality = 2560
pc_dimensionality = 2560  # Should match the PC data folder
out_folder = '.'
axis_fontsize = 17


################################################################################
# FUNCTIONS
################################################################################


def main():
    data_in_folder, pc_in_folder = sys.argv[1:]

    assert 'pc{}/'.format(pc_dimensionality) in pc_in_folder.replace('\\', '/')

    cam_variances = []
    for file in sorted(os.listdir(data_in_folder)):
        # Compute total variance
        components = np.reshape(np.fromfile(os.path.join(data_in_folder, file), dtype=np.float32), (-1, data_dimensionality))
        all_variances = np.var(components, axis=0)
        total_variance = np.sum(all_variances)

        # Compute principal component variances
        pc = np.reshape(np.fromfile(os.path.join(pc_in_folder, file), dtype=np.float32), (-1, pc_dimensionality))
        pc_variances = np.var(pc, axis=0)

        # Store computed values
        cam_variances.append((file, pc_variances, total_variance))

    # Plot coefficient of determination when using a single principal component
    for xscale in ['linear', 'log']:
        yscale = 'log'
        image_name = "single_pc_r_squared_{}_{}-{}".format(pc_dimensionality, xscale, yscale)
        y_label = "Coefficient of determination"

        fig = plt.figure(figsize=(8, 8))
        fig.canvas.set_window_title("")

        y_min = 1
        y_max = 0
        for file, pc_variances, total_variance in cam_variances:
            y_values = pc_variances / total_variance
            y_min = min(y_min, np.min(y_values))
            y_max = max(y_max, np.max(y_values))
            plt.plot(np.arange(pc_dimensionality) + 1, y_values, label=file)

        plt.grid(b=True, which='major', axis='both')
        plt.xlabel("Principal component", fontsize=axis_fontsize)
        plt.ylabel(y_label, fontsize=axis_fontsize)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.xlim((0 if xscale == 'linear' else 1, pc_dimensionality))
        if yscale == 'linear':
            plt.ylim(bottom=0)
        else:
            plt.ylim((10**floor(log(y_min, 10)), 10**ceil(log(y_max, 10))))
        plt.legend()

        plt.savefig(os.path.join(out_folder, image_name + '.png'))
        plt.savefig(os.path.join(out_folder, image_name + '.pdf'))
        plt.show()

        plt.close(fig)

    # Plot coefficient of determination when using a single principal component
    image_name = "cumulative_r_squared_{}".format(pc_dimensionality)
    y_label = "Cumulative coefficient of determination"

    fig = plt.figure(figsize=(8, 8))
    fig.canvas.set_window_title("")

    for file, pc_variances, total_variance in cam_variances:
        plt.plot(np.arange(pc_dimensionality + 1), np.concatenate(([0], np.cumsum(pc_variances))) / total_variance, label=file)

    plt.grid(b=True, which='both', axis='both')
    plt.xlabel("Number of principal components used", fontsize=axis_fontsize)
    plt.ylabel(y_label, fontsize=axis_fontsize)
    plt.yscale("linear")
    plt.xlim((0, pc_dimensionality))
    plt.ylim((0, 1))
    plt.legend()

    plt.savefig(os.path.join(out_folder, image_name + '.png'))
    plt.savefig(os.path.join(out_folder, image_name + '.pdf'))
    plt.show()

    plt.close(fig)


################################################################################
# SCRIPT ENTRY POINT
################################################################################


if __name__ == '__main__':
    exit(main())
