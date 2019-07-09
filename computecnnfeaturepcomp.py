

################################################################################
# IMPORTS
################################################################################


# Standard library imports
import os
import sys
import argparse

# Third party imports
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Third party imports
from tqdm import tqdm

# Own imports
# Make Python find own defined modules
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'modules'))
from xprint import xprint
from debug import get_attributes
from dataprocessing import process_files


################################################################################
# CONSTANTS
################################################################################


################################################################################
# OPTIONS
################################################################################


# General script behavior
skip_if_exists = True
display_plots = False
save_plots = True
create_pngs = True
create_pdfs = True
produce_output_files = True

# Feature properties
feature_dtype = np.single  # == np.float32 (float becomes np.float64)
feature_vector_length = 2560

# Plotting
equalize_color_histogram = True
#use_color_map = False
display_color_bar = True
#color_map = 'viridis'
color_map = 'jet'
#color_map = 'prism'

vary_size = False
equalize_size_histogram = True
mean_size = 4

# Files
text_output_file_extension = '.csv'
value_delimiter = ','


################################################################################
# DERIVED VARIABLES
################################################################################


produce_plots = display_plots or save_plots


################################################################################
# FUNCTIONS
################################################################################


def parse_args():
    """Parse and return command line arguments"""

    # Create parser
    parser = argparse.ArgumentParser(description="Compute principal components of the frame embeddings")

    # Add positional arguments
    parser.add_argument('input_folder', type=str,
                        help="The path of the folder that contains the input files.")
    parser.add_argument('output_folder', type=str,
                        help="The path of the folder in which to put the generated files.")
    parser.add_argument('num_components', type=int,
                        help="The number of principal components to compute.")

    # Add optional arguments
    parser.add_argument('-t', '--temp-dir-root', type=str, default=os.path.join('.', 'temp'),
                        help="The root dir for the temporary folder in which output files will be stored while they "
                             "are generated. This should be another directory than the output folder, to prevent half "
                             "finished files to end up in the output directory is the script should stop half way "
                             "through a file creation. A file will be moved to the output directory immediately after "
                             "it has been created completely.")
    parser.add_argument('-s', '--save-folder', type=str, default='',
                        help="If provided, the path to the folder into which to save the images.")
    parser.add_argument('-x', '--text', dest='text', action='store_true',
                        help="Store output components in textural format.")
    parser.add_argument('-b', '--bin', dest='text', action='store_false',
                        help="Store output components in binary format.")

    # Parse arguments
    return parser.parse_args()


def order_when_sorted(data):
    order = 0 * np.array(data)
    argsort = np.argsort(data)
    for i in argsort:
        order[argsort[i]] = i
    return order


def to_range(data, min, max, equalize_histogram=False):
    if equalize_histogram:
        data = order_when_sorted(data)

    domain_min = np.min(data)
    domain_max = np.max(data)
    return min + (data - domain_min) * ((max - min) / (domain_max - domain_min))


def pca_scatter_plot(data, title="", total_variance=None, save_folder=""):
    # Get shape of data
    num_elements, dim = data.shape

    image_name_prefix = None
    if save_folder:
        assert title != ""
        image_name_prefix = title.replace(' ', '_') + '_'
        if not os.path.isdir(save_folder):
            xprint("Creating directory '{}'".format(save_folder))
            os.makedirs(save_folder)

    for use_color_map in ((False, True) if dim >= 2 else ()):
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        fig.canvas.set_window_title(title)

        # Create scatter plot of frames as a subplot
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("1:st principal component", fontsize=15)
        ax.set_ylabel("2:nd principal component", fontsize=15)
        ax.set_title(title, fontsize=20)
        if use_color_map:
            color = np.arange(num_elements)
        else:
            color = (np.column_stack([to_range(data[:, channel + 2], 0, 1, equalize_histogram=equalize_color_histogram)
                                     for channel in range(3)])
                     if dim >= 5 else None)
        size = mean_size
        if vary_size and dim >= 6:
            size = mean_size * to_range(data[:, 5], 0, 2, equalize_histogram=equalize_size_histogram)
        the_plot = ax.scatter(data[:, 0], data[:, 1], s=size, c=color, cmap=color_map)
        ax.grid()
        if use_color_map and display_color_bar:
        #if display_color_bar:
            cbar = fig.colorbar(the_plot)
            cbar.ax.set_ylabel('Frame number')

        if save_folder:
            image_name = 'scatter_plot{}'.format('_color_map' if use_color_map else '')
            if create_pngs:
                plt.savefig(os.path.join(save_folder, image_name_prefix + image_name + '.png'))
            if create_pdfs:
                plt.savefig(os.path.join(save_folder, image_name_prefix + image_name + '.pdf'))
        if display_plots:
            plt.show()
        plt.close(fig)

    kernel_size = 1001
    assert kernel_size % 2 == 1
    kernel = np.array([1] * kernel_size) / kernel_size
    rows = 6
    cols = 2
    plots_per_page = rows * cols
    for page in range(4):
        offset = page * plots_per_page
        if offset >= dim:
            break
        fig = plt.figure(figsize=(20, 16))
        fig.canvas.set_window_title(title)
        fig.suptitle("{} principal components".format(title), fontsize=16)

        # Plot individual principal components as time series
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.7)
        for comp_idx in range(offset, offset + plots_per_page):
            if comp_idx >= dim:
                break
            curve = fig.add_subplot(rows, cols, 1 + comp_idx - offset)
            curve.plot(np.arange(num_elements - kernel_size + 1) + 1 + (kernel_size - 1) / 2,
                       np.convolve(data[:, comp_idx], kernel, mode='valid'))
            curve.grid()
            curve.set_title("Principal component #{}".format(comp_idx + 1), fontsize=10)
            curve.set_xlabel("Frame number")
            curve.set_ylabel("Component value")

        if save_folder:
            image_name = 'cp_{}--{}'.format(offset + 1, offset + plots_per_page)
            if create_pngs:
                plt.savefig(os.path.join(save_folder, image_name_prefix + image_name + '.png'))
            if create_pdfs:
                plt.savefig(os.path.join(save_folder, image_name_prefix + image_name + '.pdf'))
        if display_plots:
            # Show figure
            plt.show()
        plt.close(fig)

    if dim >= 1:
        # Plot explainability
        fig = plt.figure(figsize=(8, 8))
        fig.canvas.set_window_title(title)

        variances = np.var(data, axis=0)
        total_var = np.sum(variances) if total_variance is None else total_variance
        #plt.plot(np.arange(dim) + 1, variances / total_var)
        plt.plot(np.arange(dim + 1), np.concatenate(([0], np.cumsum(variances))) / total_var)
        plt.grid(b=True, which='both', axis='both')
        plt.xlabel("Number of principal components used")
        plt.ylabel("Coefficient of determination")
        plt.ylim((0, 1))

        if save_folder:
            image_name = 'explainability'
            if create_pngs:
                plt.savefig(os.path.join(save_folder, image_name_prefix + image_name + '.png'))
            if create_pdfs:
                plt.savefig(os.path.join(save_folder, image_name_prefix + image_name + '.pdf'))
        if display_plots:
            plt.show()
        plt.close(fig)


def create_compute_principal_components_function(num_components, save_folder, text=False):
    # Get PCA function
    pca = PCA(n_components=num_components)

    # Create the function which we are interested in
    def compute_principal_components(in_feature_file_path, out_feature_file_path):
        with open(in_feature_file_path, "rb") as in_feature_file:
            # Read features from file
            features = np.reshape(np.fromfile(in_feature_file, dtype=feature_dtype), (-1, feature_vector_length))

            # Compute principal components
            pc = pca.fit_transform(features)

            if produce_output_files:
                xprint("Creating '{}'...".format(out_feature_file_path))
                # Write principal components to file
                if text:
                    with open(out_feature_file_path, "w") as out_feature_file:
                        np.savetxt(out_feature_file, pc, delimiter=value_delimiter)
                else:
                    with open(out_feature_file_path, "wb") as out_feature_file:
                        pc.tofile(out_feature_file)

            if produce_plots:
                pca_scatter_plot(pc,
                                 title=os.path.basename(in_feature_file_path),
                                 total_variance=np.sum(np.var(features, axis=0)),
                                 save_folder=save_folder)

    return compute_principal_components


def main():
    # Parse arguments
    args = parse_args()
    for key, val in get_attributes(args).items():
        xprint("args.{}: {}".format(key, repr(val)))

    if not args.save_folder:
        global save_plots, produce_plots
        save_plots = False
        produce_plots = display_plots

    process_files(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        file_name_deriver=(lambda x: (x + text_output_file_extension) if args.text else x),
        file_processor=create_compute_principal_components_function(
            args.num_components,
            args.save_folder if save_plots else "",
            args.text),
        skip_if_exists=skip_if_exists and not produce_plots,
        temp_dir_root=args.temp_dir_root,
        produce_output_files=produce_output_files)


################################################################################
# SCRIPT ENTRY POINT
################################################################################


if __name__ == '__main__':
    exit(main())
