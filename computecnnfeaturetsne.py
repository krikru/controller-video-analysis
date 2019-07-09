

################################################################################
# IMPORTS
################################################################################


# Standard library imports
import os
import sys
import argparse

# Third party imports
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt

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
skip_if_exists = False
display_plots = True
save_plots = True
create_pngs = True
create_pdfs = True
procuce_output_files = True

# Feature properties
feature_dtype = np.single  # == np.float32 (float becomes np.float64)

# Plotting
equalize_color_histogram = True
display_color_bar = True
#color_map = 'viridis'
#color_map = 'jet'
#color_map = 'prism'
color_maps = ['viridis', 'jet']

vary_size = False
equalize_size_histogram = True
mean_size = 1

# t-SNE
n_jobs = 4
tsne_kwargs = {
}
#n_components = 2
perplexity = 30.0
early_exaggeration = 12.0,
learning_rate = 200.0
n_iter = 1000
n_iter_without_progress = 300
min_grad_norm = 1e-07
metric = 'euclidean'
#init = 'random'
#init = 'pca'
init = 'pca-standardized'
verbose = 0
random_state = None
method = 'barnes_hut'
angle = 0.5


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
    parser.add_argument('input_dim', type=int,
                        help="The dimensionality of the input data.")
    parser.add_argument('output_folder', type=str,
                        help="The path of the folder in which to put the generated files.")
    parser.add_argument('target_dim', type=int,
                        help="The dimensionality to reduce the data to.")

    # Add optional arguments
    parser.add_argument('-t', '--temp-dir-root', type=str, default=os.path.join('.', 'temp'),
                        help="The root dir for the temporary folder in which output files will be stored while they "
                             "are generated. This should be another directory than the output folder, to prevent half "
                             "finished files to end up in the output directory is the script should stop half way "
                             "through a file creation. A file will be moved to the output directory immediately after "
                             "it has been created completely.")
    parser.add_argument('-s', '--save-folder', type=str, default='',
                        help="If provided, the path to the folder into which to save the images.")

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


def create_plots_from_file(file_path, dimensionality, title=None, save_folder=""):
    with open(file_path, "rb") as file:
        #components = np.reshape(np.fromfile(file, dtype=feature_dtype), (-1, dimensionality))
        components = np.reshape(np.fromfile(file, dtype=np.float64), (-1, dimensionality))

    if title is None:
        title = os.path.basename(file_path)

    create_plots(components, title=title, save_folder=save_folder)


def create_plots(data, title="", save_folder=""):
    # Get shape of data
    num_elements, dim = data.shape

    image_name_prefix = None
    if save_folder:
        assert title != ""
        image_name_prefix = title.replace(' ', '_') + '_'
        if not os.path.isdir(save_folder):
            xprint("Creating directory '{}'".format(save_folder))
            os.makedirs(save_folder)

    for color_map in color_maps:
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        fig.canvas.set_window_title(title)

        # Create scatter plot of frames as a subplot
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title, fontsize=20)
        color = np.arange(num_elements)
        size = mean_size
        if vary_size and dim >= 6:
            size = mean_size * to_range(data[:, 5], 0, 2, equalize_histogram=equalize_size_histogram)
        the_plot = ax.scatter(data[:, 0], data[:, 1], s=size, c=color, cmap=color_map)
        ax.grid()

        # Display_color_bar:
        cbar = fig.colorbar(the_plot)
        cbar.ax.set_ylabel('Frame number')

        if save_folder:
            image_name = 'scatter_plot{}'.format('_{}'.format(color_map))
            if create_pngs:
                plt.savefig(os.path.join(save_folder, image_name_prefix + image_name) + '.png')
            if create_pdfs:
                plt.savefig(os.path.join(save_folder, image_name_prefix + image_name) + '.pdf')
        if display_plots:
            plt.show()
        plt.close(fig)


def create_compute_tsne_components_function(input_dim, target_dim, save_folder):
    # Get t-SNE function
    #tsne = TSNE(n_jobs=4, **tsne_kwargs)
    if False:
        tsne = TSNE(n_jobs=n_jobs,
                    n_components=target_dim,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    learning_rate=learning_rate,
                    n_iter=n_iter,
                    n_iter_without_progress=n_iter_without_progress,
                    min_grad_norm=min_grad_norm,
                    metric=metric,
                    init=init,
                    verbose=verbose,
                    random_state=random_state,
                    method=method,
                    angle=angle)

    # Create the function which we are interested in
    def compute_tsne_components(in_feature_file_path, out_feature_file_path):
        with open(in_feature_file_path, "rb") as in_feature_file:
            # Read features from file
            features = np.reshape(np.fromfile(in_feature_file, dtype=feature_dtype), (-1, input_dim))

            tsne = TSNE(n_jobs=4,
                        init=(features[:, 0:2] / np.var(features[:, 0:2], axis=0, keepdims=True) if init == 'pca-standardized' else
                              features[:, 0:2] if init == 'pca' else
                              init),
                        n_iter=1,
                        **tsne_kwargs)

            # Compute t-SNE components
            components = tsne.fit_transform(features)
            print("components.dtype:", components.dtype)

            xprint("features.shape:", features.shape)
            xprint("components.shape:", components.shape)

            if procuce_output_files:
                xprint("Creating '{}'...".format(out_feature_file_path))
                with open(out_feature_file_path, "wb") as out_feature_file:
                    # Write standardized features to file
                    components.tofile(out_feature_file)

            if produce_plots:
                create_plots(
                    components,
                    title=os.path.basename(in_feature_file_path),
                    save_folder=save_folder)

    return compute_tsne_components


def main():
    # Parse arguments
    args = parse_args()
    for key, val in get_attributes(args).items():
        xprint("args.{}: {}".format(key, repr(val)))

    process_files(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        file_name_deriver=(lambda x: x),
        file_processor=create_compute_tsne_components_function(
            args.input_dim,
            args.target_dim,
            args.save_folder if save_plots else ""),
        skip_if_exists=skip_if_exists and not produce_plots,
        temp_dir_root=args.temp_dir_root,
        procuce_output_files=procuce_output_files)


################################################################################
# SCRIPT ENTRY POINT
################################################################################


if __name__ == '__main__':
    exit(main())
