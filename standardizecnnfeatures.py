

################################################################################
# IMPORTS
################################################################################


# Standard library imports
import os
import sys
import argparse

# Third party imports
import numpy as np

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

# Feature properties
feature_dtype = np.single  # == np.float32 (float becomes np.float64)
feature_vector_length = 2560


################################################################################
# FUNCTIONS
################################################################################


def parse_args():
    """Parse and return command line arguments"""

    # Create parser
    parser = argparse.ArgumentParser(description="Standardize feature elements across all feature vectors")

    # Add positional arguments
    parser.add_argument('input_folder', type=str,
                        help="The path of the folder containing the partitioned frame feature files.")
    parser.add_argument('output_folder', type=str,
                        help="The path of the folder in which to put the concatenated files.")

    # Add optional arguments
    parser.add_argument('-t', '--temp-dir-root', type=str, default=os.path.join('.', 'temp'),
                        help="The root dir for the temporary folder in which output files will be stored while they "
                             "are generated. This should be another directory than the output folder, to prevent half "
                             "finished files to end up in the output directory is the script should stop half way "
                             "through a file creation. A file will be moved to the output directory immediately after "
                             "it has been created completely.")

    # Parse arguments
    return parser.parse_args()


def standardize_file(in_feature_file_path, out_feature_file_path):
    xprint("Creating '{}'...".format(out_feature_file_path))
    with open(in_feature_file_path, "rb") as in_feature_file, open(out_feature_file_path, "wb") as out_feature_file:
        # Read features from file
        features = np.reshape(np.fromfile(in_feature_file, dtype=feature_dtype), (-1, feature_vector_length))

        # Zero-center features
        features -= np.mean(features, axis=0, keepdims=True)

        # Make standard deviation become 1
        features /= np.std(features, axis=0, keepdims=True)

        # Write standardized features to file
        features.tofile(out_feature_file)


def main():
    # Parse arguments
    args = parse_args()
    for key, val in get_attributes(args).items():
        xprint("args.{}: {}".format(key, repr(val)))

    process_files(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        file_name_deriver=(lambda x: x),
        file_processor=standardize_file,
        skip_if_exists=skip_if_exists,
        temp_dir_root=args.temp_dir_root)


################################################################################
# SCRIPT ENTRY POINT
################################################################################


if __name__ == '__main__':
    exit(main())
