

################################################################################
# IMPORTS
################################################################################


# Standard library imports
import os
import sys
import argparse
import tempfile
import shutil

# Third party imports
from tqdm import tqdm

# Own imports
# Make Python find own defined modules
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'modules'))
from xprint import xprint
from debug import get_attributes
from dataprocessing import move_file


################################################################################
# CONSTANTS
################################################################################


video_file_lists = {
    os.path.join('CAMERA 1 flight strips', '100_ZOOM'): ['ZO0{}0004.MOV'.format(num + 1) for num in range(5)],
    'CAMERA 2 left controller': ['ZO{:02d}0003.MOV'.format(num + 1) for num in range(12)],
    'CAMERA 3 right controller': ['S1750002.MP4', 'S1750003.MP4', 'S1750004.MP4', 'S1750005.MP4', 'S1750006.MP4'],
    'CAMERA 4': ['000{:02d}.MTS'.format(num + 1) for num in range(17)],
    'CAMERA 5': ['000{:02d}.MTS'.format(num + 1) for num in range(17)]
}


################################################################################
# OPTIONS
################################################################################


skip_if_exists = True


################################################################################
# DERIVED VARIABLES
################################################################################


feature_file_lists = {folder: ['.'.join(file_name.split('.')[0:-1]) for file_name in file_list]
                      for folder, file_list in video_file_lists.items()}


################################################################################
# FUNCTIONS
################################################################################


def parse_args():
    """Parse and return command line arguments"""

    # Create parser
    parser = argparse.ArgumentParser(description="Concatenate the frame encoding file partitions for each camera")

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


def main():
    # Parse arguments
    args = parse_args()
    for key, val in get_attributes(args).items():
        xprint("args.{}: {}".format(key, repr(val)))

    # Create temporary folder
    if not os.path.isdir(args.temp_dir_root):
        xprint("Creating directory '{}'".format(args.temp_dir_root))
        os.makedirs(args.temp_dir_root)
    temp_folder_path = tempfile.mkdtemp(dir=args.temp_dir_root)
    xprint("Directory '{}' created.".format(temp_folder_path))

    try:
        # Concatenate files
        for short_rel_in_folder, file_list in feature_file_lists.items():
            # Derive output file name
            out_file_name = short_rel_in_folder
            while os.path.dirname(out_file_name):
                out_file_name = os.path.dirname(out_file_name)

            # Derive paths
            in_file_dir = os.path.join(args.input_folder, short_rel_in_folder)
            temp_file_path = os.path.join(temp_folder_path, out_file_name)
            out_file_path = os.path.join(args.output_folder, out_file_name)

            if skip_if_exists and os.path.isfile(out_file_path):
                # File has already been concatenated, continue with next file list
                xprint("Skipping already concatenated files in '{}'".format(in_file_dir))
                continue

            xprint("Creating '{}'...".format(temp_file_path))
            with open(temp_file_path, 'wb') as out_file:
                for in_file_name in tqdm(file_list, desc=out_file_name, file=sys.stdout):
                    in_file_path = os.path.join(in_file_dir, in_file_name)
                    with open(in_file_path, 'rb') as in_file:
                        shutil.copyfileobj(in_file, out_file)

            # Move file to its final destination
            move_file(temp_file_path, out_file_path)
    finally:
        # Remove temporary directory
        shutil.rmtree(temp_folder_path)
        xprint("Directory '{}' removed.".format(temp_folder_path))


################################################################################
# SCRIPT ENTRY POINT
################################################################################


if __name__ == '__main__':
    exit(main())
