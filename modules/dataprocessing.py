

################################################################################
# IMPORTS
################################################################################


# Standard imports
import os
import sys
import tempfile
import shutil

# Own imports
# Make Python find user defined modules
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from xprint import xprint


################################################################################
# EXCEPTIONS
################################################################################


class FileNotGeneratedException(Exception):
    pass


################################################################################
# FUNCTIONS
################################################################################


def copy_or_move_file(source, destination, copy):
    # Prepare copy/move
    # Remove any already existing file on the destination path
    if os.path.isfile(destination):
        xprint("Overwriting already fixed file '{}'".format(destination))
        os.remove(destination)
    # Create folder if not already existing
    dir_path = os.path.dirname(destination)
    if not os.path.isdir(dir_path):
        xprint("Creating directory '{}'".format(dir_path))
        os.makedirs(dir_path)

    # Perform copy/move
    if copy:
        # Copy the file
        xprint("Copying '{}' to '{}'...".format(source, destination))
        shutil.copyfile(source, destination)
    else:
        # Move the file
        xprint("Moving '{}' to '{}'...".format(source, destination))
        shutil.move(source, destination)


def copy_file(source, destination):
    copy_or_move_file(source, destination, True)


def move_file(source, destination):
    copy_or_move_file(source, destination, False)


def process_files(
        input_folder,
        output_folder,
        file_name_deriver,
        file_processor,
        file_extensions='',
        skip_if_exists=True,
        temp_dir_root='.',
        produce_output_files=True):
    """Process all files in the folder structure"""

    # Canonicalize arguments
    if isinstance(file_extensions, str):
        file_extensions = [file_extensions]
    file_extensions = [ext.lower() for ext in file_extensions]

    temp_folder_path = None
    if produce_output_files:
        # Create temporary directory for keeping files in while they are created
        if not os.path.isdir(temp_dir_root):
            xprint("Creating directory '{}'".format(temp_dir_root))
            os.makedirs(temp_dir_root)
        temp_folder_path = tempfile.mkdtemp(dir=temp_dir_root)

    try:
        processed_since_before = 0
        processed_now = 0
        for in_file_dir, dirs, in_file_names in os.walk(input_folder, topdown=True):
            # Make sure the folder structure is traversed in alphabetical order
            dirs.sort()
            in_file_names.sort()

            # Get only the part of the file directory path that is relative to input_folder
            rel_file_dir = os.path.relpath(in_file_dir, input_folder)

            # Process relevant files in folder
            for in_file_name in in_file_names:
                # Skip files with incorrect extension
                for ext in file_extensions:
                    if in_file_name.lower().endswith(ext):
                        # Video file extension confirmed, break inner for loop
                        break
                else:
                    # Extension not recognized; continue with next file
                    continue

                # Derive new file name
                out_file_name = file_name_deriver(in_file_name)

                # Derive paths
                in_file_path = os.path.join(in_file_dir, in_file_name)
                temp_file_path = None
                out_file_path = None
                if produce_output_files:
                    temp_file_path = os.path.join(temp_folder_path, out_file_name)
                    out_file_path = os.path.join(output_folder, rel_file_dir, out_file_name)

                if skip_if_exists and os.path.isfile(out_file_path):
                    # File has already been processed, continue with next one
                    processed_since_before += 1
                    xprint("Skipping already processed file '{}'".format(in_file_path))
                    continue

                # Process file. Create in a temporary folder so that the output folder contains no partially created fies.
                xprint()
                xprint("Target: '{}'".format(out_file_path))
                xprint("Processing file '{}'...".format(in_file_path))
                file_processor(in_file_path, temp_file_path)
                if produce_output_files and not os.path.isfile(temp_file_path):
                    raise FileNotGeneratedException(temp_file_path)

                if produce_output_files:
                    # Move file to its final destination
                    move_file(temp_file_path, out_file_path)

                processed_now += 1

        xprint("Files processed now:", processed_now)
        xprint("Files processed since before:", processed_since_before)

    finally:
        if produce_output_files:
            # Remove temporary directory
            shutil.rmtree(temp_folder_path)
