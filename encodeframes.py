

################################################################################
# IMPORTS
################################################################################


# Standard library imports
import os
import sys
import ctypes
import argparse

# Third party imports
from skimage.io import imread
import numpy as np
import itertools
from tqdm import tqdm

from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense, Lambda
from keras.backend import l2_normalize
from keras.applications.imagenet_utils import decode_predictions

import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'modules',
                             'efficientnet'))
import efficientnet
from efficientnet.keras import EfficientNetBn, preprocess_input
from efficientnet.preprocessing import center_crop_and_resize


# Own imports
# Make Python find own defined modules
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'modules'))
from xprint import xprint, eprint
from debug import get_attributes
from dataprocessing import process_files


################################################################################
# CONSTANTS
################################################################################


rel_panda_path = os.path.join('..', 'misc', 'panda.jpg')

in_extensions = ('.mp4', '.mov', '.mts')
out_extension = ''

expected_video_resolutions = [(2304, 1296), (1920, 1080)]
expected_frame_rates = [25.0, 29.97002997002997, 50.0]


################################################################################
# OPTIONS
################################################################################


# Image
show_image = True
#show_image = False

#frame_display_stride = 1
frame_display_stride = 5
#frame_display_stride = 10

# Deep learning
efficientnet_version = 0  # Integer number between 0 and 7 specifying the version of EfficientNet (6 and 7 don't have any pretrained weights)
#use_original_classifier = True
use_original_classifier = False
aspect_ratio = 16/9
crop_padding = 0
normalize_feature_vectors = True
desired_batch_size = 30

# Still images
#use_images = True
use_images = False

# Webcam
use_webcam = False
# See https://stackoverflow.com/a/53405720/1070480 and https://docs.opencv.org/3.4.3/d8/dfe/classcv_1_1VideoCapture.html
#cam_api_reference = cv2.CAP_ANY  # == 0 (default value)
cam_api_reference = cv2.CAP_DSHOW  # == 700

# Videos
#rel_path_to_videos = [os.path.join('Bromma videoDATA',
#                                   'CAMERA 2 left controller',
#                                   'ZO{:02d}0003.MOV'.format(n+1)) for n in range(12)]

# Saved results
save_results = True
features_file_path = 'features.txt'
feature_dtype = np.single  # == np.float32 (float becomes np.float64)


################################################################################
# DERIVED VARIABLES
################################################################################


model_name = 'efficientnet-b{}'.format(efficientnet_version)

input_size = [
    224,
    240,
    260,
    300,
    380,
    456,
    528,
    600
    ][efficientnet_version]

efficientnet_module_path = os.path.dirname(efficientnet.__file__)
panda_path = os.path.join(efficientnet_module_path, rel_panda_path)

feature_len = len(np.array([0]).astype(feature_dtype).tobytes())


################################################################################
# ASSERTS
################################################################################


assert isinstance(desired_batch_size, int) and desired_batch_size >= 1
assert feature_len == 4
assert not (use_original_classifier and save_results)


################################################################################
# LAYERS
################################################################################


def FeatureVectorL2Normalization(name=None):
    return Lambda(lambda x: l2_normalize(x, axis=1), name=name)


################################################################################
# FUNCTIONS
################################################################################


def parse_args():
    """Parse and return command line arguments"""

    # Create parser
    parser = argparse.ArgumentParser(description="Encode airport controller video frames using EfficientNet")

    # Add positional arguments
    parser.add_argument('video_folder', type=str,
                        help="The path of the folder containing the videos to be processed")
    parser.add_argument('output_folder', type=str,
                        help="The path of the folder in which to put the representation vectors. Already existing "
                             "files won't be removed or regenerated.")

    # Add optional arguments
    #parser.add_argument('-l', '--log-process', action='store_true',
    #                    help="Whether to create a log file")
    parser.add_argument('-t', '--temp-dir-root', type=str, default=os.path.join('.', 'temp'),
                        help="The root dir for the temporary folder in which output files will be stored while they "
                             "are generated. This should be another directory than the output folder, to prevent half "
                             "finished files to end up in the output directory is the script should stop half way "
                             "through a file creation. A file will be moved to the output directory immediately after "
                             "it has been created completely.")

    # Parse arguments
    return parser.parse_args()


def prevent_window_scaling():
    # Query DPI Awareness (Windows 10 and 8)
    awareness = ctypes.c_int()
    xprint("awareness.value:", awareness.value)
    errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
    xprint("errorCode:", errorCode)

    # Set DPI Awareness  (Windows 10 and 8)
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
    xprint("errorCode:", errorCode)
    # the argument is the awareness level, which can be 0, 1 or 2:
    # for 1-to-1 pixel control I seem to need it to be non-zero (I'm using level 2)

    # Set DPI Awareness  (Windows 7 and Vista)
    success = ctypes.windll.user32.SetProcessDPIAware()
    xprint("success:", success)
    # behaviour on later OSes is undefined, although when I run it on my Windows 10 machine, it seems to work with effects identical to SetProcessDpiAwareness(1)


def cv2_window_open(name):
    return (sys.platform == 'linux' and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) == 1 or
            sys.platform == 'win32' and cv2.getWindowProperty(name, 0) != -1)


def get_model(input_h, input_w):
    # Load model
    xprint("Loading model...")
    # include_top: whether to include the fully-connected layer at the top of the network
    input_tensor = Input((input_h, input_w, 3))
    base_model = EfficientNetBn(n=efficientnet_version, weights='imagenet', input_tensor=input_tensor,
                                include_top=use_original_classifier)
    xprint("Loading model finished.")

    if use_original_classifier:
        return base_model
    else:
        # Transfer learning method adapted from https://www.dlology.com/blog/transfer-learning-with-efficientnet/
        x = base_model.output
        gap_layer = GlobalAveragePooling2D(name="gap")(x)
        gmp_layer = GlobalMaxPooling2D(name="gmp")(x)
        if normalize_feature_vectors:
            norm_gap_layer = FeatureVectorL2Normalization(name="norm_gap")(gap_layer)
            norm_gmp_layer = FeatureVectorL2Normalization(name="norm_gmp")(gmp_layer)
            return Model(inputs=base_model.input, outputs=[norm_gap_layer, norm_gmp_layer])
        else:
            return Model(inputs=base_model.input, outputs=[gap_layer, gmp_layer])


def classify_panda(model, image_shape):
    rgb_panda_image = imread(panda_path)
    if show_image:
        bgr_panda_image = cv2.cvtColor(rgb_panda_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('panda_image', bgr_panda_image)

    # preprocess input
    cropped_rgb_panda_image = center_crop_and_resize(rgb_panda_image, image_size=image_shape)
    if show_image:
        cropped_bgr_panda_image = cv2.cvtColor(cropped_rgb_panda_image.astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imshow('cropped_panda_image', cropped_bgr_panda_image)
    preprocessed_rgb_panda_image = preprocess_input(cropped_rgb_panda_image)
    batch = np.expand_dims(preprocessed_rgb_panda_image, 0)

    # make prediction and decode
    predictions = model.predict(batch)
    xprint(decode_predictions(predictions))

    if show_image:
        while cv2_window_open('panda_image') and cv2_window_open('cropped_panda_image'):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def test_webcams(model, image_shape):
    for cam_idx in itertools.count():
        cap = cv2.VideoCapture(cam_idx + cam_api_reference)
        if not cap.isOpened():
            break

        ret, bgr_frame = cap.read()
        xprint("Cam {} image shape: {}".format(cam_idx, bgr_frame.shape))
        while True:
            ret, bgr_frame = cap.read()
            cv2.imshow('frame', bgr_frame)

            # Use neural network
            cam_batch_size = 1
            verbose = False

            # print("model.input_shape:", model.input_shape)
            # input_size = model.input_shape[2]
            cropped_bgr_frame = center_crop_and_resize(bgr_frame, image_size=image_shape)

            cv2.imshow('cropped_frame', cropped_bgr_frame.astype('uint8'))
            xprint("cropped_bgr_frame.shape:", cropped_bgr_frame.shape)
            cropped_rgb_frame = cv2.cvtColor(cropped_bgr_frame.astype('uint8'), cv2.COLOR_BGR2RGB)
            batch = preprocess_input(cropped_rgb_frame.reshape((-1,) + cropped_rgb_frame.shape))

            prediction = model.predict(batch, batch_size=cam_batch_size, verbose=verbose)
            if use_original_classifier:
                for decoded_frame_prediction in decode_predictions(prediction):
                    xprint(decoded_frame_prediction)
            else:
                if not isinstance(prediction, list):
                    prediction = [prediction]

                print()
                for output_head in prediction:
                    for frame_output in output_head:
                        print(frame_output.shape, frame_output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if not (cv2_window_open('frame') and cv2_window_open('cropped_frame')):
                break

        cap.release()
        cv2.destroyAllWindows()


def get_frame_batches(video, image_shape, desired_batch_size):
    video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    if (video_w, video_h) not in expected_video_resolutions or frame_rate not in expected_frame_rates:
        xprint("Resolution: {}x{}".format(video_w, video_h))
        xprint("Frame rate:", frame_rate)
        raise Exception("Unexpected video property detected")
    frame_idx = 0
    batch = []
    while True:
        ret, bgr_frame = video.read()
        if not ret:
            break
        else:
            if show_image and frame_idx % frame_display_stride == 0:
                image_scale = 0.5
                new_width, new_height = bgr_frame.shape[1] * image_scale, bgr_frame.shape[0] * image_scale
                resized_bgr_frame = cv2.resize(bgr_frame, (int(new_width), int(new_height)))
                cv2.imshow('frame', resized_bgr_frame)

            cropped_bgr_frame = center_crop_and_resize(bgr_frame, image_size=image_shape, crop_padding=crop_padding)

            if show_image:
                cv2.imshow('cropped_frame', cropped_bgr_frame.astype('uint8'))

            cropped_rgb_frame = cv2.cvtColor(cropped_bgr_frame.astype('uint8'), cv2.COLOR_BGR2RGB)
            batch.append(preprocess_input(cropped_rgb_frame))

            if show_image:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise Exception("q key pressed during video processing")

                if not (cv2_window_open('frame') and cv2_window_open('cropped_frame')):
                    raise Exception("Window closed during video processing")

            if len(batch) == desired_batch_size:
                yield np.asarray(batch)
                batch = []
        frame_idx += 1

    if len(batch) > 0:
        yield np.asarray(batch)

    cv2.destroyAllWindows()


def process_video(video_path, feature_file_path, model, image_shape, verbose=False):
    xprint("Creating '{}'...".format(feature_file_path))
    with open(feature_file_path, "wb") as features_file:
        # Open video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            eprint("ERROR: Failed open video '{}'".format(video_path))
            exit(1)

        # Get video metadata
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = video.get(cv2.CAP_PROP_FPS)

        # Print metadata
        xprint("Resolution: {}x{}".format(video_w, video_h))
        xprint("Duration:", num_frames / frame_rate)
        xprint("Number of frames:", num_frames)
        xprint("Frame rate:", frame_rate)

        # Process video
        num_batches = (num_frames + desired_batch_size - 1) // desired_batch_size
        for batch in tqdm(
                get_frame_batches(video, image_shape, desired_batch_size),
                desc=video_path,
                total=num_batches,
                file=sys.stdout):
            current_batch_size = len(batch)
            predictions = model.predict(np.asarray(batch), batch_size=current_batch_size, verbose=verbose)
            if use_original_classifier:
                for decoded_frame_prediction in decode_predictions(predictions):
                    xprint(decoded_frame_prediction)
            else:
                if not isinstance(predictions, list):
                    predictions = [predictions]
                assert len(predictions) >= 1  # At least one output head
                assert len(predictions[0]) == current_batch_size  # Exactly one output per input
                for frame_predictions in zip(*predictions):
                    feature_vector = np.concatenate(frame_predictions)
                    if save_results:
                        feature_byte_string = feature_vector.astype(feature_dtype).tobytes()
                        features_file.write(feature_byte_string)


def get_output_name(input_name):
    lower_input_name = input_name.lower()
    for ext in in_extensions:
        if lower_input_name.endswith(ext):
            return input_name[:-len(ext)]
    raise Exception("File has no or unknown file extension: '{}'".format(lower_input_name))


def main():
    # Parse arguments
    args = parse_args()
    for key, val in get_attributes(args).items():
        xprint("args.{}: {}".format(key, repr(val)))

    # Prevent window from being scaled by OS, such that pixels in displayed images correspond well to pixels on the
    # screen
    if sys.platform == 'win32':
        prevent_window_scaling()

    # Determine input size to network
    if not use_original_classifier:
        input_h = input_size
        input_w = int(input_size * aspect_ratio)
    else:
        input_h = input_w = input_size
    image_shape = (input_h, input_w)

    # Create the neural network
    model = get_model(input_h, input_w)

    # Print model information
    xprint(model.summary())

    # Use the neural network in different sources
    if use_images and use_original_classifier:
        classify_panda(model, image_shape)

    if use_webcam and show_image:
        test_webcams(model, image_shape)

    process_files(
        input_folder=args.video_folder,
        output_folder=args.output_folder,
        file_name_deriver=get_output_name,
        file_processor=lambda in_path, out_path: process_video(in_path, out_path, model, image_shape, verbose=False),
        file_extensions=in_extensions,
        skip_if_exists=True,
        temp_dir_root=args.temp_dir_root)


################################################################################
# SCRIPT ENTRY POINT
################################################################################


if __name__ == '__main__':
    exit(main())
