

################################################################################
# IMPORTS
################################################################################


# Standard library imports
import os
import sys
import argparse
from random import randint
from collections import defaultdict

# Third party imports
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2

# Own imports
# Make Python find own defined modules
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'modules'))
from xprint import xprint, eprint
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

camera_number_to_rel_paths = {
    1: os.path.join('CAMERA 1 flight strips', '100_ZOOM'),
    2: 'CAMERA 2 left controller',
    3: 'CAMERA 3 right controller',
    4: 'CAMERA 4',
    5: 'CAMERA 5'
}


################################################################################
# OPTIONS
################################################################################


pause_between_frames = True
save_images = False


################################################################################
# DERIVED VARIABLES
################################################################################


feature_file_lists = {folder: ['.'.join(file_name.split('.')[0:-1]) for file_name in file_list]
                      for folder, file_list in video_file_lists.items()}


################################################################################
# CLASSES
################################################################################


class Video:
    def __init__(self, path, start_frame, end_frame):
        self.path = path
        self.start_frame = start_frame
        self.end_frame = end_frame


class Camera:
    def __init__(self, input_folder, camera_rel_path):
        self.input_folder = input_folder
        self.camera_rel_path = camera_rel_path
        self.indexed_videos = []
        #self.indexed_frames_start = 1
        self.indexed_frames_end = 1
        self.frame_rate = None

    def index_next_video(self):
        video_path = os.path.join(self.input_folder,
                                  self.camera_rel_path,
                                  video_file_lists[self.camera_rel_path][len(self.indexed_videos)])

        # Open video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            eprint("ERROR: Failed open video '{}'".format(video_path))
            exit(1)

        # Get video metadata
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        #video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = video.get(cv2.CAP_PROP_FPS)

        if self.frame_rate is None:
            self.frame_rate = frame_rate
        else:
            if frame_rate != self.frame_rate:
                eprint("ERROR: Frame rate in '{}' differs from previous videos in sequence!".format(video_path))
                exit(1)

        # Add video to index
        new_frames_end = self.indexed_frames_end + num_frames
        self.indexed_videos.append(Video(video_path, self.indexed_frames_end, new_frames_end))
        self.indexed_frames_end = new_frames_end

    def get_bgr_frame(self, frame_number):
        # Index enough videos
        while frame_number >= self.indexed_frames_end:
            if len(self.indexed_videos) >= len(video_file_lists[self.camera_rel_path]):
                eprint("ERROR: Frame number is outside of sequence: {} (max is {})".format(frame_number,
                                                                                           self.indexed_frames_end - 1))
                exit(1)
            self.index_next_video()

        # Find right video
        video_index = 0
        while self.indexed_videos[video_index].end_frame <= frame_number:
            video_index += 1

        # Open video
        video = cv2.VideoCapture(self.indexed_videos[video_index].path)
        if not video.isOpened():
            eprint("ERROR: Failed open video '{}'".format(self.indexed_videos[video_index].path))
            exit(1)

        # Get frame
        return get_bgr_frame_of_video(video, frame_number - self.indexed_videos[video_index].start_frame)


################################################################################
# FUNCTIONS
################################################################################


def parse_args():
    """Parse and return command line arguments"""

    # Create parser
    parser = argparse.ArgumentParser(description="Get specific frames for specific camera")

    # Add positional arguments
    parser.add_argument('input_folder', type=str,
                        help="The path of the folder containing the video file sequences.")
    parser.add_argument('output_folder', type=str,
                        help="The path of the folder in which to save the images.")
    #parser.add_argument('output_folder', type=str,
    #                    help="The path of the folder in which to put the concatenated files.")

    # Add optional arguments
    parser.add_argument('-f', '--frames', action='append', nargs='+', type=int,
                        help="Specify a camera number [1 to 5] followed by a sequence of frame numbers for that "
                             "camera. This option can be used multiple times to extract frames from more than one "
                             "camera.")

    # Parse arguments
    return parser.parse_args()


def cv2_window_open(name):
    return (sys.platform == 'linux' and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) == 1 or
            sys.platform == 'win32' and cv2.getWindowProperty(name, 0) != -1)


def get_bgr_frame_of_video(video, position):
    # Find right "chunk" of frames
    video.set(cv2.CAP_PROP_POS_FRAMES, position)
    current_pos = video.get(cv2.CAP_PROP_POS_FRAMES)

    # Progress one frame at a time until at the correct frame
    while current_pos < position:
        # Go to next frame
        video.grab()
        current_pos = video.get(cv2.CAP_PROP_POS_FRAMES)

    # Check if the frame number is correct now
    if current_pos == position:
        ret, bgr_image = video.retrieve()
        if ret:
            # Successfully got frame
            return bgr_image

    # Failed getting frame
    return None


def main():
    # Parse arguments
    args = parse_args()
    for key, val in get_attributes(args).items():
        xprint("args.{}: {}".format(key, repr(val)))

    # Check arguments
    if args.frames is None:
        xprint("No frames specified.")

    camera_frame_numbers = defaultdict(list)

    for camera_frame_seq in args.frames:
        camera_number = camera_frame_seq[0]
        frame_numbers = camera_frame_seq[1:]

        if camera_number not in camera_number_to_rel_paths:
            eprint("ERROR: Camera number invalid, should be any of {}; received {}".format(
                ", ".join([str(num) for num in sorted(camera_number_to_rel_paths.keys())]),
                camera_number))
            exit(1)

        for frame_number in frame_numbers:
            if frame_number < 1:
                eprint("ERROR: Frame numbers must be positive; received", frame_number)
                exit(1)
            camera_frame_numbers[camera_number].append(frame_number)

    # Display frames
    cameras = {}
    for camera_number, frame_numbers in camera_frame_numbers.items():
        if camera_number not in cameras:
            cameras[camera_number] = Camera(args.input_folder, camera_number_to_rel_paths[camera_number])
        camera = cameras[camera_number]

        pause_between_frames = True

        #frame_numbers = [1, 2, 3, 316124, 316125]
        frame_numbers = frame_numbers if camera_number != 4 else range(50000, 52000, 5)
        for frame_number in frame_numbers:
            pause_between_frames = (camera_number != 4)
            bgr_image = camera.get_bgr_frame(frame_number)
            #bgr_image = camera.get_bgr_frame(randint(1, 316125))

            #grb_image = cv2.cvtColor(bgr_image.astype('uint8'), cv2.COLOR_BGR2RGB)
            #plt.imshow(grb_image)
            #plt.show()

            #image_scale = 0.5
            #new_width, new_height = bgr_frame.shape[1] * image_scale, bgr_frame.shape[0] * image_scale
            #resized_bgr_frame = cv2.resize(bgr_frame, (int(new_width), int(new_height)))

            bgr_image_plus_text = bgr_image[:]

            margin = 30
            bottomLeftCornerOfText = (margin, bgr_image.shape[0] - margin)
            for foreground in [False, True]:
                cv2.putText(
                    img=bgr_image,
                    text="Frame {}".format(frame_number),
                    org=bottomLeftCornerOfText,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5,
                    color=(255, 255, 255) if foreground else (0, 0, 0),
                    thickness=2 if foreground else 5,
                    lineType=2,
                    bottomLeftOrigin=False)

            camera_name = camera_number_to_rel_paths[camera_number]
            while os.path.dirname(camera_name):
                camera_name = os.path.dirname(camera_name)
            camera_name = camera_name.replace(' ', '_')

            if save_images:
                cv2.imwrite(os.path.join(args.output_folder, "{}_frame_{}.png".format(camera_name, frame_number)),
                            bgr_image)
            cv2.imshow('frame', bgr_image)
            while cv2_window_open('frame'):
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if not pause_between_frames:
                    break
        cv2.destroyAllWindows()

################################################################################
# SCRIPT ENTRY POINT
################################################################################


if __name__ == '__main__':
    exit(main())
