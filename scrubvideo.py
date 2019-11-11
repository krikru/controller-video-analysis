

################################################################################
# IMPORTS
################################################################################


# Standard library imports
import os
import sys
import argparse

# Third party imports
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import cv2
from PySide2.QtCore import Qt, Signal, Slot
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QWidget, QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QSpacerItem, QFormLayout, QSlider, QLabel, QSizePolicy, QComboBox, QLineEdit

# Own imports
# Make Python find own defined modules
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'modules'))
from xprint import xprint, eprint
from debug import get_attributes


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

problem_videos = {  # (<camera number>, <video number>)
    (4, 1),
    (5, 1)
}

feature_data_type = np.float32


################################################################################
# OPTIONS
################################################################################


pc_dimensionality = 100

aspect_ratio = 16.0/9.0
problem_video_bgr_color = [128, 0, 128]

time_plot_minimum_height = 200

#kernel_size = 1001
#kernel_size = 601
kernel_size = 251
unexpectedness_epsilon = 1e-6


################################################################################
# DERIVED VARIABLES
################################################################################


feature_file_lists = {folder: ['.'.join(file_name.split('.')[0:-1]) for file_name in file_list]
                      for folder, file_list in video_file_lists.items()}


################################################################################
# ASSERTS
################################################################################


assert kernel_size % 2 == 1


################################################################################
# CLASSES
################################################################################


class Video:
    def __init__(self, path, start_frame, end_frame):
        self.path = path
        self.start_frame = start_frame
        self.end_frame = end_frame


class Camera:
    def __init__(self, input_folder, camera_number):
        self.input_folder = input_folder
        self.camera_number = camera_number
        self.camera_rel_path = camera_number_to_rel_paths[camera_number]
        self.indexed_videos = []
        #self.indexed_frames_start = 1
        self.indexed_frames_end = 1
        self.frame_rate = None

    def get_bgr_frame(self, frame_number, quick=False):
        # Index enough videos
        while frame_number >= self.indexed_frames_end:
            if len(self.indexed_videos) >= len(video_file_lists[self.camera_rel_path]):
                eprint("ERROR: Frame number is outside of sequence: {} (max is {})".format(frame_number,
                                                                                           self.indexed_frames_end - 1))
                exit(1)
            self._index_next_video()

        # Find right video
        video_index = 0
        while self.indexed_videos[video_index].end_frame <= frame_number:
            video_index += 1

        video_number = video_index + 1
        if (self.camera_number, video_number) in problem_videos:
            return np.array([[problem_video_bgr_color]]).astype('uint8')

        # Open video
        video_capture = cv2.VideoCapture(self.indexed_videos[video_index].path)
        if not video_capture.isOpened():
            eprint("ERROR: Failed open video '{}'".format(self.indexed_videos[video_index].path))
            exit(1)

        # Get frame
        return get_bgr_frame_of_video(video_capture, frame_number - self.indexed_videos[video_index].start_frame, quick=quick)

    def get_last_frame_number(self):
        self._index_all_videos()
        return self.indexed_frames_end - 1

    def _index_next_video(self):
        video_path = os.path.join(self.input_folder,
                                  self.camera_rel_path,
                                  video_file_lists[self.camera_rel_path][len(self.indexed_videos)])

        # Open video
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            eprint("ERROR: Failed open video '{}'".format(video_path))
            exit(1)

        # Get video metadata
        num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        #video_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        #video_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

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

    def _index_all_videos(self):
        while len(video_file_lists[self.camera_rel_path]) > len(self.indexed_videos):
            # There are more videos to index
            self._index_next_video()


class FrameViewer(QLabel):
    def __init__(self, aspect_ratio):
        self.camera = None
        self.bgr_frame = None

        QLabel.__init__(self)
        # Choose a 16:9 resolution
        #self.video_size = QSize(640, 360)  # Half 720p
        self.aspect_ratio = aspect_ratio

        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)
        self.setMinimumSize(1, 1)  # Allow he window to shrink in such a way that the pixmap in the label decreases in size

        self.resizeEvent = self.on_resized

    def display_bgr_frame(self, bgr_frame):
        self.bgr_frame = bgr_frame
        if self.bgr_frame is None:
            self.clear()
            return

        # Resize frame
        resized_bgr_frame = cv2.resize(
            self.bgr_frame,
            (self.width(), self.height()),
            interpolation=cv2.INTER_CUBIC if self.width() > self.bgr_frame.shape[1] else cv2.INTER_AREA)

        # Convert to RGB
        resized_rgb_frame = cv2.cvtColor(resized_bgr_frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        image = QImage(resized_rgb_frame, resized_rgb_frame.shape[1], resized_rgb_frame.shape[0],
                       resized_rgb_frame.strides[0], QImage.Format_RGB888)

        # Display QImage
        #pixmap = QPixmap.fromImage(image)
        #pixmap.detach()
        #self.setPixmap(pixmap)
        self.setPixmap(QPixmap.fromImage(image))

    def set_camera(self, camera):
        self.camera = camera

    def set_frame_number(self, frame_number, quick=False):
        if frame_number is None:
            self.clear()
        else:
            self.display_bgr_frame(self.camera.get_bgr_frame(frame_number, quick=quick))

    def heightForWidth(self, width):
        return round(width / self.aspect_ratio)

    def on_resized(self, event):
        self.display_bgr_frame(self.bgr_frame)




class TimePlot(FigureCanvasQTAgg):
#class TimePlot(QLabel):
    def __init__(self, dpi=100):  # or dpi=72, 96, etc.
        self.figure = Figure(figsize=(640/dpi, 380/dpi), dpi=dpi, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        super().__init__(self.figure)

        self.components = None
        self.variances = None
        self.unexpectednedd = None
        self.component_number = None
        self.frame_number = None

    def set_pc_file(self, pc_file_path):
        assert 'pc{}/'.format(pc_dimensionality) in pc_file_path.replace('\\', '/')
        self.components = np.reshape(np.fromfile(pc_file_path, dtype=feature_data_type), (-1, pc_dimensionality))
        self.variances = np.var(self.components, axis=0, keepdims=True)
        self.unexpectedness = np.sum(np.square(self.components) / (self.variances + unexpectedness_epsilon), axis=1)
        self.plot_component()

    def set_component_number(self, component_number):
        self.component_number = component_number
        self.plot_component()

    def set_frame_number(self, frame_number):
        self.frame_number = frame_number
        self.plot_component()

    def plot_component(self):
        self.figure.clear()

        if self.components is None or self.component_number is None or self.frame_number is None:
            self.figure.canvas.draw()
            return

        ax = self.figure.add_subplot(111)

        # Get shape of data
        num_frames, num_components = self.components.shape

        assert num_components == pc_dimensionality

        kernel = np.array([1] * kernel_size) / kernel_size

        #fig = plt.figure(figsize=(20, 16))
        #fig.canvas.set_window_title(title)
        #fig.suptitle("{} principal components".format(title), fontsize=16)

        # Plot individual principal components as time series
        #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.7)

        ax.axvline(x=self.frame_number, color='red')

        component_index = self.component_number - 1
        ax.plot(np.arange(num_frames - kernel_size + 1) + 1 + (kernel_size - 1) / 2,
                     np.convolve(self.components[:, component_index] if component_index >= 0 else self.unexpectedness, kernel, mode='valid'))
        ax.grid()
        #plt.title("Principal component #{}".format(self.component_index + 1), fontsize=10)
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Component value")

        self.figure.canvas.draw()

        #self.plot()
        #plt.close(fig)


class VideoScrubberWindow(QMainWindow):
    def __init__(self, video_folder, pc_folder, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        # Constants
        self.video_folder = video_folder
        self.pc_folder = pc_folder

        # Variables
        self.camera_number = None
        self.camera = None
        self.component_number = None

        main_layout = QVBoxLayout()
        #self.layout().setSizeConstraint(QLayout.SetFixedSize)
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.setWindowTitle("Video scrubber")

        # Create a canvas for the frame
        self.frame_viewer = FrameViewer(aspect_ratio=aspect_ratio)
        main_layout.addWidget(self.frame_viewer)

        # Create a slider
        self.slider = QSlider(orientation=Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.set_frame_number)
        main_layout.addWidget(self.slider)

        # Create a canvas for the time plot
        self.time_plot = TimePlot()
        self.time_plot.setMinimumHeight(time_plot_minimum_height)
        main_layout.addWidget(self.time_plot)

        choice_outer_layout = QHBoxLayout()
        main_layout.addLayout(choice_outer_layout)

        choice_layout = QFormLayout()
        choice_outer_layout.addLayout(choice_layout)

        # Drop down menu for choosing camera
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([str(number) for number in sorted(list(camera_number_to_rel_paths.keys()))])
        self.camera_combo.currentTextChanged.connect(self.set_camera_number)
        choice_layout.addRow("Camera number:", self.camera_combo)

        # Text box for choosing component:
        self.pc_text_box = QLineEdit()
        self.pc_text_box.textEdited.connect(self.set_component_number)
        choice_layout.addRow("Principal component number (1--{}), or 0 for unexpectedness:".format(pc_dimensionality), self.pc_text_box)

        choice_horizontal_spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        choice_outer_layout.addSpacerItem(choice_horizontal_spacer)

        self.set_camera_number(None)

    def set_camera_number(self, camera_number):
        self.camera_number = camera_number
        if self.camera_number is not None:
            self.camera_combo.setCurrentText(str(self.camera_number))
            self.camera_number = int(self.camera_number)
            self.camera = Camera(self.video_folder, self.camera_number)
            self.frame_viewer.set_camera(self.camera)
            self.slider.setEnabled(True)
            self.slider.setRange(1, self.camera.get_last_frame_number())
            self.set_frame_number(int(self.camera.get_last_frame_number()/2))
            camera_rel_path = camera_number_to_rel_paths[self.camera_number]
            self.time_plot.set_pc_file(self._get_pc_file_path(camera_rel_path))
            #self.time_plot.set_component_number(1)
        else:
            self.slider.setEnabled(False)

    def set_component_number(self, component_number):
        self.component_number = None if component_number is None else int(component_number)
        if self.component_number is not None:
            self.pc_text_box.setText(str(self.component_number))
            if not 0 <= self.component_number <= pc_dimensionality:
                self.component_number = None
        self.time_plot.set_component_number(self.component_number)

    def _get_pc_file_path(self, camera_rel_path):
        # Derive output file name
        pc_file_name = camera_rel_path
        while os.path.dirname(pc_file_name):
            pc_file_name = os.path.dirname(pc_file_name)
        return os.path.join(self.pc_folder, pc_file_name)

    @Slot()
    def set_frame_number(self, frame_number, quick=True):
        if self.camera_number is None:
            assert frame_number is None
        else:
            assert 1 <= frame_number <= self.camera.get_last_frame_number()

        #print("Setting frame number", frame_number)

        self.frame_viewer.set_frame_number(frame_number, quick=quick)
        self.time_plot.set_frame_number(frame_number)
        if frame_number is not None:
            self.slider.setValue(frame_number)


################################################################################
# FUNCTIONS
################################################################################


def parse_args():
    """Parse and return command line arguments"""

    # Create parser
    parser = argparse.ArgumentParser(description="Desktop application that lets you 'scrub' through camera recordings")

    # Add positional arguments
    parser.add_argument('video_folder', type=str,
                        help="The path of the folder containing the video file sequences.")
    parser.add_argument('pc_folder', type=str,
                        help="The path of the folder containing the principal component files.")

    # Add optional arguments
    #parser.add_argument('-f', '--frames', action='append', nargs='+', type=int,
    #                    help="Specify a camera number [1 to 5] followed by a sequence of frame numbers for that "
    #                         "camera. This option can be used multiple times to extract frames from more than one "
    #                         "camera.")

    # Parse arguments
    return parser.parse_args()


def get_bgr_frame_of_video(video_capture, target_position, quick=False):
    # Find right "chunk" of frames
    if True:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_position)
    else:
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.set(cv2.CAP_PROP_POS_MSEC, target_position * (1000.0 /frame_rate))

    current_pos = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
    if target_position != current_pos:
        num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Out of totally {} frames:".format(num_frames))
        print("Target frame:", target_position)
        print("Reached frame:", current_pos)
        print("Frame diff:", target_position - current_pos)

    if not quick:
        # Progress one frame at a time until at the correct frame
        while current_pos < target_position:
            # Go to next frame
            previous_pos = current_pos
            video_capture.grab()
            current_pos = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            if current_pos <= previous_pos:
                assert current_pos >= previous_pos
                break

    # Check if the frame number is correct now
    if current_pos == target_position:
        ret, bgr_image = video_capture.retrieve()
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

    # Create a Qt application
    app = QApplication(sys.argv)

    # Create a window
    window = VideoScrubberWindow(args.video_folder, args.pc_folder)

    window.set_camera_number(4)
    window.set_frame_number(203000)
    window.set_component_number(10)

    # Show window
    window.show()

    # Enter Qt application main loop
    return app.exec_()

################################################################################
# SCRIPT ENTRY POINT
################################################################################


if __name__ == '__main__':
    exit(main())
