"""
GUI for launching a study session.

Enters participant ID + condition, then starts/stops audio, video, and
rosbag recording together.
"""

import datetime
import os
import signal
import subprocess
import sys
import threading

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# --- Study configuration -----------------------------------------------
# Edit these to match the actual 2x2 design.
FACTOR_A_NAME = 'Group'
FACTOR_A_LEVELS = ['Control', 'Experiment']
FACTOR_B_NAME = 'Ordering'
FACTOR_B_LEVELS = ['Environment A', 'Environment B']

# Topics captured in the rosbag alongside the GUI's own audio/video capture.
BAG_TOPICS = ['/camera/image_raw', '/joint_states']

DEFAULT_OUTPUT_DIR = os.path.expanduser('~/investigating_peer_recordings')
DEFAULT_CAMERA_DEVICE = '/dev/video0'
DEFAULT_AUDIO_DEVICE = 'default'
VIDEO_FPS = 30.0


class VideoRecorder(threading.Thread):
    """Grabs frames from a camera device and writes them to an mp4 file."""

    def __init__(self, device, out_path):
        super().__init__(daemon=True)
        self.device = device
        self.out_path = out_path
        self._stop_event = threading.Event()
        self.error = None

    def run(self):
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            self.error = f'Could not open camera device {self.device}'
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.out_path, fourcc, VIDEO_FPS, (width, height))

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                writer.write(frame)
        finally:
            cap.release()
            writer.release()

    def stop(self):
        self._stop_event.set()


class SessionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Study Session Recorder')

        self.video_recorder = None
        self.audio_proc = None
        self.bag_proc = None
        self.session_dir = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        form = QFormLayout()

        self.participant_id_edit = QLineEdit()
        self.participant_id_edit.setPlaceholderText('e.g. P01')
        form.addRow('Participant ID:', self.participant_id_edit)

        self.factor_a_combo = QComboBox()
        self.factor_a_combo.addItems(FACTOR_A_LEVELS)
        form.addRow(f'{FACTOR_A_NAME}:', self.factor_a_combo)

        self.factor_b_combo = QComboBox()
        self.factor_b_combo.addItems(FACTOR_B_LEVELS)
        form.addRow(f'{FACTOR_B_NAME}:', self.factor_b_combo)

        output_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit(DEFAULT_OUTPUT_DIR)
        browse_btn = QPushButton('Browse...')
        browse_btn.clicked.connect(self._browse_output_dir)
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(browse_btn)
        form.addRow('Output directory:', output_row)

        self.camera_device_edit = QLineEdit(DEFAULT_CAMERA_DEVICE)
        form.addRow('Camera device:', self.camera_device_edit)

        self.audio_device_edit = QLineEdit(DEFAULT_AUDIO_DEVICE)
        form.addRow('Audio device:', self.audio_device_edit)

        layout.addLayout(form)

        self.status_label = QLabel('Idle')
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        self.start_button = QPushButton('Start Recording')
        self.start_button.clicked.connect(self._start_recording)
        self.stop_button = QPushButton('Stop Recording')
        self.stop_button.clicked.connect(self._stop_recording)
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        layout.addLayout(button_row)

        self.setLayout(layout)

    def _browse_output_dir(self):
        chosen = QFileDialog.getExistingDirectory(
            self, 'Select output directory', self.output_dir_edit.text())
        if chosen:
            self.output_dir_edit.setText(chosen)

    def _set_inputs_enabled(self, enabled):
        self.participant_id_edit.setEnabled(enabled)
        self.factor_a_combo.setEnabled(enabled)
        self.factor_b_combo.setEnabled(enabled)
        self.output_dir_edit.setEnabled(enabled)
        self.camera_device_edit.setEnabled(enabled)
        self.audio_device_edit.setEnabled(enabled)

    def _start_recording(self):
        participant_id = self.participant_id_edit.text().strip()
        if not participant_id:
            QMessageBox.warning(
                self, 'Missing participant ID', 'Enter a participant ID before starting.')
            return

        factor_a = self.factor_a_combo.currentText()
        factor_b = self.factor_b_combo.currentText()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f'{participant_id}_{factor_a}_{factor_b}_{timestamp}'.replace(' ', '-')
        session_dir = os.path.join(self.output_dir_edit.text().strip(), session_name)

        try:
            os.makedirs(session_dir, exist_ok=False)
        except OSError as exc:
            QMessageBox.critical(self, 'Could not create session folder', str(exc))
            return

        self.session_dir = session_dir

        # Video
        video_path = os.path.join(session_dir, 'video.mp4')
        self.video_recorder = VideoRecorder(self.camera_device_edit.text().strip(), video_path)
        self.video_recorder.start()

        # Audio
        audio_path = os.path.join(session_dir, 'audio.wav')
        self.audio_proc = subprocess.Popen(
            [
                'arecord',
                '-D',
                self.audio_device_edit.text().strip(),
                '-f',
                'cd',
                '-t',
                'wav',
                audio_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Rosbag
        bag_path = os.path.join(session_dir, 'rosbag')
        self.bag_proc = subprocess.Popen(
            ['ros2', 'bag', 'record', '-o', bag_path] + BAG_TOPICS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self._set_inputs_enabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f'Recording -> {session_dir}')

    def _stop_recording(self):
        if self.video_recorder is not None:
            self.video_recorder.stop()
            self.video_recorder.join(timeout=5)
            if self.video_recorder.error:
                QMessageBox.warning(self, 'Video recording error', self.video_recorder.error)
            self.video_recorder = None

        for proc in (self.audio_proc, self.bag_proc):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
        self.audio_proc = None
        self.bag_proc = None

        self._set_inputs_enabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText(f'Saved to {self.session_dir}' if self.session_dir else 'Idle')

    def closeEvent(self, event):
        if self.stop_button.isEnabled():
            reply = QMessageBox.question(
                self,
                'Recording in progress',
                'A recording is still running. Stop it and exit?',
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self._stop_recording()
        event.accept()


def main():
    app = QApplication(sys.argv)
    gui = SessionGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
