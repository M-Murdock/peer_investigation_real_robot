#!/usr/bin/env python3
"""
study_control_panel.py — browser control panel for running study sessions.

Serves a static page (web/study_control_panel/index.html) with Home,
participant-ID entry, env A/B selection, and Start/Stop Study buttons.
Follows gen3-lite-ros2/src/joy_web_bridge.py's Tornado pattern (already a
workspace dependency, chosen there because rosbridge_library fails to
import in this workspace) — but needs no WebSocket and no rclpy at all,
since every action here is triggering a `ros2` CLI subprocess, not
publishing a live topic.

The real safety interlocks (only one study at a time, no homing while a
study runs, homing required before a study can start, input validation)
live server-side in the handlers below, not just as disabled buttons in
the page's JS — a stale browser tab or a stray `fetch` call must not be
able to bypass them. This panel computes nothing new about robot motion:
it only shells out to already-validated command lines (`ros2 run
gen3-lite-ros2 joint_playback ...`, `ros2 launch kinova_gen3_lite_rl_bridge
shared_autonomy.launch.py mode:=live ...`, `ros2 bag record ...`,
`pw-record ...`) and does NOT start kortex_bringup itself — hardware
bringup stays a separate manual step, matching every launch file in this
package.

Usage:
    ros2 run kinova_gen3_lite_rl_bridge study_control_panel
    ros2 run kinova_gen3_lite_rl_bridge study_control_panel --http-port 8080
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
import signal
import subprocess
import sys
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from kinova_gen3_lite_rl_bridge.study_session_validation import (
    is_valid_env_id,
    is_valid_user_id,
    normalize_user_id,
)
import tornado.ioloop
import tornado.web

DEFAULT_HTTP_PORT = 8080
DEFAULT_RL_PROJECT_DIR = '/home/mavis/ros2_ws/src/peer_investigation_real_robot'
DEFAULT_SESSIONS_DIR = os.path.expanduser('~/kinova_study_sessions')
DEFAULT_AUDIO_DEVICE = 'default'

# /joint_states + /joy + the actual blended commands sent — enough to
# fully reconstruct a session for analysis. No camera topic: unlike
# investigating_peer_real_robot/session_gui.py's study (on `main`), this
# project has no camera set up.
BAG_TOPICS = ['/joint_states', '/joy', '/joint_trajectory_controller/joint_trajectory']

STATUS_IDLE = 'idle'
STATUS_RUNNING = 'running'
STATUS_DONE = 'done'
STATUS_FAILED = 'failed'


@dataclass
class PanelConfig:
    home_pose_path: str
    sessions_dir: str
    audio_device: str = DEFAULT_AUDIO_DEVICE


class PanelState:
    """Server-side process/session state — the actual source of truth."""

    def __init__(self):
        self.home_proc: Optional[subprocess.Popen] = None
        self.study_proc: Optional[subprocess.Popen] = None
        self.bag_proc: Optional[subprocess.Popen] = None
        self.audio_proc: Optional[subprocess.Popen] = None
        self.has_homed = False
        self.last_home_result: Optional[str] = None
        self.last_study_result: Optional[str] = None
        self.audio_start_error: Optional[str] = None

    def audio_recording(self) -> bool:
        return self.audio_proc is not None and self.audio_proc.poll() is None

    def homing_status(self) -> str:
        if self.home_proc is not None:
            returncode = self.home_proc.poll()
            if returncode is None:
                return STATUS_RUNNING
            self.last_home_result = STATUS_DONE if returncode == 0 else STATUS_FAILED
            if returncode == 0:
                self.has_homed = True
            self.home_proc = None
        return self.last_home_result or STATUS_IDLE

    def study_status(self) -> str:
        if self.study_proc is not None:
            returncode = self.study_proc.poll()
            if returncode is None:
                return STATUS_RUNNING
            self.last_study_result = STATUS_DONE if returncode == 0 else STATUS_FAILED
            self.study_proc = None
        return self.last_study_result or STATUS_IDLE


def _session_paths(sessions_dir: str, user_id: str, timestamp: str):
    """
    Return (session_dir, bag_path, audio_path) for one session.

    One session directory holds a `rosbag/` subdirectory (the bag's own
    `-o` target — `ros2 bag record` creates that path itself, so it must
    not already exist) and `audio.wav` as siblings — matches
    investigating_peer_real_robot/session_gui.py's own layout.
    """
    session_dir = os.path.join(sessions_dir, f'{user_id}_{timestamp}')
    bag_path = os.path.join(session_dir, 'rosbag')
    audio_path = os.path.join(session_dir, 'audio.wav')
    return session_dir, bag_path, audio_path


def _parse_pipewire_sources(pw_dump_json: str) -> list:
    """
    Parse `pw-dump` JSON into [{id, label}, ...] for the device picker.

    `id` is a PipeWire node.name (stable across reboots — tied to the
    hardware, unlike node id numbers) for direct use as `pw-record
    --target`. We use pw-record/pw-dump rather than raw ALSA device
    strings (arecord -D hw:.../plughw:...) because on a PipeWire-managed
    system PipeWire holds the hardware nodes itself — a direct ALSA
    open of anything but its own virtual "default" device fails with
    "Device or resource busy" (confirmed empirically: even
    `sysdefault:CARD=...` collided). pw-record talks to PipeWire's own
    graph instead of grabbing hardware directly, so it doesn't hit that.
    'default' (meaning: omit --target, let PipeWire auto-pick) is always
    first/present even if nothing was parsed, so the field is never empty.
    """
    devices = [{'id': 'default', 'label': 'default'}]
    try:
        objects = json.loads(pw_dump_json)
    except ValueError:
        return devices
    for obj in objects:
        props = (obj.get('info') or {}).get('props') or {}
        if props.get('media.class') != 'Audio/Source':
            continue
        node_name = props.get('node.name')
        if not node_name:
            continue
        label = props.get('node.description') or props.get('node.nick') or node_name
        devices.append({'id': node_name, 'label': label})
    return devices


def _list_audio_devices() -> list:
    try:
        result = subprocess.run(
            ['pw-dump'], capture_output=True, text=True, timeout=5, check=False)
    except OSError:
        return [{'id': 'default', 'label': 'default'}]
    return _parse_pipewire_sources(result.stdout)


def _stop_process(proc: Optional[subprocess.Popen]) -> None:
    """
    Stop proc gracefully, escalating if it doesn't exit in time.

    SIGINT first (lets `ros2 bag record` finalize its file, and gives
    `ros2 launch`'s children a chance to shut down cleanly), then
    terminate/kill as a fallback — mirrors session_gui.py's stop sequence.
    """
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class StatusHandler(tornado.web.RequestHandler):
    def initialize(self, state: PanelState, config: PanelConfig):
        self._state = state
        self._config = config

    def get(self):
        self.write({
            'homing': self._state.homing_status(),
            'study': self._state.study_status(),
            'has_homed': self._state.has_homed,
            'audio_recording': self._state.audio_recording(),
            'audio_start_error': self._state.audio_start_error,
            'default_audio_device': self._config.audio_device,
        })


class AudioDevicesHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({'devices': _list_audio_devices()})


class HomeHandler(tornado.web.RequestHandler):
    def initialize(self, state: PanelState, config: PanelConfig):
        self._state = state
        self._config = config

    def post(self):
        if self._state.study_status() == STATUS_RUNNING:
            self.set_status(409)
            self.write({'error': 'a study is currently running'})
            return
        if self._state.homing_status() == STATUS_RUNNING:
            self.set_status(409)
            self.write({'error': 'homing is already in progress'})
            return

        self._state.last_home_result = None
        self._state.home_proc = subprocess.Popen(
            ['ros2', 'run', 'gen3-lite-ros2', 'joint_playback', self._config.home_pose_path],
        )
        self.write({'status': 'started'})


class StartStudyHandler(tornado.web.RequestHandler):
    def initialize(self, state: PanelState, config: PanelConfig):
        self._state = state
        self._config = config

    def post(self):
        try:
            body = json.loads(self.request.body or b'{}')
        except ValueError:
            self.set_status(400)
            self.write({'error': 'invalid JSON body'})
            return

        user_id = body.get('user_id', '')
        env_id = body.get('env_id', '')
        audio_device = body.get('audio_device') or self._config.audio_device

        if self._state.homing_status() == STATUS_RUNNING:
            self.set_status(409)
            self.write({'error': 'homing is in progress'})
            return
        if self._state.study_status() == STATUS_RUNNING:
            self.set_status(409)
            self.write({'error': 'a study is already running'})
            return
        if not self._state.has_homed:
            self.set_status(400)
            self.write({'error': 'home the robot before starting a study'})
            return
        if not is_valid_user_id(user_id):
            self.set_status(400)
            self.write({'error': 'participant ID must be numeric and non-empty'})
            return
        if not is_valid_env_id(env_id):
            self.set_status(400)
            self.write({'error': f'env_id must be one of {("env_a", "env_b")}'})
            return

        user_id = normalize_user_id(user_id)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir, bag_path, audio_path = _session_paths(
            self._config.sessions_dir, user_id, timestamp)
        os.makedirs(session_dir, exist_ok=True)
        self._append_session_log(timestamp, user_id, env_id, session_dir)

        self._state.last_study_result = None
        self._state.bag_proc = subprocess.Popen(
            ['ros2', 'bag', 'record', '-o', bag_path] + BAG_TOPICS,
        )
        self._state.study_proc = subprocess.Popen(
            [
                'ros2', 'launch', 'kinova_gen3_lite_rl_bridge', 'shared_autonomy.launch.py',
                'mode:=live', f'env_id:={env_id}',
            ],
        )

        # Best-effort: a missing mic/pw-record shouldn't block the study
        # itself, since the bag + arm control matter more than audio.
        # pw-record's own stderr is kept (not DEVNULL'd) in a sibling log
        # file -- a device that fails to link still exits 0 with a silent
        # recording, so this is the only place a failure would show up.
        audio_cmd = ['pw-record']
        if audio_device and audio_device != 'default':
            audio_cmd += ['--target', audio_device]
        audio_cmd.append(audio_path)

        self._state.audio_start_error = None
        try:
            with open(os.path.join(session_dir, 'audio_stderr.log'), 'wb') as audio_stderr:
                self._state.audio_proc = subprocess.Popen(
                    audio_cmd, stdout=subprocess.DEVNULL, stderr=audio_stderr,
                )
        except OSError as exc:
            self._state.audio_proc = None
            self._state.audio_start_error = str(exc)

        response = {'status': 'started', 'session_dir': session_dir, 'audio_device': audio_device}
        if self._state.audio_start_error is not None:
            response['audio_error'] = self._state.audio_start_error
        self.write(response)

    def _append_session_log(self, timestamp, user_id, env_id, session_dir):
        log_path = os.path.join(self._config.sessions_dir, 'sessions.csv')
        is_new = not os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(['timestamp', 'user_id', 'env_id', 'session_dir'])
            writer.writerow([timestamp, user_id, env_id, session_dir])


class StopStudyHandler(tornado.web.RequestHandler):
    def initialize(self, state: PanelState):
        self._state = state

    def post(self):
        _stop_process(self._state.study_proc)
        _stop_process(self._state.bag_proc)
        _stop_process(self._state.audio_proc)
        self._state.study_proc = None
        self._state.bag_proc = None
        self._state.audio_proc = None
        self.write({'status': 'stopped'})


def _default_web_path():
    # Falls back to the installed copy under this package's share
    # directory (matches joy_web_bridge.py's _default_web_path()).
    return get_package_share_directory('kinova_gen3_lite_rl_bridge') + '/web/study_control_panel'


def make_app(state: PanelState, config: PanelConfig, web_path: str):
    return tornado.web.Application([
        (r'/status', StatusHandler, {'state': state, 'config': config}),
        (r'/audio_devices', AudioDevicesHandler),
        (r'/home', HomeHandler, {'state': state, 'config': config}),
        (r'/start_study', StartStudyHandler, {'state': state, 'config': config}),
        (r'/stop_study', StopStudyHandler, {'state': state}),
        (r'/(.*)', tornado.web.StaticFileHandler,
         {'path': web_path, 'default_filename': 'index.html'}),
    ])


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--web-path', default=None)
    parser.add_argument('--http-port', type=int, default=DEFAULT_HTTP_PORT)
    parser.add_argument('--rl-project-dir', default=DEFAULT_RL_PROJECT_DIR)
    parser.add_argument('--sessions-dir', default=DEFAULT_SESSIONS_DIR)
    parser.add_argument('--audio-device', default=DEFAULT_AUDIO_DEVICE)
    parsed = parser.parse_args(args=args)

    web_path = parsed.web_path or _default_web_path()
    home_pose_path = os.path.join(parsed.rl_project_dir, 'config', 'home_pose.txt')
    config = PanelConfig(
        home_pose_path=home_pose_path,
        sessions_dir=parsed.sessions_dir,
        audio_device=parsed.audio_device,
    )
    state = PanelState()

    app = make_app(state, config, web_path)
    app.listen(parsed.http_port)
    print(
        f'[study_control_panel] serving at http://0.0.0.0:{parsed.http_port}/',
        file=sys.stderr,
    )
    print(f'[study_control_panel] home pose file: {home_pose_path}', file=sys.stderr)
    print(f'[study_control_panel] sessions dir: {parsed.sessions_dir}', file=sys.stderr)
    print(f'[study_control_panel] audio device: {parsed.audio_device}', file=sys.stderr)

    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
