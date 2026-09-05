"""
Drive scripts/policy_server.py as a subprocess and exercise the IPC layer.

Runs the real policy_server.py (via `uv run`) and exercises the
ipc_protocol wire format directly, with no ROS involved — validates the
IPC layer standalone before ever wiring it into policy_bridge_node.

Skipped when `uv` isn't on PATH or the reference checkpoint doesn't exist
(e.g. a fresh checkout without the uv environment synced, or CI without
the RL project's heavy dependencies installed) — this test intentionally
exercises the real uv/torch/mjlab stack, not a mock, so it can't run
everywhere `colcon test` runs.
"""

from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / 'checkpoints' / '2026-08-27_12-52-48' / 'model_9999.pt'
STARTUP_TIMEOUT_S = 90.0

sys.path.insert(0, str(REPO_ROOT / 'kinova_gen3_lite_rl_bridge'))
import ipc_protocol as proto  # noqa: E402,I100

pytestmark = pytest.mark.skipif(
    shutil.which('uv') is None or not CHECKPOINT.exists(),
    reason='requires `uv` on PATH and the reference checkpoint to be present',
)


@pytest.fixture
def policy_server(tmp_path):
    # A path unique per test, not a shared fixed path: policy_server.py's
    # cleanup only runs on a graceful exit, not on the SIGTERM this
    # fixture's teardown sends, so a fixed path could leave a stale socket
    # *file* behind for the next test to mistake for a ready server.
    socket_path = str(tmp_path / 'policy.sock')
    proc = subprocess.Popen(
        [
            'uv',
            'run',
            'python',
            'scripts/policy_server.py',
            '--checkpoint',
            str(CHECKPOINT),
            '--socket-path',
            socket_path,
            '--device',
            'cpu',
        ],
        cwd=REPO_ROOT,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.connect(socket_path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            assert proc.poll() is None, 'policy_server.py exited before accepting connections'
            time.sleep(0.5)
    else:
        proc.terminate()
        pytest.fail(f'policy_server.py never accepted a connection within {STARTUP_TIMEOUT_S}s')

    try:
        yield socket_path
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_ping_pong(policy_server):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(policy_server)
        conn.sendall(proto.pack_header(proto.MSG_PING, seq=1))
        header = proto.recv_exact(conn, proto.HEADER_SIZE)
        msg_type, seq = proto.unpack_header(header)
        assert msg_type == proto.MSG_PONG
        assert seq == 1


def test_infer_returns_correctly_shaped_action(policy_server):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(policy_server)
        obs = [0.0] * proto.OBS_DIM
        conn.sendall(proto.pack_header(proto.MSG_INFER_REQ, seq=1) + proto.pack_obs(obs))
        header = proto.recv_exact(conn, proto.HEADER_SIZE)
        msg_type, seq = proto.unpack_header(header)
        assert msg_type == proto.MSG_INFER_RESP
        assert seq == 1
        payload = proto.recv_exact(conn, proto.ACTION_SIZE)
        action = proto.unpack_action(payload)
        assert len(action) == proto.ACTION_DIM
        assert all(v == v for v in action)  # no NaNs
