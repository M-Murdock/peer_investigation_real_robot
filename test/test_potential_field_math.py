"""
Drive scripts/potential_field_server.py as a subprocess, over the IPC layer.

Sibling to test_ipc_roundtrip.py, same pattern (real uv subprocess, real
socket, no ROS) — validates the potential-field math end to end before
ever wiring it into shared_autonomy_node.

Skipped when `uv` isn't on PATH (this test intentionally exercises the
real uv/kinpy stack, not a mock, so it can't run everywhere `colcon test`
does — same reasoning as test_ipc_roundtrip.py, but no checkpoint is
needed here since this server has no trained weights to load).
"""

from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP_TIMEOUT_S = 30.0  # much faster than policy_server.py: no checkpoint to load

sys.path.insert(0, str(REPO_ROOT / 'kinova_gen3_lite_rl_bridge'))
import ipc_protocol as proto  # noqa: E402,I100

pytestmark = pytest.mark.skipif(
    shutil.which('uv') is None, reason='requires `uv` on PATH'
)

# Verified by hand (see the Milestone 3 planning session): the real
# end-effector position at the all-zero joint configuration for this
# robot's URDF, cross-checked against the same value observed from real
# hardware TF at the home pose.
EE_POS_AT_ZERO = (0.05700039, -0.01000051, 1.00324982)


def _start_server(tmp_path, k_p: float = 2.0):
    socket_path = str(tmp_path / 'field.sock')
    proc = subprocess.Popen(
        [
            'uv', 'run', 'python', 'scripts/potential_field_server.py',
            '--socket-path', socket_path, '--k-p', str(k_p),
        ],
        cwd=REPO_ROOT,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.connect(socket_path)
            return proc, socket_path
        except (FileNotFoundError, ConnectionRefusedError):
            assert proc.poll() is None, (
                'potential_field_server.py exited before accepting connections'
            )
            time.sleep(0.2)
    proc.terminate()
    pytest.fail(
        f'potential_field_server.py never accepted a connection within {STARTUP_TIMEOUT_S}s'
    )


@pytest.fixture
def field_server(tmp_path):
    proc, socket_path = _start_server(tmp_path)
    try:
        yield socket_path
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _field_request(socket_path, joint_pos, target_xyz, seq=1):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(socket_path)
        conn.sendall(
            proto.pack_header(proto.MSG_FIELD_REQ, seq)
            + proto.pack_field_req(joint_pos, target_xyz)
        )
        header = proto.recv_exact(conn, proto.HEADER_SIZE)
        msg_type, resp_seq = proto.unpack_header(header)
        assert msg_type == proto.MSG_FIELD_RESP
        assert resp_seq == seq
        payload = proto.recv_exact(conn, proto.ACTION_SIZE)
        return proto.unpack_action(payload)


def _jog_request(socket_path, joint_pos, cartesian_delta, seq=1):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(socket_path)
        conn.sendall(
            proto.pack_header(proto.MSG_JOG_REQ, seq)
            + proto.pack_field_req(joint_pos, cartesian_delta)
        )
        header = proto.recv_exact(conn, proto.HEADER_SIZE)
        msg_type, resp_seq = proto.unpack_header(header)
        assert msg_type == proto.MSG_JOG_RESP
        assert resp_seq == seq
        payload = proto.recv_exact(conn, proto.ACTION_SIZE)
        return proto.unpack_action(payload)


def test_ping_pong(field_server):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(field_server)
        conn.sendall(proto.pack_header(proto.MSG_PING, seq=1))
        header = proto.recv_exact(conn, proto.HEADER_SIZE)
        msg_type, seq = proto.unpack_header(header)
        assert msg_type == proto.MSG_PONG
        assert seq == 1


def test_zero_delta_when_target_is_current_ee_position(field_server):
    joint_pos = [0.0] * 6
    delta = _field_request(field_server, joint_pos, EE_POS_AT_ZERO)

    assert all(abs(v) < 1e-4 for v in delta), f'expected ~zero delta, got {delta}'


def test_delta_is_finite_for_an_offset_target(field_server):
    joint_pos = [0.0] * 6
    target = (EE_POS_AT_ZERO[0] + 0.05, EE_POS_AT_ZERO[1], EE_POS_AT_ZERO[2])
    delta = _field_request(field_server, joint_pos, target)

    assert len(delta) == proto.ACTION_DIM
    assert all(v == v and abs(v) < 1e6 for v in delta)  # no NaN/Inf, no blow-up
    assert any(abs(v) > 1e-6 for v in delta), 'expected a nonzero delta for an offset target'


def test_delta_magnitude_scales_with_k_p(tmp_path):
    joint_pos = [0.0] * 6
    target = (EE_POS_AT_ZERO[0] + 0.05, EE_POS_AT_ZERO[1], EE_POS_AT_ZERO[2])

    low_dir = tmp_path / 'low'
    high_dir = tmp_path / 'high'
    low_dir.mkdir()
    high_dir.mkdir()
    proc_low, sock_low = _start_server(low_dir, k_p=1.0)
    proc_high, sock_high = _start_server(high_dir, k_p=4.0)
    try:
        delta_low = _field_request(sock_low, joint_pos, target)
        delta_high = _field_request(sock_high, joint_pos, target)
    finally:
        proc_low.terminate()
        proc_high.terminate()
        proc_low.wait(timeout=10)
        proc_high.wait(timeout=10)

    norm_low = sum(v * v for v in delta_low) ** 0.5
    norm_high = sum(v * v for v in delta_high) ** 0.5
    assert norm_high > norm_low, (
        f'expected higher k_p to give a larger delta: {norm_low} vs {norm_high}'
    )


def test_jog_zero_cartesian_delta_gives_zero_joint_delta(field_server):
    joint_pos = [0.0] * 6
    delta = _jog_request(field_server, joint_pos, (0.0, 0.0, 0.0))

    assert all(abs(v) < 1e-9 for v in delta), f'expected exactly zero delta, got {delta}'


def test_jog_is_linear_in_the_cartesian_delta(field_server):
    # The Jacobian conversion is linear for a fixed joint_pos: negating or
    # scaling the input must negate/scale the output correspondingly. This
    # is a strong, FK-free regression check (this test file runs under
    # system Python, without kinpy, so it can't independently verify "this
    # moves toward +X" the way the uv-side planning session's ad hoc check
    # did) — but it precisely pins the linear-algebra behavior the real
    # Cartesian jog feature depends on.
    joint_pos = [0.1, -0.2, 0.3, -0.1, 0.2, 0.0]
    forward = _jog_request(field_server, joint_pos, (0.05, 0.0, 0.0))
    backward = _jog_request(field_server, joint_pos, (-0.05, 0.0, 0.0))
    doubled = _jog_request(field_server, joint_pos, (0.1, 0.0, 0.0))

    assert any(abs(v) > 1e-6 for v in forward), 'expected a nonzero delta'
    for f, b in zip(forward, backward):
        assert abs(f + b) < 1e-6, f'expected backward to negate forward: {forward} vs {backward}'
    for f, d in zip(forward, doubled):
        assert abs(2 * f - d) < 1e-6, (
            f'expected doubled input to double the delta: {forward} vs {doubled}'
        )
