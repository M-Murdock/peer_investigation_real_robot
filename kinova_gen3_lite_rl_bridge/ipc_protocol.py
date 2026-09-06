"""
Wire format between policy_bridge_node and scripts/policy_server.py.

Shared between policy_bridge_node (ROS, Python 3.12) and
scripts/policy_server.py (uv, Python 3.13) over a single persistent
Unix-domain-socket connection.

Deliberately stdlib-only (no rclpy, no torch) so this one file can be
imported unmodified by both processes despite their disjoint dependency
sets: as part of the kinova_gen3_lite_rl_bridge package on the ROS side,
and via a sys.path insert pointing at this package's directory on the uv
side (see scripts/policy_server.py) — there is exactly one copy of this
file. Fixed-size struct packing, not JSON: OBS_DIM/ACTION_DIM are fixed at
design time (they come from the trained checkpoint's observation/action
space), so variable-length framing buys nothing but risk.
"""

import socket
import struct

OBS_DIM = 21
ACTION_DIM = 6
JOINT_DIM = 6
TARGET_DIM = 3

MAGIC = 0xB0

MSG_PING = 0x02
MSG_PONG = 0x82
MSG_INFER_REQ = 0x01
MSG_INFER_RESP = 0x81
# Additive — for scripts/potential_field_server.py (Milestone 3). The
# existing PING/INFER messages above and everything that speaks them
# (policy_server.py, policy_client.py) are unmodified by this addition.
MSG_FIELD_REQ = 0x03
MSG_FIELD_RESP = 0x83
# Additive — for a direct Cartesian-delta-to-joint-delta conversion
# (Milestone 4), served by the same potential_field_server.py process.
# Same wire shape as FIELD_REQ/ACTION (reuses pack_field_req/unpack_field_req
# and pack_action/unpack_action — the second 3-float field just means
# "desired cartesian_delta" here instead of "target_xyz"), but a distinct
# msg_type so the protocol stays self-documenting.
MSG_JOG_REQ = 0x04
MSG_JOG_RESP = 0x84

HEADER_FMT = '!BBI'  # magic(1) | msg_type(1) | seq(uint32)
HEADER_SIZE = struct.calcsize(HEADER_FMT)

OBS_FMT = f'!{OBS_DIM}f'
OBS_SIZE = struct.calcsize(OBS_FMT)

ACTION_FMT = f'!{ACTION_DIM}f'
ACTION_SIZE = struct.calcsize(ACTION_FMT)

# FIELD_REQ payload: current joint_pos (6 floats) followed by target_xyz
# (3 floats). FIELD_RESP reuses ACTION_FMT/ACTION_SIZE — same shape as a
# joint delta.
FIELD_REQ_FMT = f'!{JOINT_DIM}f{TARGET_DIM}f'
FIELD_REQ_SIZE = struct.calcsize(FIELD_REQ_FMT)


class ProtocolError(Exception):
    """Malformed header or payload on the IPC socket."""


def pack_header(msg_type: int, seq: int) -> bytes:
    return struct.pack(HEADER_FMT, MAGIC, msg_type, seq)


def unpack_header(buf: bytes) -> tuple[int, int]:
    """Return (msg_type, seq). Raise ProtocolError if magic doesn't match."""
    magic, msg_type, seq = struct.unpack(HEADER_FMT, buf)
    if magic != MAGIC:
        raise ProtocolError(f'bad magic byte: {magic:#x}')
    return msg_type, seq


def pack_obs(obs) -> bytes:
    return struct.pack(OBS_FMT, *obs)


def unpack_obs(buf: bytes) -> tuple[float, ...]:
    return struct.unpack(OBS_FMT, buf)


def pack_action(action) -> bytes:
    return struct.pack(ACTION_FMT, *action)


def unpack_action(buf: bytes) -> tuple[float, ...]:
    return struct.unpack(ACTION_FMT, buf)


def pack_field_req(joint_pos, target_xyz) -> bytes:
    return struct.pack(FIELD_REQ_FMT, *joint_pos, *target_xyz)


def unpack_field_req(buf: bytes) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return (joint_pos, target_xyz)."""
    values = struct.unpack(FIELD_REQ_FMT, buf)
    return values[:JOINT_DIM], values[JOINT_DIM:]


def recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise ProtocolError on EOF."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ProtocolError(f'connection closed with {remaining} bytes still expected')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)
