"""
Thin IPC client for the potential_field_server.py process.

Sibling to policy_client.py, not a modification of it — same
connect/reconnect/fail-safe shape, over a separate socket (independent
lifecycle from the policy server, so shared_autonomy_node's
action_source can pick either backend). See ipc_protocol.py's
MSG_FIELD_REQ/MSG_FIELD_RESP for the wire format.
"""

import logging
import socket
import time

from kinova_gen3_lite_rl_bridge import ipc_protocol as proto
import numpy as np

_RECONNECT_BACKOFF_S = 1.0


class PotentialFieldClient:
    def __init__(self, socket_path: str, logger: logging.Logger | None = None):
        self._socket_path = socket_path
        self._logger = logger or logging.getLogger(__name__)
        self._conn: socket.socket | None = None
        self._seq = 0
        self._last_connect_attempt = 0.0

    def _ensure_connected(self) -> bool:
        if self._conn is not None:
            return True
        now = time.monotonic()
        if now - self._last_connect_attempt < _RECONNECT_BACKOFF_S:
            return False
        self._last_connect_attempt = now
        try:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.connect(self._socket_path)
            self._conn = conn
            self._logger.info(f'connected to potential field server at {self._socket_path}')
            return True
        except OSError as exc:
            self._logger.warning(f'failed to connect to potential field server: {exc}')
            self._conn = None
            return False

    def _drop_connection(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

    def ping(self) -> bool:
        """Send a PING and wait for the matching PONG. Used once at startup."""
        if not self._ensure_connected():
            return False
        seq = self._next_seq()
        try:
            self._conn.sendall(proto.pack_header(proto.MSG_PING, seq))
            header = proto.recv_exact(self._conn, proto.HEADER_SIZE)
            msg_type, resp_seq = proto.unpack_header(header)
            return msg_type == proto.MSG_PONG and resp_seq == seq
        except (OSError, proto.ProtocolError) as exc:
            self._logger.warning(f'ping failed: {exc}')
            self._drop_connection()
            return False

    def compute_delta(
        self, joint_pos: np.ndarray, target_xyz: np.ndarray
    ) -> np.ndarray | None:
        """Send (joint_pos, target_xyz), return the field's joint delta, or None on failure."""
        if not self._ensure_connected():
            return None
        seq = self._next_seq()
        try:
            self._conn.sendall(
                proto.pack_header(proto.MSG_FIELD_REQ, seq)
                + proto.pack_field_req(joint_pos, target_xyz)
            )
            header = proto.recv_exact(self._conn, proto.HEADER_SIZE)
            msg_type, resp_seq = proto.unpack_header(header)
            if msg_type != proto.MSG_FIELD_RESP or resp_seq != seq:
                raise proto.ProtocolError(f'unexpected response {msg_type:#x}/{resp_seq}')
            payload = proto.recv_exact(self._conn, proto.ACTION_SIZE)
            return np.array(proto.unpack_action(payload), dtype=np.float32)
        except (OSError, proto.ProtocolError) as exc:
            self._logger.warning(f'compute_delta failed, dropping connection: {exc}')
            self._drop_connection()
            return None

    def compute_jog_delta(
        self, joint_pos: np.ndarray, cartesian_delta: np.ndarray
    ) -> np.ndarray | None:
        """Send (joint_pos, cartesian_delta), return the joint delta, or None on failure."""
        if not self._ensure_connected():
            return None
        seq = self._next_seq()
        try:
            self._conn.sendall(
                proto.pack_header(proto.MSG_JOG_REQ, seq)
                + proto.pack_field_req(joint_pos, cartesian_delta)
            )
            header = proto.recv_exact(self._conn, proto.HEADER_SIZE)
            msg_type, resp_seq = proto.unpack_header(header)
            if msg_type != proto.MSG_JOG_RESP or resp_seq != seq:
                raise proto.ProtocolError(f'unexpected response {msg_type:#x}/{resp_seq}')
            payload = proto.recv_exact(self._conn, proto.ACTION_SIZE)
            return np.array(proto.unpack_action(payload), dtype=np.float32)
        except (OSError, proto.ProtocolError) as exc:
            self._logger.warning(f'compute_jog_delta failed, dropping connection: {exc}')
            self._drop_connection()
            return None

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return self._seq

    def close(self) -> None:
        self._drop_connection()
