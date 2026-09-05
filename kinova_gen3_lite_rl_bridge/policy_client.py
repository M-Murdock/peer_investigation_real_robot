"""
Thin IPC client for the policy_server.py process (see ipc_protocol.py).

Keeps one persistent Unix-domain-socket connection and reconnects with
backoff on failure. Every call is fail-safe by design: on any error this
returns None rather than raising, so a caller's control loop can simply
skip a cycle instead of crashing or publishing something unvalidated.
"""

import logging
import socket
import time

from kinova_gen3_lite_rl_bridge import ipc_protocol as proto
import numpy as np

_RECONNECT_BACKOFF_S = 1.0


class PolicyClient:
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
            self._logger.info(f'connected to policy server at {self._socket_path}')
            return True
        except OSError as exc:
            self._logger.warning(f'failed to connect to policy server: {exc}')
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

    def infer(self, obs: np.ndarray) -> np.ndarray | None:
        """Send one observation, return the policy's action, or None on failure."""
        if not self._ensure_connected():
            return None
        seq = self._next_seq()
        try:
            self._conn.sendall(proto.pack_header(proto.MSG_INFER_REQ, seq) + proto.pack_obs(obs))
            header = proto.recv_exact(self._conn, proto.HEADER_SIZE)
            msg_type, resp_seq = proto.unpack_header(header)
            if msg_type != proto.MSG_INFER_RESP or resp_seq != seq:
                raise proto.ProtocolError(f'unexpected response {msg_type:#x}/{resp_seq}')
            payload = proto.recv_exact(self._conn, proto.ACTION_SIZE)
            return np.array(proto.unpack_action(payload), dtype=np.float32)
        except (OSError, proto.ProtocolError) as exc:
            self._logger.warning(f'infer failed, dropping connection: {exc}')
            self._drop_connection()
            return None

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return self._seq

    def close(self) -> None:
        self._drop_connection()
