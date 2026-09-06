r"""Analytic potential-field inference server for real-hardware deployment.

Sibling to policy_server.py, not a replacement — that server and the RL
policy path it serves stay fully intact and independently launchable
(see shared_autonomy_node's `action_source` parameter). Runs under this
project's uv environment (kinpy available) as a plain socket server with
zero ROS imports and, unlike policy_server.py, zero torch/mjlab/rsl_rl —
just kinpy FK/Jacobian math, so there's no checkpoint-loading wait before
it's ready.

Usage:
    uv run python scripts/potential_field_server.py \\
        --socket-path /tmp/kinova_potential_field.sock
"""

import argparse
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kinova_gen3_lite_rl_bridge'))
import ipc_protocol as proto  # noqa: E402,I100 (path insert must precede this)

from kinova_mjlab_reaching.deploy.potential_field import (  # noqa: E402
    cartesian_delta_to_joint_delta,
    compute_field_delta,
    load_chain,
)


def _serve_connection(conn: socket.socket, chain, k_p: float, damping_lambda: float) -> None:
    while True:
        try:
            header = proto.recv_exact(conn, proto.HEADER_SIZE)
            msg_type, seq = proto.unpack_header(header)

            if msg_type == proto.MSG_PING:
                conn.sendall(proto.pack_header(proto.MSG_PONG, seq))
                continue

            if msg_type == proto.MSG_FIELD_REQ:
                payload = proto.recv_exact(conn, proto.FIELD_REQ_SIZE)
                joint_pos, target_xyz = proto.unpack_field_req(payload)
                delta = compute_field_delta(chain, joint_pos, target_xyz, k_p, damping_lambda)
                conn.sendall(
                    proto.pack_header(proto.MSG_FIELD_RESP, seq) + proto.pack_action(delta.tolist())
                )
                continue

            if msg_type == proto.MSG_JOG_REQ:
                payload = proto.recv_exact(conn, proto.FIELD_REQ_SIZE)
                joint_pos, cartesian_delta = proto.unpack_field_req(payload)
                delta = cartesian_delta_to_joint_delta(
                    chain, joint_pos, cartesian_delta, damping_lambda
                )
                conn.sendall(
                    proto.pack_header(proto.MSG_JOG_RESP, seq) + proto.pack_action(delta.tolist())
                )
                continue

            raise proto.ProtocolError(f'unexpected msg_type {msg_type:#x}')
        except proto.ProtocolError as exc:
            print(f'[potential_field_server] connection error: {exc}', file=sys.stderr)
            return
        except OSError:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--socket-path', type=Path, default=Path('/tmp/kinova_potential_field.sock')
    )
    parser.add_argument('--k-p', type=float, default=2.0)
    parser.add_argument('--damping-lambda', type=float, default=0.05)
    args = parser.parse_args()

    print('[potential_field_server] loading kinematic chain...', file=sys.stderr)
    chain = load_chain()
    print('[potential_field_server] chain loaded.', file=sys.stderr)

    socket_path = str(args.socket_path)
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(socket_path)
        sock.listen(1)
        print(f'[potential_field_server] listening on {socket_path}', file=sys.stderr)
        try:
            while True:
                conn, _ = sock.accept()
                with conn:
                    _serve_connection(conn, chain, args.k_p, args.damping_lambda)
        finally:
            os.unlink(socket_path)


if __name__ == '__main__':
    main()
