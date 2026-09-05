r"""RL policy inference server for real-hardware deployment.

Runs under this project's uv environment (Python 3.13, torch/mjlab/rsl_rl
available) as a plain socket server with zero ROS imports — rclpy is a
system Python 3.12 package and cannot share a process with this
environment (mjlab hard-requires python>=3.13). kinova_gen3_lite_rl_bridge
(the ROS side, Python 3.12) is the only client, talking over the
ipc_protocol wire format on a Unix domain socket.

Usage:
    uv run python scripts/policy_server.py \\
        --checkpoint checkpoints/2026-08-27_12-52-48/model_9999.pt
"""

import argparse
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kinova_gen3_lite_rl_bridge'))
import ipc_protocol as proto  # noqa: E402,I100 (path insert must precede this)

from kinova_mjlab_reaching.deploy.policy_loader import load_reaching_policy  # noqa: E402
import torch  # noqa: E402


def _serve_connection(conn: socket.socket, policy) -> None:
    while True:
        try:
            header = proto.recv_exact(conn, proto.HEADER_SIZE)
            msg_type, seq = proto.unpack_header(header)

            if msg_type == proto.MSG_PING:
                conn.sendall(proto.pack_header(proto.MSG_PONG, seq))
                continue

            if msg_type == proto.MSG_INFER_REQ:
                payload = proto.recv_exact(conn, proto.OBS_SIZE)
                obs = torch.tensor(
                    proto.unpack_obs(payload), dtype=torch.float32
                ).unsqueeze(0)
                # get_inference_policy()'s callable indexes its input by
                # observation-group name (rsl_rl's MLPModel.get_latent does
                # `obs[obs_group]`) rather than taking a raw tensor — 'actor'
                # is the group name reach_env_cfg.py's ObservationManager
                # registers the policy's own 21-dim obs under.
                with torch.inference_mode():
                    action = policy({'actor': obs}).squeeze(0).tolist()
                conn.sendall(
                    proto.pack_header(proto.MSG_INFER_RESP, seq)
                    + proto.pack_action(action)
                )
                continue

            raise proto.ProtocolError(f'unexpected msg_type {msg_type:#x}')
        except proto.ProtocolError as exc:
            print(f'[policy_server] connection error: {exc}', file=sys.stderr)
            return
        except OSError:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--socket-path', type=Path, default=Path('/tmp/kinova_rl_policy.sock'))
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print(f'[policy_server] loading checkpoint {args.checkpoint} ...', file=sys.stderr)
    policy = load_reaching_policy(args.checkpoint, device=args.device)
    print('[policy_server] policy loaded.', file=sys.stderr)

    socket_path = str(args.socket_path)
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(socket_path)
        sock.listen(1)
        print(f'[policy_server] listening on {socket_path}', file=sys.stderr)
        try:
            while True:
                conn, _ = sock.accept()
                with conn:
                    _serve_connection(conn, policy)
        finally:
            os.unlink(socket_path)


if __name__ == '__main__':
    main()
