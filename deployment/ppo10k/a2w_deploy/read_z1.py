"""Receive-only reader of the classic z1_ctrl SDK RecvState (packed 208 bytes).

Uses the SDK feedback port exclusively for this short read. Sends no UDP
packets, state changes, joint targets, or gripper commands.
"""
import argparse
import json
import math
from pathlib import Path
import socket
import struct
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--port', type=int, default=8072)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    packets, lengths, last, peer = 0, {}, None, None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('127.0.0.1', args.port))
        sock.settimeout(.2)
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            try:
                data, peer = sock.recvfrom(4096)
            except socket.timeout:
                continue
            lengths[str(len(data))] = lengths.get(str(len(data)), 0) + 1
            if len(data) != 208:
                continue
            packets += 1
            joints = []
            for i in range(7):
                offset = 6 + i * 22
                tau, dq, ddq, q = struct.unpack_from('<4f', data, offset)
                motors = []
                for j in range(2):
                    temp, error, connected = struct.unpack_from('<bBB', data, offset+16+3*j)
                    motors.append(dict(temperature=temp, error=error, connection=connected >> 6))
                joints.append(dict(q_rad=q, q_deg=math.degrees(q), dq=dq, ddq=ddq,
                                   tau=tau, motors=motors))
            last = dict(header_hex=data[:2].hex(), fsm_state=struct.unpack_from('<i', data, 2)[0],
                        joints=joints, sdk_cartesian_rpy_xyz=struct.unpack_from('<6d', data, 160))
        report = dict(receive_only=True, udp_packets_sent=0, seconds=time.monotonic()-start,
                      packets=packets, packet_lengths=lengths, peer=peer, last_state=last)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
