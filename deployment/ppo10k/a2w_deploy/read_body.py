"""Read A2W feedback through PC2; does not construct Hardware or use Z1."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ._hardware import BodyReader


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=root/'deploy.nx.json')
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--output',type=Path,default=Path('../logs/body_readonly.json'))
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    reader=BodyReader(cfg['body_host'],cfg['body_peer_port'],cfg['body_local_port'])
    reader.wait_state()
    before=reader.status()
    start=time.monotonic()
    first=reader.state()
    ticks=set()
    while time.monotonic()-start<args.seconds:
        s=reader.state(); ticks.add(s['tick'])
        time.sleep(.002)
    elapsed=time.monotonic()-start
    last=reader.state()
    after=reader.status()
    report=dict(seconds=elapsed,received=last['messages']-first['messages'],
                receiver_hz=(last['messages']-first['messages'])/elapsed,
                observed_tick_values=len(ticks),first_tick=first['tick'],last_tick=last['tick'],
                bridge_before=before,bridge_after=after,last_state=last,
                all_finite=bool(np.isfinite(last['q']+last['dq']+last['quaternion']+last['gyro']).all()),
                z1_transport_created=False,hardware_controller_created=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
