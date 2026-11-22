"""CPU actor replay benchmark; never imports the hardware transport."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .policy import Actor


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--output', type=Path, default=Path('../logs/actor_benchmark.json'))
    args = parser.parse_args()
    model = root/'models/ppo10k_actor.jit'
    actor = Actor(model)
    samples = np.load(root/'verification/actor_parity.npz')
    observations = samples['observations']
    outputs = np.array([actor(obs) for obs in observations])
    error = np.abs(outputs-samples['actions'])
    np.testing.assert_allclose(outputs, samples['actions'], atol=1e-4, rtol=1e-5)
    for i in range(50):
        actor(observations[i % len(observations)])
    times = []
    for i in range(args.iterations):
        start = time.perf_counter()
        actor(observations[i % len(observations)])
        times.append((time.perf_counter()-start)*1000)
    report = dict(architecture=platform.machine(), python=platform.python_version(),
                  torch=actor.torch.__version__, numpy=np.__version__,
                  actor_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  input='synthetic observations, local CPU reference; no live robot states',
                  reference_frames=len(observations), max_abs_action_error=float(error.max()),
                  iterations=args.iterations, inference_ms=dict(mean=float(np.mean(times)),
                  p95=float(np.percentile(times,95)), p99=float(np.percentile(times,99)),
                  maximum=float(np.max(times))), over_20ms=int(np.sum(np.array(times)>20)),
                  motor_transport_created=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
