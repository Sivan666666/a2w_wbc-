"""Verify trusted project-owned25K files on CPU; no optimizer/simulator/GPU use."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('supermimic_25k_baseline.json').read_text())
    for kind in ('checkpoint', 'actor'):
        path = args.model_dir / manifest[kind]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest[kind + '_sha256']:
            raise ValueError(f'{kind} SHA256 mismatch: {path}')
    import torch
    torch.set_num_threads(2)
    # Only deserialize after exact known-good SHA256 verification.
    ck = torch.load(args.model_dir / manifest['checkpoint'], map_location='cpu', weights_only=False)
    expected = {'iteration': manifest['iteration'], 'env_transitions': manifest['physical_transitions'],
                'imagined_transitions': manifest['imagined_transitions'],
                'physical_steps_per_env': manifest['physical_steps_per_env']}
    for key, value in expected.items():
        if ck[key] != value:
            raise ValueError(f'checkpoint metadata mismatch: {key}')
    if abs(ck['wall_seconds'] - manifest['training_wall_seconds']) > 1e-6:
        raise ValueError('checkpoint elapsed time mismatch')
    actor = torch.jit.load(str(args.model_dir / manifest['actor']), map_location='cpu').eval()
    if 'norm.mean' not in actor.state_dict():
        raise ValueError('RunningNorm missing')
    with torch.inference_mode():
        result = actor(torch.zeros(2, 234))
    if result.shape != (2, 22) or not torch.isfinite(result).all():
        raise ValueError('Actor interface failure')
    print(json.dumps({'status': 'PASS_CPU_ONLY', 'iteration': ck['iteration'],
                      'actor_shape': list(result.shape), 'training_started': False,
                      'physical_simulation_tested': False}, indent=2))


if __name__ == '__main__':
    main()
