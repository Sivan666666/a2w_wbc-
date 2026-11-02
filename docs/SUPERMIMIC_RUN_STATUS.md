# Initial launch record — 2026-11-02

**Superseded:** the user subsequently cancelled the queued PPO control. Only
the current BPTT training continues; no automatic paired benchmark remains.
See [cancellation record](SUPERMIMIC_PPO_CANCELLED.md). The text below is historical.

This is a launch-time snapshot, not a live dashboard or a completed benchmark.

- Source branch: `codex/supermimic-a2w`.
- Training implementation commit: `e393f8a`.
- Local worktree: `/home/sivan/whole_body/wbc/a2w_wbc_supermimic_20261102`.
- Host: PS1 (`ssh ps@192.168.1.106`), physical GPU 3.
- Remote source: `/mnt/sda/ps/leggedmanip-lab/supermimic_20261102`.
- Container: `a2w-bptt-ppo-pair-seed42-20261102`.
- Run root: `/mnt/sda/ps/leggedmanip-lab/supermimic_20261102/runs/pair_seed42`.
- A100 was not used: changed SSH host identity has not been independently verified.

## Observed launch state

Fresh BPTT training passed the model warmup and was actually updating Actor.
At the checked snapshot: 1,024 physical control steps/env, 4,227,072 total physical
transitions including startup validation, 129 Actor updates, finite recorded
losses/gradients, full checkpoint saved at `bptt/latest.pt`. This is early training;
no convergence or PPO-speedup claim is warranted. Initial policies can fall/reset.

Initial random Actor SHA256:
`805f1ab7d50089dac64dc6bd2a7fc51b39429111b61c6295631e574efcea9f10`.
The original Precision 50K checkpoint is not loaded. The PPO control is queued,
not currently running, and will load only the new step-zero random weights.

## Queue / budgets

1. BPTT: 4096 envs, seed 42, 120000 physical control steps/env including 512 warmup.
2. PPO: same env/task/seed/initial mean-network weights, 5000 x 24 steps/env.
3. Final physical evaluation: each method, seeds 20261111/12/13, 64 envs x 200 s.
4. Six-checkpoint physical learning-curve diagnostic per method, 64 envs x 20 s.
5. Aggregate `comparison.json`, `comparison.md`, `learning_curves.json`.

The queue stops on a failed command; it does not silently fall back to PPO or load
an older pretrained policy. No background promise of automatic repair is implied.
GPU 0/1/2 jobs were left untouched.

## Inspect

```bash
ssh ps@192.168.1.106 \
  'tail -n 1 /mnt/sda/ps/leggedmanip-lab/supermimic_20261102/runs/pair_seed42/bptt/progress.jsonl'
ssh ps@192.168.1.106 \
  'docker inspect -f "{{.State.Status}} {{.State.ExitCode}}" a2w-bptt-ppo-pair-seed42-20261102'
```

TensorBoard: `bptt/tensorboard/` and the PPO run directory. Source hashes,
installed package versions and imported module paths are recorded separately in
each run's `source_manifest.json`. Both training axes (physical transitions and
wall time) are retained; imagined transitions do not count as physical samples.
