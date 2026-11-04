# Superseding instruction: BPTT only

2026-11-03 update: the old BPTT worker was also explicitly cancelled. The user
requested a fresh Go2-supermimic-aligned experiment. See
[alignment record](GO2_SUPERMIMIC_ALIGNMENT.md). The paragraphs below describe
the earlier November 2 decision, not permission to restart the old worker.

2026-11-02: User explicitly cancelled the queued fresh PPO control. The existing
PPO results already exist; first train only the new from-scratch BPTT policy.

- The currently running BPTT worker is left uninterrupted, without reinitializing
  or resuming it. Its source, optimization settings and budget are unchanged.
- The launcher's post-BPTT tail no longer starts PPO or a paired evaluation.
- A separate PPO entrypoint guard also checks `PPO_CANCELLED_BY_USER` in this run
  root, so stale launchers cannot accidentally start the cancelled job.
- The original container/run directory name still says `pair`; this is just the
  launch-time name, not evidence that a PPO control is running or queued.
- Earlier paired-experiment plans/status are superseded by this instruction.
- No completed PPO training or result is deleted.
