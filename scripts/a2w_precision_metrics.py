"""Offline metrics for world-fixed targets and prescribed trajectories."""
import numpy as np


def stats(x, scale=1):
    x = np.asarray(x, dtype=np.float64).ravel() * scale
    if not len(x):
        return None
    return dict(mean=float(x.mean()), rmse=float(np.sqrt(np.mean(x*x))),
                p95=float(np.quantile(x, .95)), max=float(x.max()))


def summarize(data, mode, dt=.02):
    # Shape trial, step, environment. Each trial is independently initialized.
    e = data['error_m']
    failed = data['terminated'].any(axis=1)
    T, S, N = e.shape
    steady_steps = round(5/dt)
    steady = e[:, -steady_steps:].mean(axis=1)
    drift = np.linalg.norm(data['root_position_w'][..., :2]-data['initial_root_w'][:, None, :, :2], axis=-1)
    result = dict(mode=mode, trials=T*N, frames=e.size, dt=dt,
                  failed_trials=int(failed.sum()), termination_events=int(data['terminated'].sum()),
                  all_time_position_cm=stats(e, 100),
                  orientation_rad=stats(data['orientation_rad']),
                  base_max_displacement_per_trial_cm=stats(drift.max(axis=1), 100))
    if mode == 'point':
        result['steady_last5s_position_cm'] = stats(e[:, -steady_steps:], 100)
        result['per_trial_steady_mean_cm'] = stats(steady, 100)
        result['arrival'] = {}
        for threshold in (.02, .05):
            hold = round(.5/dt)
            counts = np.zeros((T, N), dtype=int)
            first = np.full((T, N), np.nan)
            for step in range(S):
                counts = np.where(e[:, step] <= threshold, counts+1, 0)
                new = (counts >= hold) & np.isnan(first)
                # Report start of the first qualifying 0.5 s interval.
                first[new] = (step-hold+2)*dt
            success = np.isfinite(first) & ~failed
            result['arrival'][f'{threshold*100:.0f}cm'] = dict(
                successes=int(success.sum()), total=T*N,
                fraction=float(success.mean()), hold_seconds=.5,
                time_definition='Start of first >=0.5s consecutive interval within threshold; only reset-free trials qualify.',
                arrival_time_s=stats(first[success]),
                censored_or_failed=int((~success).sum()))
    else:
        moving = data['phase_time_s'] > 5
        result['tracking_position_cm'] = stats(e[moving], 100)
        result['issued_target_tracking_cm'] = stats(data['issued_error_m'][moving], 100)
        result['tracking_orientation_rad'] = stats(data['orientation_rad'][moving])
        result['trajectories'] = {}
        for k, name in enumerate(('circle', 'figure8', 'reach_retract')):
            selected = (data['kind'] == k)[:, None, :] & moving
            trials = data['kind'] == k
            result['trajectories'][name] = dict(
                trials=int(trials.sum()), failed_trials=int((trials & failed).sum()),
                position_cm=stats(e[selected], 100),
                orientation_rad=stats(data['orientation_rad'][selected]))
    return result
