"""Independent trace validation and explicitly conditional steady-state metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from a2w_precision_metrics import stats

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
a=p.parse_args()
out={}
hashes=set()
for mode in ['point','trajectory']:
    arrays={}
    checks=[]
    for seed in [20260916,20260917,20260918]:
        root=a.root/mode/f'seed{seed}'
        assert json.loads((root/'status.json').read_text())['stage']=='COMPLETE'
        manifest=json.loads((root/'manifest.json').read_text())
        hashes.add(manifest['checkpoint_sha256'])
        with np.load(root/'trace.npz') as data:
            assert data['error_m'].shape==(10,1000,64)
            assert all(np.isfinite(data[k]).all() for k in data.files)
            e=np.linalg.norm(data['ee_position_w']-data['reference_w'],axis=-1)
            residual=float(np.max(np.abs(e-data['error_m'])))
            assert residual<1e-5
            assert data['mapping_residual_m'].max()<1e-5
            if mode=='point':
                assert np.max(np.abs(data['reference_w']-data['reference_w'][:,:1]))<1e-5
                nominal=np.asarray(json.loads((root/'targets.json').read_text())['points'])
                nominal[...,0]-=.174
                assert np.max(np.abs(nominal[...,:2]-data['command'][:,0,:,:2]))<1e-4
            checks.append(dict(seed=seed,error_recomputation_max_residual_m=residual))
            for k in ['error_m','terminated','reference_w','root_position_w','initial_root_w']:
                arrays.setdefault(k,[]).append(data[k])
    data={k:np.concatenate(v) for k,v in arrays.items()}
    e=data['error_m'].astype('float64')
    failed=data['terminated'].any(axis=1)
    report=dict(validation=checks,failed_trials=int(failed.sum()))
    if mode=='point':
        result=json.loads((a.root/mode/'results.json').read_text())
        steady=e[:,-250:].mean(axis=1)
        report['thresholds']={}
        for threshold in [.02,.05]:
            count=np.zeros(failed.shape,int)
            first=np.full(failed.shape,np.nan)
            for i in range(1000):
                count=np.where(e[:,i]<=threshold,count+1,0)
                first[(count>=25)&np.isnan(first)]=(i-23)*.02
            arrived=np.isfinite(first)&~failed
            before_window=arrived&(first+.5<=15.)
            assert int(arrived.sum())==result['arrival'][f'{threshold*100:.0f}cm']['successes']
            report['thresholds'][f'{threshold*100:.0f}cm']=dict(
                arrived=int(arrived.sum()),total=int(failed.size),
                final5s_mean_within_threshold_and_no_failure=int(((steady<=threshold)&~failed).sum()),
                final5s_all_frames_within_threshold_and_no_failure=int(((e[:,-250:].max(axis=1)<=threshold)&~failed).sum()),
                arrived_before_final5s_window=int(before_window.sum()),
                conditional_final5s_mean_cm=stats(steady[before_window],100))
        z=data['reference_w'][:,0,:,2]
        report['height_bins']=[]
        for lo,hi in [(0,.3),(.3,.5),(.5,.7),(.7,1.1)]:
            mask=(z>=lo)&(z<hi)
            report['height_bins'].append(dict(low=lo,high=hi,trials=int(mask.sum()),steady_mean_cm=stats(steady[mask],100)))
    out[mode]=report
assert len(hashes)==1
out['checkpoint_sha256']=next(iter(hashes))
(a.root/'validation_and_steady_analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,indent=2,allow_nan=False))
