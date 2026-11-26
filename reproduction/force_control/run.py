"""Opt-in force control. Defaults to a bounded smoke test, never a long training run."""
import argparse,sys,json,time,traceback,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import asdict
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['smoke','train','evaluate'],default='smoke');p.add_argument('--iterations',type=int);p.add_argument('--num_envs',type=int,default=16);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume');p.add_argument('--checkpoint');p.add_argument('--steps',type=int,default=300);AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if a.mode=='train' and (a.iterations is None or a.iterations<=0):p.error('train requires explicit total --iterations')
if a.mode=='evaluate' and not a.checkpoint:p.error('evaluate requires --checkpoint')
if a.mode!='train' and a.resume:p.error('--resume is for train only')
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=bool(a.resume))
def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
save('status.json',{'stage':'starting','mode':a.mode})
app=AppLauncher(a).app
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from force_control.core import ForceConfig,force_reward
from force_control.isaac import make_cfg,term
from force_control.learning import CSEPolicy,CSEPPO,PPOConfig
from force_control.inference import PolicyHistory, POLICY_TERMS


def main():
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    cfg,audit=make_cfg(a.device,a.num_envs);cfg.seed=a.seed
    cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn.texture_file=None
    # Relocate mesh paths ONLY in an output copy, never mutate baseline URDF/cache.
    original=Path(cfg.scene.robot.spawn.asset_path);tree=ET.parse(original)
    assert not any('wrist_up_mount' in e.get('name','') or 'wrist_down_mount' in e.get('name','') for e in tree.getroot())
    for mesh in tree.getroot().iter('mesh'):
        path=original.parent/'meshes'/Path(mesh.get('filename')).name
        if not path.is_file():raise FileNotFoundError(path)
        mesh.set('filename',str(path))
    relocated=out/'model.urdf';tree.write(relocated,encoding='utf-8',xml_declaration=True)
    cfg.scene.robot.spawn.asset_path=str(relocated);cfg.scene.robot.spawn.usd_dir=str(out/'usd_sdk_no_ff')
    cfg.log_dir=str(out)
    if a.mode!='train':cfg.observations.policy.enable_corruption=False
    save('reward_audit.json',audit);save('force_config.json',asdict(ForceConfig()))
    env=ManagerBasedRLEnv(cfg=cfg);obs,_=env.reset();robot=env.scene['robot'];c=term(env)
    assert type(robot.actuators['arm']).__name__=='DelayedPDActuator'
    armids=robot.find_joints([f'joint{i}' for i in range(1,7)],preserve_order=True)[0]
    bounds=robot.data.joint_pos_limits[0,armids].cpu()
    expected=torch.tensor([[-2.61799387799,2.61799387799],[0,2.87979326579],[-2.88559265359,0],[-1.51843644924,1.51843644924],[-1.34390352404,1.34390352404],[-2.79252680319,2.79252680319]])
    assert torch.allclose(bounds,expected,atol=2e-5)
    assert obs['estimator'].shape==(a.num_envs,9)
    dims=[obs['policy'].shape[1],obs['critic'].shape[1],env.action_manager.total_action_dim]
    assert dims[2]==22
    manifest={'base':audit['base'],'mode':a.mode,'seed':a.seed,'num_envs':a.num_envs,'observation_dimensions':dims,'estimator_labels':['base_velocity_xyz','external_force_yaw_xyz','ee_position_base_yaw_xyz'],'history_length':10,'arm_actuator':type(robot.actuators['arm']).__name__,'arm_limits':bounds.tolist(),'arm_gravity_feedforward':False,'physical_gravity':list(cfg.sim.gravity),'physics_dt':env.physics_dt,'control_dt':env.step_dt,'baseline_urdf_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'upstream_commit':'c760e1d74ad165d3c069d4f57ab5d066f6a41eb6','total_updates':a.iterations if a.mode=='train' else 2 if a.mode=='smoke' else 0,'checkpoint_resume_environment':'fresh randomized episodes; learner/optimizers/RNG/counter restored, not an exact continuation of PhysX state'}
    save('manifest.json',manifest)
    manifest['policy_terms'] = list(POLICY_TERMS)
    manifest['history_layout'] = 'term-major; each term oldest-to-newest; reset repeats first frame'
    manifest['code_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    save('manifest.json',manifest)
    model=CSEPolicy(*dims).to(a.device);learner=CSEPPO(model)
    if a.resume or a.checkpoint:learner.load(a.resume or a.checkpoint)
    env.training_step_offset=learner.iteration*learner.cfg.steps
    if a.mode=='smoke':
        # Bounded force/position mode exercise, not a learned-performance experiment.
        ids=torch.arange(a.num_envs,device=a.device);modes=ids%2==0
        c.override={'mode':modes,'free':torch.zeros_like(modes),'command':torch.tensor([20.,-10.,30.],device=a.device).expand(a.num_envs,-1)}
        c._update_command();observed=[];peak=0.;counts=0
        history = PolicyHistory(a.num_envs, a.device)
        torch.testing.assert_close(history.push(history.latest_frame(obs['policy'])), obs['policy'])
        history_error = 0.
        original_apply=c.apply_force
        def counted_apply():
            nonlocal counts
            counts+=1;original_apply()
        c.apply_force=counted_apply
        for i in range(a.steps):
            with torch.no_grad():obs,r,done,timeout,info=env.step(model(obs['policy']))
            assembled=history.push(history.latest_frame(obs['policy']),done|timeout)
            history_error=max(history_error,float((assembled-obs['policy']).abs().max()))
            torch.testing.assert_close(assembled,obs['policy'])
            assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(r).all()
            peak=max(peak,float(c.state.external_w.abs().max()))
            observed.append([i,float(c.state.external_w.abs().mean()),float(c.state.mode.float().mean()),int(done.sum())])
        assert counts==a.steps*cfg.decimation and peak>0
        assert float(c.state.external_w[~c.state.mode].abs().max())==0
        # Check partial reset does not retain force labels from previous episode.
        reset_ids=ids[:max(1,a.num_envs//2)];env._reset_idx(reset_ids)
        assert torch.count_nonzero(c.state.external_w[reset_ids])==0
        c.override=None;obs,_=env.reset()
        losses=[]
        for _ in range(2):batch,obs=learner.collect(env,obs);losses.append(learner.update(batch))
        learner.save(out/'smoke_checkpoint.pt',metadata=manifest);learner.export(out/'smoke_actor.jit')
        exported=torch.jit.load(str(out/'smoke_actor.jit'),map_location=a.device)
        with torch.no_grad():delta=float((model(obs['policy'])-exported(obs['policy'])).abs().max())
        assert delta<1e-5
        restored=CSEPPO(CSEPolicy(*dims).to(a.device));restored.load(out/'smoke_checkpoint.pt')
        assert restored.iteration==2
        assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in model.state_dict().items())
        save('smoke_result.json',{'pass':True,'physics_substeps_with_force_callback':counts,'maximum_external_force_component_N':peak,'reset_force_cleared':True,'finite':True,'losses':losses,'export_max_abs_error':delta,'inference_history_max_abs_error':history_error,'checkpoint_model_equal':True,'formal_training_started':False,'training_updates_for_pipeline_test_only':2,'no_claim_of_trained_force_accuracy':True})
        np.save(out/'force_smoke.npy',np.asarray(observed))
    elif a.mode=='train':
        if learner.iteration>=a.iterations:raise ValueError('Checkpoint already reaches total budget')
        while learner.iteration<a.iterations:
            batch,obs=learner.collect(env,obs);loss=learner.update(batch)
            save('status.json',{'stage':'training','completed_updates':learner.iteration,'total_updates':a.iterations,'loss':loss,'updated_at':time.time()})
            if learner.iteration%500==0 or learner.iteration==a.iterations:
                learner.save(out/f'checkpoint_{learner.iteration:07d}.pt',manifest);learner.export(out/f'actor_{learner.iteration:07d}.jit')
    else:
        rows=[]
        for i in range(a.steps):
            with torch.no_grad():
                estimate=model.estimator(obs['policy']);truth=obs['estimator'].clone();forcecmd=c.state.command.clone();mode=c.state.mode.clone()
                rows.append({'force_est_error':estimate[:,3:6]-truth[:,3:6],'force_track_error':truth[:,3:6]-forcecmd,'mode':mode})
                obs,_,_,_,_=env.step(model(obs['policy']))
        arr={k:torch.stack([x[k] for x in rows]).cpu().numpy() for k in rows[0]};np.savez_compressed(out/'evaluation.npz',**arr)
        metrics={}
        mask=arr['mode']
        for k in ['force_est_error','force_track_error']:
            x=arr[k][mask];metrics[k]={'mean_abs_per_axis_N':np.abs(x).mean(0).tolist(),'rmse_per_axis_N':np.sqrt((x*x).mean(0)).tolist()} if len(x) else {'no_force_frames':True}
        save('evaluation.json',metrics)
    env.close();save('status.json',{'stage':'complete','mode':a.mode,'updated_at':time.time(),'completed_updates':learner.iteration})
try:main()
except Exception:
    save('status.json',{'stage':'failed','traceback':traceback.format_exc()});raise
finally:app.close()
