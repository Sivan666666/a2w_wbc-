"""Sampling/gating used by both PPO fine-tuning branches."""
import torch

def gate(command):
    # Full penalty <=.08 on every axis; smooth taper to zero at .1.
    u=((.1-command.abs().amax(dim=-1))/.02).clamp(0.,1.)
    return u*u*(3.-2.*u)

def ramp(step,final_actual,num_steps_per_env=24,warmup_iterations=500):
    alpha=min(max(step/(num_steps_per_env*warmup_iterations),0.),1.)
    return alpha,alpha

def sample_commands(n,device,bounds):
    lo=torch.tensor([v[0] for v in bounds],device=device)
    hi=torch.tensor([v[1] for v in bounds],device=device)
    category_draw=torch.rand(n,device=device)
    mode=torch.where(category_draw<.2,0,torch.where(category_draw<.4,1,2))
    cmd=lo+torch.rand(n,3,device=device)*(hi-lo)
    cmd[mode==0]=0.
    slow=mode==1
    cmd[slow]=0.
    cmd[slow,0]=torch.rand(int(slow.sum()),device=device)*.2-.1
    # Normal movement is the original uniform box conditioned on being outside
    # the low-speed gate, so no extra standing mass is accidentally introduced.
    reject=(mode==2)&(cmd.abs()<.1).all(dim=-1)
    while bool(reject.any()):
        cmd[reject]=lo+torch.rand(int(reject.sum()),3,device=device)*(hi-lo)
        reject=(mode==2)&(cmd.abs()<.1).all(dim=-1)
    return cmd,mode
