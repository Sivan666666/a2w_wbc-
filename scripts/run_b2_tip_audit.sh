#!/usr/bin/env bash
set -euo pipefail
source /mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab/scripts/repro_env.sh
for c in end_effector_1kg usd_arm_mass_tip usd_arm_mass_tip_no_passive; do
    python scripts/audit_b2_bias_mujoco.py --robot b2_z1 \
        --audit-case "$c" --physics-audit logs/b2_bias_audit_20260915/physics.json \
        --policy logs/rsl_rl/b2_z1_wbc/2026-09-14_14-24-22_repro_seed42_20260914_142415/exported/policy.pt \
        --xml mujoco/robots/b2_z1/scene.xml --output "logs/b2_bias_audit_20260915/$c" \
        --stationary --text neowa --width .072 --height .14 --spacing .088 \
        --baseline-z .68 --forward .75 --plane-x .9239999999999999 --draw-speed .11 --no-video
done
