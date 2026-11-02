#!/usr/bin/env bash
set -euo pipefail
source /mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab/scripts/repro_env.sh
policy=logs/rsl_rl/b2_z1_wbc/2026-09-14_14-24-22_repro_seed42_20260914_142415/exported/policy.pt
out=logs/b2_neowa_comparison_20260915
# Primary video first, then the two previously tested A2W heights.
for entry in 15cm:.68 10cm:.63 original:.53; do
    variant=${entry%%:*}
    height=${entry##*:}
    python scripts/repro_mujoco_neowa_compare.py \
        --robot b2_z1 --policy "$policy" --xml mujoco/robots/b2_z1/scene.xml \
        --output "$out/$variant" --stationary --text neowa \
        --width .072 --height .14 --spacing .088 --baseline-z "$height" \
        --forward .75 --plane-x .9239999999999999 --draw-speed .11
done
# The A2W branch must remain identical to the pre-existing recorded run.
python scripts/repro_mujoco_neowa_compare.py \
    --robot a2w_z1 \
    --policy logs/rsl_rl/a2w_z1_wbc/2026-09-14_14-25-49_repro_seed42_20260914_142542/exported/policy.pt \
    --xml mujoco/robots/a2w_z1/scene.xml --output "$out/a2w_regression_15cm" \
    --stationary --text neowa --width .072 --height .14 --spacing .088 \
    --baseline-z .68 --forward .75 --plane-x .9239999999999999 --draw-speed .11 --no-video
