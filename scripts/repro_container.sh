#!/usr/bin/env bash
# Run only this project's code in a fresh container; no unrelated workspaces are mounted.
set -euo pipefail
gpu="${1:?physical GPU}"; name="${2:?container name}"; shift 2
root=/mnt/sda/ps/leggedmanip-lab
exec docker run -d --name "$name" --gpus "device=$gpu" --user "$(id -u):$(id -g)" \
  --shm-size=8g --log-driver local --log-opt max-size=10m --log-opt max-file=2 \
  --read-only --tmpfs /tmp:rw,size=4g,mode=1777 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  -v /usr/share/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json:ro \
  -v "$root:$root" \
  -v /home/ps/miniconda3/envs/isaacsim5.1:/home/ps/miniconda3/envs/isaacsim5.1:ro \
  robodojo-ps1-runtime:cuda12.8 "$@"
