"""Export the supplied Precision PPO checkpoint's deterministic actor on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/model_10000.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/ppo10k_actor.jit"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = checkpoint["actor_state_dict"]
    # This checkpoint has Identity obs_normalizer; distribution.log_std_param
    # is only used for training sampling, never for deterministic deployment.
    model = nn.Sequential(nn.Linear(234, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                          nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 22)).eval()
    model.load_state_dict({k.removeprefix("mlp."): v for k, v in source.items() if k.startswith("mlp.")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(model).save(str(args.output))
    scripted = torch.jit.load(str(args.output)).eval()
    torch.manual_seed(42)
    obs = torch.randn(1024, 234)
    with torch.inference_mode():
        error = (scripted(obs) - model(obs)).abs().max().item()
    report = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "actor_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "checkpoint_iteration": checkpoint.get("iter"),
              "input": 234, "output": 22, "activation": "ELU",
              "normalizer": "Identity", "torch": torch.__version__,
              "export_parity_max_abs": error, "actor_state_keys": list(source)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
