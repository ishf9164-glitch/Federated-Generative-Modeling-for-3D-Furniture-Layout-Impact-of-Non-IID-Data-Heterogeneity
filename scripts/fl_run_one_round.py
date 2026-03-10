# diverse_synth/scripts/fl_run_one_round.py
# Run ONE federated local round using your original train_with_wandb.py command.
#
# Key idea:
# - train_with_wandb.py only resumes from: {checkpoint_dir}/{tag}/checkpoint.tar
# - It saves model weights ONLY in checkpoint_eval{epoch}.tar (not in final.tar).
# - It loops epoch in range(start_epoch, epochs+1) (inclusive),
#   so to run exactly 1 epoch, set epochs = start_epoch.
#
# Usage (from diverse_synth/scripts):
#   python fl_run_one_round.py --room bedroom --client_id 1 --round 1 \
#       --global_ckpt_path /path/to/global_checkpoint.tar \
#       --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P
#
# It will:
# - Create a temp config overriding training.tag/save_frequency/epochs
# - Write checkpoint.tar (based on a template) into checkpoint_dir/tag/
# - Run train_with_wandb.py
# - Print path of newest checkpoint_eval*.tar

import os
import io
import glob
import argparse
import shutil
import subprocess
from typing import Any, Dict, Optional

import yaml
import torch

# make project importable (same as train_with_wandb.py)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import load_config


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _load_torch(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _save_torch(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def _find_latest_checkpoint_eval(exp_dir: str) -> Optional[str]:
    cands = glob.glob(os.path.join(exp_dir, "checkpoint_eval*.tar"))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def _make_tag(room: str, client_id: int) -> str:
    # stable per-client tag (do NOT include round, otherwise resume breaks)
    return f"fl_{room}_client{client_id}"


def _merge_global_into_template(template: Dict[str, Any], global_ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """
    train_with_wandb.py expects:
      epoch, vae_state_dict, unet_state_dict, vae_optimizer_state_dict, unet_optimizer_state_dict
    We keep optimizer states from template, and overwrite model weights/epoch using global_ckpt.
    """
    out = dict(template)

    if "vae_state_dict" in global_ckpt:
        out["vae_state_dict"] = global_ckpt["vae_state_dict"]
    if "unet_state_dict" in global_ckpt:
        out["unet_state_dict"] = global_ckpt["unet_state_dict"]

    # epoch controls start_epoch in train_with_wandb.py
    if "epoch" in global_ckpt:
        out["epoch"] = int(global_ckpt["epoch"])
    elif "epoch" not in out:
        out["epoch"] = 0

    # keep kernel_mask_dict if present
    if "kernel_mask_dict" in global_ckpt:
        out["kernel_mask_dict"] = global_ckpt["kernel_mask_dict"]

    # Required optimizer keys (must exist, otherwise train script will KeyError)
    if "vae_optimizer_state_dict" not in out or "unet_optimizer_state_dict" not in out:
        raise ValueError("Template checkpoint must contain optimizer state dicts.")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", choices=["bedroom", "livingroom", "diningroom"], required=True)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)

    parser.add_argument("--generator_type", default="EnhancedBetaTCVAE")
    parser.add_argument("--discriminator_type", default="UNet3P")

    parser.add_argument("--config_file", default="", help="Optional explicit config yaml path")
    parser.add_argument("--local_epochs", type=int, default=1, help="How many epochs per FL round (default=1)")

    parser.add_argument("--global_ckpt_path", default="", help="Global checkpoint.tar path (torch.load-able dict)")
    parser.add_argument("--template_ckpt_path", default="", help="Template checkpoint.tar path to preserve optimizer states")

    parser.add_argument("--wandb_entity", default="", help="Optional wandb entity; leave empty to disable wandb")
    args = parser.parse_args()

    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    # base config
    cfg_path = args.config_file
    if not cfg_path:
        cfg_path = os.path.normpath(os.path.join(scripts_dir, f"../config/{args.room}_config.yaml"))
    base_cfg = _read_yaml(cfg_path)

    # derive checkpoint_dir from config
    ckpt_dir = base_cfg["training"]["checkpoint_dir"]
    tag = _make_tag(args.room, args.client_id)
    exp_dir = os.path.join(ckpt_dir, tag)
    os.makedirs(exp_dir, exist_ok=True)

    # load template checkpoint
    # priority: existing local checkpoint.tar -> --template_ckpt_path
    local_checkpoint_tar = os.path.join(exp_dir, "checkpoint.tar")
    template_path = local_checkpoint_tar if os.path.isfile(local_checkpoint_tar) else args.template_ckpt_path
    if not template_path or not os.path.isfile(template_path):
        raise FileNotFoundError(
            "Template checkpoint not found. You must create one first with fl_init_global_ckpt.py "
            "and pass it via --template_ckpt_path (or ensure local checkpoint.tar exists)."
        )
    template_ckpt = _load_torch(template_path)

    # load global checkpoint (if not provided, treat as round0 template itself)
    if args.global_ckpt_path:
        global_ckpt = _load_torch(args.global_ckpt_path)
    else:
        global_ckpt = template_ckpt

    # merge global weights into template (keep optimizer states)
    merged_ckpt = _merge_global_into_template(template_ckpt, global_ckpt)

    # IMPORTANT: to run exactly N epochs, because training loop is inclusive:
    #   for epoch in range(start_epoch, epochs+1)
    # set epochs = start_epoch + (local_epochs-1)
    start_epoch = int(merged_ckpt.get("epoch", 0))
    run_until = start_epoch + (args.local_epochs - 1)

    # make temp config overriding tag / epochs / save_frequency
    tmp_cfg = dict(base_cfg)
    tmp_cfg.setdefault("training", {})
    tmp_cfg["training"]["tag"] = tag
    tmp_cfg["training"]["save_frequency"] = 1
    tmp_cfg["training"]["epochs"] = run_until

    tmp_cfg_path = os.path.join("/tmp", f"fl_{args.room}_c{args.client_id}_r{args.round}.yaml")
    _write_yaml(tmp_cfg_path, tmp_cfg)

    # write checkpoint.tar for train_with_wandb.py to resume
    checkpoint_tar_path = os.path.join(exp_dir, "checkpoint.tar")
    _save_torch(merged_ckpt, checkpoint_tar_path)

    # run training command (disable wandb by default to avoid account dependency)
    env = os.environ.copy()
    if not args.wandb_entity:
        env["WANDB_MODE"] = "disabled"
        env["WANDB_SILENT"] = "true"

    cmd = [
        "python", "train_with_wandb.py",
        "--config_file", tmp_cfg_path,
        "--generator_type", args.generator_type,
        "--discriminator_type", args.discriminator_type,
    ]
    if args.wandb_entity:
        cmd += ["--wandb_entity", args.wandb_entity]

    print(f"[INFO] round={args.round} client={args.client_id} room={args.room}")
    print(f"[INFO] tag={tag}")
    print(f"[INFO] start_epoch={start_epoch}, run_until={run_until} (local_epochs={args.local_epochs})")
    print(f"[INFO] checkpoint.tar -> {checkpoint_tar_path}")
    print(f"[INFO] cmd: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=scripts_dir, check=True, env=env)

    latest_eval = _find_latest_checkpoint_eval(exp_dir)
    if not latest_eval:
        raise FileNotFoundError(
            f"No checkpoint_eval*.tar found in {exp_dir}. "
            "Make sure save_frequency=1 and training ran at least 1 epoch."
        )

    print(f"[OK] updated checkpoint: {latest_eval}")


if __name__ == "__main__":
    main()
