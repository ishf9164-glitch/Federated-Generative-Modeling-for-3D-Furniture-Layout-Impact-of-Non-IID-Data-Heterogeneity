# diverse_synth/scripts/fl_init_global_ckpt.py
# Create an initial/template checkpoint.tar that is compatible with train_with_wandb.py
# It includes:
#   epoch, vae_state_dict, unet_state_dict,
#   vae_optimizer_state_dict, unet_optimizer_state_dict,
#   (optional) kernel_mask_dict
#
# Usage (from diverse_synth/scripts):
#   python fl_init_global_ckpt.py --room bedroom --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P

import os
import argparse
import torch

# make project importable (same as train_with_wandb.py)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import load_config
from synthesis import NetworkBuilder, Optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", choices=["bedroom", "livingroom", "diningroom"], default="bedroom")
    parser.add_argument("--config_file", default="", help="Optional explicit config yaml path")
    parser.add_argument("--generator_type", default="EnhancedBetaTCVAE")
    parser.add_argument("--discriminator_type", default="UNet3P")
    parser.add_argument("--tag", default="", help="Override training.tag for template")
    parser.add_argument("--checkpoint_dir", default="", help="Override training.checkpoint_dir for template")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    cfg_path = args.config_file
    if not cfg_path:
        cfg_path = os.path.normpath(os.path.join(os.path.dirname(__file__), f"../config/{args.room}_config.yaml"))

    config = load_config(cfg_path)

    # Override tag/dir if requested
    if args.tag:
        config["training"]["tag"] = args.tag
    if args.checkpoint_dir:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir

    tag = config["training"]["tag"]
    ckpt_dir = config["training"]["checkpoint_dir"]

    # Choose device
    device = torch.device("cuda:0") if (args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

    # Ensure dirs
    exp_dir = os.path.join(ckpt_dir, tag)
    os.makedirs(exp_dir, exist_ok=True)
    out_path = os.path.join(exp_dir, "checkpoint.tar")

    # Build networks
    vae = NetworkBuilder.build_network(
        config,
        network_type=args.generator_type,
        device=device,
        kernel_mask_dict=None
    )
    unet = NetworkBuilder.build_network(config, network_type=args.discriminator_type, device=device)

    # Build optimizers (same hyperparams as train_with_wandb.py)
    vae_opt = Optimizer.build_optimizer(
        vae.parameters(),
        lr=config["training"].get("lr", 1e-3),
        optimizer=config["training"].get("optimizer", "Adam"),
        momentum=config["training"].get("momentum", 0.9),
        weight_decay=config["training"].get("weight_decay", 0.0),
        betas=None
    )
    unet_opt = Optimizer.build_optimizer(
        unet.parameters(),
        lr=config["training"].get("lr", 1e-3),
        optimizer=config["training"].get("optimizer", "Adam"),
        momentum=config["training"].get("momentum", 0.9),
        weight_decay=config["training"].get("weight_decay", 0.0),
        betas=None
    )

    # Collect state dicts (handle DataParallel style)
    try:
        vae_sd = vae.module.state_dict()
    except Exception:
        vae_sd = vae.state_dict()

    try:
        unet_sd = unet.module.state_dict()
    except Exception:
        unet_sd = unet.state_dict()

    save_dict = {
        "epoch": 0,
        "vae_state_dict": vae_sd,
        "unet_state_dict": unet_sd,
        "vae_optimizer_state_dict": vae_opt.state_dict(),
        "unet_optimizer_state_dict": unet_opt.state_dict(),
    }

    # kernel_mask_dict is used by EnhancedBetaTCVAE in your code
    if hasattr(vae, "kernel_mask_dict"):
        save_dict["kernel_mask_dict"] = vae.kernel_mask_dict

    torch.save(save_dict, out_path)
    print(f"[OK] template checkpoint saved: {out_path}")
    print(f"     room={args.room}, tag={tag}, device={device}")


if __name__ == "__main__":
    main()
