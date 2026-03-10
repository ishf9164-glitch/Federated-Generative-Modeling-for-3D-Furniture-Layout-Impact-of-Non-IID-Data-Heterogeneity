# FedAvg.py
# Single-GPU Federated Averaging (FedAvg) simulation:
# Each round: dense -> neutral -> sparse (sequential local training), then server aggregation.

import argparse
import logging
import os
import sys
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import load_config
from synthesis.datasets.Common import filter_function
from synthesis.datasets.FrontDataset import get_encoded_dataset
from synthesis import NetworkBuilder, Optimizer


# -----------------------------
# Utils: FedAvg on state_dict
# -----------------------------
def fedavg_state_dict(state_dicts, weights):
    """
    state_dicts: list[OrderedDict[str, Tensor]]
    weights: list[float] (sum to 1)
    """
    assert len(state_dicts) == len(weights) and len(state_dicts) > 0
    out = OrderedDict()

    keys = state_dicts[0].keys()
    for k in keys:
        v0 = state_dicts[0][k]
        # only average floating tensors; keep non-float buffers from the first client
        if torch.is_tensor(v0) and v0.dtype.is_floating_point:
            acc = None
            for sd, w in zip(state_dicts, weights):
                vv = sd[k].detach().float().cpu()
                acc = vv * w if acc is None else acc + vv * w
            out[k] = acc.to(v0.dtype)
        else:
            out[k] = v0
    return out


def copy_state_dict_to_cpu(sd: OrderedDict):
    cpu_sd = OrderedDict()
    for k, v in sd.items():
        if torch.is_tensor(v):
            cpu_sd[k] = v.detach().cpu().clone()
        else:
            cpu_sd[k] = deepcopy(v)
    return cpu_sd


# -----------------------------
# Train / Eval (reuse your logic)
# -----------------------------
def local_train_epochs(
    vae, unet,
    train_loader,
    config,
    device,
    local_epochs,
    lr,
    optimizer_name,
    momentum,
    weight_decay,
):
    """
    Local training for one client starting from current model weights.
    Returns:
        avg_train_total_loss (float)
    NOTE: optimizer state is RESET each time (typical for simple FedAvg simulation).
    """
    vae_optimizer = Optimizer.build_optimizer(
        vae.parameters(),
        lr=lr,
        optimizer=optimizer_name,
        momentum=momentum,
        weight_decay=weight_decay,
        betas=None
    )
    unet_optimizer = Optimizer.build_optimizer(
        unet.parameters(),
        lr=lr,
        optimizer=optimizer_name,
        momentum=momentum,
        weight_decay=weight_decay,
        betas=None
    )

    vae.train()
    unet.train()

    total_loss_sum = 0.0
    total_batches = 0

    for _ in range(local_epochs):
        stat_total = 0.0
        batches = 0

        for _, batch_data_label in enumerate(tqdm(train_loader, leave=False)):
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)

            vae_optimizer.zero_grad()
            unet_optimizer.zero_grad()

            inputs_abs = batch_data_label['x_abs']
            labels_abs = batch_data_label['x_abs']
            labels_rel = batch_data_label['x_rel']

            package = vae(inputs_abs)
            feature_x = unet(package[-1].detach())

            vae_loss_dict, current_idx, ground_truth_idx, reconstruct_idx = vae.loss_function(
                ground_truth=labels_abs,
                package=package,
                dataset_size=len(train_loader.dataset),
            )

            unet_loss_dict = unet.loss_function(
                room_type=config['data']['room_type'],
                ground_truth=labels_rel,
                reconstruct_x=feature_x,
                batch_idx=current_idx,
                reconstruct_idx=reconstruct_idx,
                ground_truth_idx=ground_truth_idx
            )

            loss_dict = {**vae_loss_dict, **unet_loss_dict}
            total_loss = torch.zeros(1, device=device)
            for k, v in loss_dict.items():
                if k == 'mi_loss':
                    continue
                total_loss += v
            total_loss.backward()

            vae_optimizer.step()
            unet_optimizer.step()

            stat_total += total_loss.item()
            batches += 1

        if batches > 0:
            total_loss_sum += stat_total / batches
            total_batches += 1

    avg_train_total_loss = total_loss_sum / max(1, total_batches)
    return avg_train_total_loss


@torch.no_grad()
def evaluate_total_loss(vae, unet, val_loader, config, device):
    vae.eval()
    unet.eval()

    stat_total = 0.0
    batches = 0

    for _, batch_data_label in enumerate(tqdm(val_loader, leave=False)):
        for k in batch_data_label:
            batch_data_label[k] = batch_data_label[k].to(device)

        inputs_abs = batch_data_label['x_abs']
        labels_abs = batch_data_label['x_abs']
        labels_rel = batch_data_label['x_rel']

        package = vae(inputs_abs)
        feature_x = unet(package[-1].detach())

        vae_loss_dict, current_idx, ground_truth_idx, reconstruct_idx = vae.loss_function(
            ground_truth=labels_abs,
            package=package,
            dataset_size=len(val_loader.dataset),
        )

        unet_loss_dict = unet.loss_function(
            room_type=config['data']['room_type'],
            ground_truth=labels_rel,
            reconstruct_x=feature_x,
            batch_idx=current_idx,
            reconstruct_idx=reconstruct_idx,
            ground_truth_idx=ground_truth_idx
        )

        eval_loss_dict = {**vae_loss_dict, **unet_loss_dict}
        total_loss = torch.zeros(1, device=device)
        for k, v in eval_loss_dict.items():
            if k == 'mi_loss':
                continue
            total_loss += v

        stat_total += total_loss.item()
        batches += 1

    return stat_total / max(1, batches)


# -----------------------------
# Data loaders per client
# -----------------------------
def build_loaders(config, args):
    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        split=config["training"].get("splits", ["train", "val"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 16),
        num_workers=args.n_processes,
        worker_init_fn=train_dataset.worker_init_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # bounds file (per-client, consistent with your original pipeline)
    # use each client's own bounds for its validation set (keeps code simple)
    # Note: for FL, weights aggregation doesn't depend on bounds.
    exp_tag = config['training'].get('tag', 'exp')
    ckpt_dir = config['training']['checkpoint_dir']
    exp_dir = os.path.join(ckpt_dir, exp_tag)
    os.makedirs(exp_dir, exist_ok=True)

    path_to_bounds = os.path.join(exp_dir, "bounds.npz")
    np.savez(
        path_to_bounds,
        sizes=train_dataset.bounds["sizes"],
        translations=train_dataset.bounds["translations"],
        angles=train_dataset.bounds["angles"]
    )

    val_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=path_to_bounds,
        split=config["validation"].get("splits", ["test"])
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["validation"].get("batch_size", 16),
        num_workers=args.n_processes,
        worker_init_fn=val_dataset.worker_init_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    return train_loader, val_loader


# -----------------------------
# Main: FedAvg rounds
# -----------------------------
def main(argv):
    parser = argparse.ArgumentParser("Single-GPU FedAvg Simulation")
    # parser.add_argument("--config_bedroom", required=True, help="bedroom client config yaml")
    # parser.add_argument("--config_livingroom", required=True, help="livingroom client config yaml")
    # parser.add_argument("--config_diningroom", required=True, help="diningroom client config yaml")
    # parser.add_argument("--config_dense", required=True, help="dense client config yaml")
    # parser.add_argument("--config_neutral", required=True, help="neutral client config yaml")
    # parser.add_argument("--config_sparse", required=True, help="sparse client config yaml")
    # parser.add_argument("--config_IID_client1", required=True, help="IID_client1 client config yaml")
    # parser.add_argument("--config_IID_client2", required=True, help="IID_client2 client config yaml")
    # parser.add_argument("--config_IID_client3", required=True, help="IID_client3 client config yaml")
    parser.add_argument("--config_quantity_client1", required=True, help="quantity_client1 client config yaml")
    parser.add_argument("--config_quantity_client2", required=True, help="quantity_client2 client config yaml")
    parser.add_argument("--config_quantity_client3", required=True, help="quantity_client3 client config yaml")
    # parser.add_argument("--config_compound_client1", required=True, help="compound_client1 client config yaml")
    # parser.add_argument("--config_compound_client2", required=True, help="compound_client2 client config yaml")
    # parser.add_argument("--config_compound_client3", required=True, help="compound_client3 client config yaml")

    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--n_processes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--generator_type", default="EnhancedBetaTCVAE")
    parser.add_argument("--discriminator_type", default="UNet3P")

    parser.add_argument("--wandb_entity", required=True)
    parser.add_argument("--wandb_project", default="diverse-synth")
    parser.add_argument("--run_name", default="fedavg_compound_client")

    args = parser.parse_args(argv)

    logging.getLogger("trimesh").setLevel(logging.ERROR)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    # Load client configs (order is important: bedroom -> livingroom -> diningroom)
    cfg_quantity_client1 = load_config(args.config_quantity_client1)
    cfg_quantity_client2 = load_config(args.config_quantity_client2)
    cfg_quantity_client3 = load_config(args.config_quantity_client3)

    clients = [
        ("quantity_client1", cfg_quantity_client1),
        ("quantity_client2", cfg_quantity_client2),
        ("quantity_client3", cfg_quantity_client3),
    ]

    # Init wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config={
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "clients_order": [c[0] for c in clients],
            "generator_type": args.generator_type,
            "discriminator_type": args.discriminator_type,
            "device": str(device),
        }
    )

    # Build loaders per client
    loaders = {}
    for cname, cfg in clients:
        # --- 新增调试代码：打印该 client 读取的 stats 路径 ---
        stats_name = cfg["data"].get("train_stats", "dataset_stats.txt")
        data_dir = cfg["data"].get("dataset_directory", "")
        # 通常代码逻辑会拼接路径，这里我们模拟一下常见的拼接方式
        potential_stats_path = os.path.join(data_dir, stats_name)
        
        print(f"\n[DEBUG] Client: {cname}")
        print(f"        Dataset Dir: {os.path.abspath(data_dir)}")
        print(f"        Stats Filename: {stats_name}")
        print(f"        Expected Stats Path: {os.path.abspath(potential_stats_path)}")
        # ----------------------------------------------

        tr, va = build_loaders(cfg, args)
        loaders[cname] = (tr, va)
        
        # --- 新增调试代码：打印实际加载出来的类别数量 ---
        labels = tr.dataset.class_labels
        print(f"        Loaded Labels Count: {len(labels)}")
        print(f"        Labels: {labels[:5]}... (total {len(labels)})")
        # ----------------------------------------------
        print(f"[{cname}] train scenes={len(tr.dataset)}, val scenes={len(va.dataset)}")

    # Sanity: class labels must match across clients
    base_labels = loaders["quantity_client1"][0].dataset.class_labels
    for cname in ["quantity_client2", "quantity_client3"]:
        assert loaders[cname][0].dataset.class_labels == base_labels, f"class_labels mismatch: {cname}"

    # -----------------------------
    # Build GLOBAL model once (dense config for architecture)
    # Load kernel_mask_dict (and optionally weights) from checkpoint.tar to avoid fan_in=0
    # -----------------------------
    from copy import deepcopy  # 确保文件顶部或这里 import 了 deepcopy
    
    kernel_mask_dict = None
    
    ckpt_dir = cfg_quantity_client1["training"]["checkpoint_dir"]
    tag = cfg_quantity_client1["training"].get("tag")
    checkpoint_path = os.path.join(ckpt_dir, tag, "checkpoint.tar")
    
    checkpoint = None
    if os.path.isfile(checkpoint_path):
        print(f"[Global] load checkpoint path: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        kernel_mask_dict = checkpoint.get("kernel_mask_dict")
    
    vae = NetworkBuilder.build_network(
        cfg_quantity_client1,
        network_type=args.generator_type,
        device=device,
        kernel_mask_dict=kernel_mask_dict
    )
    unet = NetworkBuilder.build_network(
        cfg_quantity_client1,
        network_type=args.discriminator_type,
        device=device
    )
    
    # 可选：如果 checkpoint 里也存了权重，就从该 checkpoint 继续训练
    if checkpoint is not None and "vae_state_dict" in checkpoint and "unet_state_dict" in checkpoint:
        vae.load_state_dict(checkpoint["vae_state_dict"], strict=True)
        unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
        print("[Global] Successfully Load Model...")


    # Global hyperparams (use dense cfg training block as reference; you can change to fixed constants)
    lr = cfg_quantity_client1["training"].get("lr", 1e-3)
    optimizer_name = cfg_quantity_client1["training"].get("optimizer", "Adam")
    momentum = cfg_quantity_client1["training"].get("momentum", 0.9)
    weight_decay = cfg_quantity_client1["training"].get("weight_decay", 0.0)

    # FedAvg rounds
    for r in range(1, args.rounds + 1):
        print(f"\n================ ROUND {r:03d} ================\n")

        # snapshot global weights (CPU copies)
        global_vae_sd = copy_state_dict_to_cpu(vae.state_dict())
        global_unet_sd = copy_state_dict_to_cpu(unet.state_dict())

        client_vae_sds = []
        client_unet_sds = []
        client_sizes = []

        # local training sequentially: dense -> neutral -> sparse
        for cname, cfg in clients:
            tr_loader, _ = loaders[cname]
            n_k = len(tr_loader.dataset)
            client_sizes.append(n_k)

            # load global weights into model (start from same global each client)
            vae.load_state_dict(global_vae_sd, strict=True)
            unet.load_state_dict(global_unet_sd, strict=True)
            vae.to(device)
            unet.to(device)

            print(f"[Client {cname}] local train start (n={n_k}, E={args.local_epochs})")
            local_train_loss = local_train_epochs(
                vae, unet, tr_loader, cfg, device,
                local_epochs=args.local_epochs,
                lr=lr,
                optimizer_name=optimizer_name,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            print(f"[Client {cname}] local train done, avg_total_loss={local_train_loss:.6f}")

            # store client weights to CPU
            client_vae_sds.append(copy_state_dict_to_cpu(vae.state_dict()))
            client_unet_sds.append(copy_state_dict_to_cpu(unet.state_dict()))

            wandb.log({
                "round": r,
                f"{cname}/local_train_total_loss": local_train_loss,
                f"{cname}/n_train": n_k
            })

            # free gpu cache
            torch.cuda.empty_cache()

        # Server aggregation (FedAvg)
        total = float(sum(client_sizes))
        weights = [n / total for n in client_sizes]

        new_vae_sd = fedavg_state_dict(client_vae_sds, weights)
        new_unet_sd = fedavg_state_dict(client_unet_sds, weights)

        vae.load_state_dict(new_vae_sd, strict=True)
        unet.load_state_dict(new_unet_sd, strict=True)

        # Global evaluation: evaluate on each client's val, report weighted mean
        eval_losses = {}
        eval_weighted = 0.0
        for (cname, cfg), w in zip(clients, weights):
            _, va_loader = loaders[cname]
            loss = evaluate_total_loss(vae, unet, va_loader, cfg, device)
            eval_losses[cname] = loss
            eval_weighted += w * loss

        log_dict = {"round": r, "global/eval_total_loss_weighted": eval_weighted}
        for cname in eval_losses:
            log_dict[f"{cname}/eval_total_loss"] = eval_losses[cname]

        print(f"[Server] aggregated eval_total_loss (weighted) = {eval_weighted:.6f}")
        for cname in eval_losses:
            print(f"  - {cname}: {eval_losses[cname]:.6f}")

        wandb.log(log_dict)

        # Save global checkpoint each round (resumable)
        # Use dense checkpoint_dir/tag as global output root
        ckpt_dir = cfg_quantity_client1["training"]["checkpoint_dir"]
        tag = cfg_quantity_client1["training"].get("tag", "fedavg_global")
        out_dir = os.path.join(ckpt_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        save_dict = {
            "round": r,
            "vae_state_dict": copy_state_dict_to_cpu(vae.state_dict()),
            "unet_state_dict": copy_state_dict_to_cpu(unet.state_dict()),
            "clients_order": [c[0] for c in clients],
            "weights": weights,
        }
        # keep kernel_mask_dict for EnhancedBetaTCVAE if present
        if hasattr(vae, "kernel_mask_dict"):
            save_dict["kernel_mask_dict"] = deepcopy(vae.kernel_mask_dict)

        torch.save(save_dict, os.path.join(out_dir, "checkpoint.tar"))

    wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
