# client.py
# Federated Client (Docker container)
# - Connects to FL server via SocketIO(WebSocket)
# - Receives global checkpoint (base64-encoded torch checkpoint.tar content)
# - Writes it to {checkpoint_dir}/{tag}/checkpoint.tar so train_with_wandb.py can resume
# - Runs ONE local training round using your original command (bedroom/livingroom/diningroom)
# - Finds the newest checkpoint_eval*.tar (or final.tar) and uploads it back to server

import os
import io
import time
import glob
import base64
import tempfile
import subprocess
from typing import Any, Dict, Optional, Tuple

import yaml
import torch
import socketio

# =========================
# Client identity / routing
# =========================
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))
ROOM_TYPE = os.getenv("ROOM_TYPE", "bedroom")  # bedroom | livingroom | diningroom
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "1"))  # FedAvg weight (optional)

# =========================
# Server connection
# =========================
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:5000")

# =========================
# Training command params (your original commands)
# =========================
GENERATOR_TYPE = os.getenv("GENERATOR_TYPE", "EnhancedBetaTCVAE")
DISCRIMINATOR_TYPE = os.getenv("DISCRIMINATOR_TYPE", "UNet3P")

# Project layout assumptions (consistent with your commands)
TRAIN_WORKDIR = os.getenv("TRAIN_WORKDIR", "/workspace/diverse_synth/scripts")  # contains train_with_wandb.py
BASE_CONFIG_DIR = os.path.normpath(os.path.join(TRAIN_WORKDIR, "../config"))
BASE_SAVEPOINT_DIR = os.path.normpath(os.path.join(TRAIN_WORKDIR, "../savepoint"))

CONFIG_FILE_MAP = {
    "bedroom": os.path.join(BASE_CONFIG_DIR, "bedroom_config.yaml"),
    "livingroom": os.path.join(BASE_CONFIG_DIR, "livingroom_config.yaml"),
    "diningroom": os.path.join(BASE_CONFIG_DIR, "diningroom_config.yaml"),
}

# One-round training budget
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "1"))  # each FL round runs 1 epoch by default

# How to locate the updated checkpoint after local training
# (train_with_wandb.py saves checkpoint_eval{epoch}.tar and final.tar under {checkpoint_dir}/{tag}/)
CKPT_PATTERN = os.getenv("CKPT_PATTERN", "checkpoint_eval*.tar")  # fallback to final.tar

# Unique tag per client to avoid collisions
TAG = os.getenv("TAG", f"fl_client{CLIENT_ID}_{ROOM_TYPE}")


# =========================
# SocketIO client
# =========================
sio = socketio.Client(logger=False, engineio_logger=False)


def b64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))


def bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def set_first_key_recursive(d: Any, key: str, value: Any) -> bool:
    """
    Tries to set the first occurrence of `key` in nested dicts.
    Returns True if set, False otherwise.
    """
    if isinstance(d, dict):
        if key in d:
            d[key] = value
            return True
        for _, v in d.items():
            if set_first_key_recursive(v, key, value):
                return True
    elif isinstance(d, list):
        for v in d:
            if set_first_key_recursive(v, key, value):
                return True
    return False


def prepare_round_config(base_cfg_path: str, tag: str, local_epochs: int) -> Tuple[str, str]:
    """
    Creates a temporary YAML config for this client/round to ensure:
    - unique tag (so multiple clients don't overwrite checkpoints)
    - checkpoint_dir points to ../savepoint (default)
    - epochs set to LOCAL_EPOCHS (best-effort)
    Returns (temp_cfg_path, checkpoint_dir_used)
    """
    cfg = read_yaml(base_cfg_path)

    # best-effort overrides (key names may differ across projects)
    # 1) tag
    set_first_key_recursive(cfg, "tag", tag)

    # 2) checkpoint_dir (if exists in config, enforce it)
    # use BASE_SAVEPOINT_DIR by default
    ckpt_dir_used = BASE_SAVEPOINT_DIR
    if set_first_key_recursive(cfg, "checkpoint_dir", ckpt_dir_used):
        pass

    # 3) epochs / save_frequency (best effort)
    set_first_key_recursive(cfg, "epochs", local_epochs)
    set_first_key_recursive(cfg, "save_frequency", 1)

    # Write temp config
    os.makedirs("/tmp/fl_configs", exist_ok=True)
    temp_cfg_path = os.path.join("/tmp/fl_configs", f"{tag}_temp.yaml")
    write_yaml(temp_cfg_path, cfg)
    return temp_cfg_path, ckpt_dir_used


def write_global_checkpoint(global_ckpt_bytes: bytes, checkpoint_dir: str, tag: str) -> str:
    """
    Writes global checkpoint bytes into {checkpoint_dir}/{tag}/checkpoint.tar
    so train_with_wandb.py can resume from it.
    Returns the written path.
    """
    out_dir = os.path.join(checkpoint_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "checkpoint.tar")
    with open(out_path, "wb") as f:
        f.write(global_ckpt_bytes)
    return out_path


def run_local_training(temp_cfg_path: str) -> None:
    """
    Executes your original command, except we pass a temp config file
    (same script, same generator/discriminator args).
    """
    cmd = [
        "python", "train_with_wandb.py",
        "--config_file", temp_cfg_path,
        "--generator_type", GENERATOR_TYPE,
        "--discriminator_type", DISCRIMINATOR_TYPE,
    ]
    print(f"[client{CLIENT_ID}] run: {' '.join(cmd)} (cwd={TRAIN_WORKDIR})")
    subprocess.run(cmd, cwd=TRAIN_WORKDIR, check=True)


def find_latest_updated_ckpt(checkpoint_dir: str, tag: str) -> Optional[str]:
    """
    Finds the newest checkpoint file under {checkpoint_dir}/{tag}/
    Prefer checkpoint_eval*.tar; fallback to final.tar.
    """
    base = os.path.join(checkpoint_dir, tag)
    cand = glob.glob(os.path.join(base, CKPT_PATTERN))
    if not cand:
        final_path = os.path.join(base, "final.tar")
        if os.path.exists(final_path):
            return final_path
        return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def do_one_round(round_idx: int, global_ckpt_b64: str) -> None:
    """
    Full client round:
    - decode global ckpt bytes
    - prepare temp config
    - write checkpoint.tar
    - run training
    - locate newest updated ckpt
    - upload updated ckpt bytes to server
    """
    if ROOM_TYPE not in CONFIG_FILE_MAP:
        raise ValueError(f"Unsupported ROOM_TYPE={ROOM_TYPE}, must be one of {list(CONFIG_FILE_MAP.keys())}")

    base_cfg = CONFIG_FILE_MAP[ROOM_TYPE]
    if not os.path.exists(base_cfg):
        raise FileNotFoundError(f"Config file not found: {base_cfg}")

    # Prepare round config (unique tag + 1 epoch best-effort)
    temp_cfg_path, ckpt_dir_used = prepare_round_config(base_cfg, TAG, LOCAL_EPOCHS)

    # Write global checkpoint into checkpoint.tar
    global_bytes = b64_to_bytes(global_ckpt_b64) if global_ckpt_b64 else b""
    # If server sends empty (round 0 init), create a minimal checkpoint dict so torch.load won't break
    if not global_bytes:
        obj = {"epoch": 0, "vae_state_dict": {}, "unet_state_dict": {}}
        buf = io.BytesIO()
        torch.save(obj, buf)
        global_bytes = buf.getvalue()

    ckpt_path = write_global_checkpoint(global_bytes, ckpt_dir_used, TAG)
    print(f"[client{CLIENT_ID}] wrote global checkpoint: {ckpt_path}")

    # Run local training
    run_local_training(temp_cfg_path)

    # Find updated checkpoint
    updated_path = find_latest_updated_ckpt(ckpt_dir_used, TAG)
    if not updated_path:
        raise FileNotFoundError(f"No updated checkpoint found under {os.path.join(ckpt_dir_used, TAG)}")

    updated_bytes = read_file_bytes(updated_path)
    updated_b64 = bytes_to_b64(updated_bytes)

    # Upload to server
    sio.emit("client_update", {"round": round_idx, "ckpt_b64": updated_b64, "num_samples": NUM_SAMPLES})
    print(f"[client{CLIENT_ID}] uploaded update: {updated_path} (round={round_idx})")


# =========================
# SocketIO handlers
# =========================
@sio.event
def connect():
    print(f"[client{CLIENT_ID}] connected -> {SERVER_URL}")
    sio.emit("register", {"client_id": CLIENT_ID, "room_type": ROOM_TYPE, "num_samples": NUM_SAMPLES})


@sio.on("round_start")
def on_round_start(data: Dict[str, Any]):
    """
    data: {round: int, global_ckpt_b64: str, selected_client_ids: [..]}
    """
    round_idx = int(data["round"])
    global_b64 = str(data.get("global_ckpt_b64", ""))

    # Optional: if server sends selection list, non-selected clients can idle.
    selected_ids = data.get("selected_client_ids")
    if isinstance(selected_ids, list) and selected_ids and (CLIENT_ID not in selected_ids):
        print(f"[client{CLIENT_ID}] not selected for round {round_idx}, skip")
        return

    print(f"[client{CLIENT_ID}] round_start={round_idx}, room={ROOM_TYPE}, tag={TAG}")
    try:
        do_one_round(round_idx, global_b64)
    except Exception as e:
        print(f"[client{CLIENT_ID}] round {round_idx} failed: {e}")
        # Optional: notify server
        sio.emit("client_update", {"round": round_idx, "ckpt_b64": "", "num_samples": NUM_SAMPLES})


@sio.on("round_end")
def on_round_end(data: Dict[str, Any]):
    print(f"[client{CLIENT_ID}] round_end={data.get('round')}")


@sio.on("server_status")
def on_server_status(data: Dict[str, Any]):
    # keep it quiet; print only important server messages
    msg = data.get("msg", "")
    if msg in {"round_started", "round_completed", "waiting_clients"}:
        print(f"[server_status] {data}")


@sio.event
def disconnect():
    print(f"[client{CLIENT_ID}] disconnected")


# =========================
# Main
# =========================
if __name__ == "__main__":
    # simple reconnect loop
    while True:
        try:
            sio.connect(SERVER_URL, transports=["websocket"])
            sio.wait()
        except Exception as e:
            print(f"[client{CLIENT_ID}] connect error: {e}, retry in 3s")
            time.sleep(3)
