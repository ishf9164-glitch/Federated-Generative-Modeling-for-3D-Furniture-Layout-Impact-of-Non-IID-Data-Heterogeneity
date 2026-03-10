# server.py
# Federated Learning Coordinator (Server) using Flask-SocketIO + WebSocket
# - Controls rounds
# - Broadcasts global checkpoint to clients
# - Receives client updates (checkpoint.tar bytes)
# - Aggregates with FedAvg over (vae_state_dict, unet_state_dict)

import os
import time
import base64
import random
import threading
import io
from typing import Dict, Any, Tuple, Optional, List

import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO

import torch

# =========================
# Config (env)
# =========================
HOST = os.getenv("FL_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("FL_SERVER_PORT", "5000"))

ROUNDS = int(os.getenv("FL_ROUNDS", "10"))
CLIENTS_TOTAL = int(os.getenv("FL_CLIENTS_TOTAL", "2"))
CLIENTS_PER_ROUND = int(os.getenv("FL_CLIENTS_PER_ROUND", str(CLIENTS_TOTAL)))
SELECTION = os.getenv("FL_CLIENT_SELECTION", "all").lower()  # all | random

ROUND_TIMEOUT_SEC = int(os.getenv("FL_ROUND_TIMEOUT_SEC", "3600"))
WEIGHT_BY = os.getenv("FL_WEIGHT_BY", "num_samples").lower()  # num_samples | uniform

# =========================
# Flask-SocketIO
# =========================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# =========================
# State
# =========================
lock = threading.Lock()

# sid -> info
clients: Dict[str, Dict[str, Any]] = {}
# round -> sid -> update
round_updates: Dict[int, Dict[str, Dict[str, Any]]] = {}

# global checkpoint (as python dict) + b64 payload for broadcast
global_state: Dict[str, Any] = {
    "round": 0,
    "epoch": 0,
    "vae_state_dict": {},   # torch tensors
    "unet_state_dict": {},  # torch tensors
}
global_ckpt_b64: str = ""


# =========================
# Helpers
# =========================
def _torch_save_to_b64(obj: Any) -> str:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _torch_load_from_b64(b64: str) -> Any:
    raw = base64.b64decode(b64.encode("utf-8"))
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu")


def _fedavg_state_dicts(
    sds: List[Tuple[Dict[str, torch.Tensor], float]]
) -> Dict[str, torch.Tensor]:
    """
    sds: [(state_dict, weight), ...]
    returns weighted average state_dict
    """
    total_w = sum(w for _, w in sds)
    if total_w <= 0:
        raise ValueError("Total weight must be > 0")

    keys = list(sds[0][0].keys())
    out: Dict[str, torch.Tensor] = {}

    for k in keys:
        # only average tensors
        if not torch.is_tensor(sds[0][0][k]):
            continue
        acc = None
        for sd, w in sds:
            # tolerate missing keys (skip)
            if k not in sd or not torch.is_tensor(sd[k]):
                continue
            x = sd[k].float() * (w / total_w)
            acc = x if acc is None else acc + x
        if acc is not None:
            out[k] = acc
    return out


def _select_clients() -> List[str]:
    with lock:
        sids = list(clients.keys())
    if SELECTION == "all" or len(sids) <= CLIENTS_PER_ROUND:
        return sids
    return random.sample(sids, CLIENTS_PER_ROUND)


def _wait_updates(round_idx: int, selected_sids: List[str]) -> Dict[str, Dict[str, Any]]:
    deadline = time.time() + ROUND_TIMEOUT_SEC
    while time.time() < deadline:
        with lock:
            got = round_updates.get(round_idx, {})
            if all(sid in got for sid in selected_sids):
                return got
        socketio.sleep(0.2)
    with lock:
        return round_updates.get(round_idx, {})


def _ensure_global_b64():
    global global_ckpt_b64
    global_ckpt_b64 = _torch_save_to_b64(global_state)


def _init_global():
    # round 0 global model (empty dicts acceptable as placeholder)
    global global_state
    global_state = {
        "round": 0,
        "epoch": 0,
        "vae_state_dict": {},
        "unet_state_dict": {},
    }
    _ensure_global_b64()


# =========================
# Training loop
# =========================
def coordinator_loop():
    _init_global()
    socketio.emit("server_status", {"msg": "server_ready", "round": 0})

    # wait for clients
    while True:
        with lock:
            n = len(clients)
        if n >= CLIENTS_TOTAL:
            break
        socketio.emit("server_status", {"msg": "waiting_clients", "connected": n, "needed": CLIENTS_TOTAL})
        socketio.sleep(1.0)

    for r in range(1, ROUNDS + 1):
        selected = _select_clients()
        if not selected:
            socketio.emit("server_status", {"msg": "no_clients_connected", "round": r})
            socketio.sleep(1.0)
            continue

        # broadcast round start (global ckpt as b64 bytes)
        payload = {
            "round": r,
            "global_ckpt_b64": global_ckpt_b64,
            "selected_client_ids": [clients[sid]["client_id"] for sid in selected],
        }
        socketio.emit("round_start", payload)
        socketio.emit("server_status", {"msg": "round_started", "round": r, "selected": len(selected)})

        got = _wait_updates(r, selected)
        received = [(sid, got[sid]) for sid in selected if sid in got]

        socketio.emit("server_status", {"msg": "round_updates_received", "round": r, "received": len(received)})

        if not received:
            socketio.emit("server_status", {"msg": "round_failed_no_updates", "round": r})
            continue

        # load client checkpoints and aggregate
        vae_list: List[Tuple[Dict[str, torch.Tensor], float]] = []
        unet_list: List[Tuple[Dict[str, torch.Tensor], float]] = []

        for sid, upd in received:
            try:
                ckpt_obj = _torch_load_from_b64(upd["ckpt_b64"])
                # expected keys: vae_state_dict, unet_state_dict
                vae_sd = ckpt_obj.get("vae_state_dict", {})
                unet_sd = ckpt_obj.get("unet_state_dict", {})
                ns = float(upd.get("num_samples", 1.0))
                w = ns if WEIGHT_BY == "num_samples" else 1.0

                # only accept dicts
                if isinstance(vae_sd, dict) and isinstance(unet_sd, dict):
                    vae_list.append((vae_sd, w))
                    unet_list.append((unet_sd, w))
                else:
                    socketio.emit("server_status", {"msg": "skip_bad_update_format", "round": r, "sid": sid})
            except Exception as e:
                socketio.emit("server_status", {"msg": "skip_bad_update_load", "round": r, "sid": sid, "error": str(e)})

        if not vae_list or not unet_list:
            socketio.emit("server_status", {"msg": "round_failed_all_bad_updates", "round": r})
            continue

        new_vae = _fedavg_state_dicts(vae_list)
        new_unet = _fedavg_state_dicts(unet_list)

        global_state["round"] = r
        global_state["vae_state_dict"] = new_vae
        global_state["unet_state_dict"] = new_unet
        _ensure_global_b64()

        socketio.emit("round_end", {"round": r, "global_ckpt_b64": global_ckpt_b64})
        socketio.emit("server_status", {"msg": "round_completed", "round": r})

    socketio.emit("server_status", {"msg": "training_finished", "rounds": ROUNDS})


# =========================
# Socket events
# =========================
@socketio.on("connect")
def on_connect():
    socketio.emit("server_status", {"msg": "client_connected"})


@socketio.on("register")
def on_register(data: Dict[str, Any]):
    """
    data: {client_id: int, room_type: str, num_samples: int}
    """
    from flask import request
    sid = request.sid

    info = {
        "client_id": int(data.get("client_id", -1)),
        "room_type": str(data.get("room_type", "unknown")),
        "num_samples": int(data.get("num_samples", 1)),
        "ts": time.time(),
    }
    with lock:
        clients[sid] = info

    socketio.emit("server_status", {"msg": "client_registered", "sid": sid, **info}, broadcast=True)


@socketio.on("client_update")
def on_client_update(data: Dict[str, Any]):
    """
    data: {round: int, ckpt_b64: str, num_samples: int}
    ckpt_b64 should encode a torch-saved dict that includes:
      - vae_state_dict
      - unet_state_dict
    """
    from flask import request
    sid = request.sid

    r = int(data["round"])
    ckpt_b64 = str(data["ckpt_b64"])
    num_samples = int(data.get("num_samples", 1))

    with lock:
        round_updates.setdefault(r, {})[sid] = {
            "ckpt_b64": ckpt_b64,
            "num_samples": num_samples,
            "ts": time.time(),
        }

    socketio.emit("server_status", {"msg": "update_received", "round": r, "sid": sid})


@socketio.on("disconnect")
def on_disconnect():
    from flask import request
    sid = request.sid
    with lock:
        clients.pop(sid, None)
    socketio.emit("server_status", {"msg": "client_disconnected", "sid": sid}, broadcast=True)


# =========================
# Main
# =========================
if __name__ == "__main__":
    # start coordinator loop in background
    socketio.start_background_task(coordinator_loop)
    socketio.run(app, host=HOST, port=PORT)
