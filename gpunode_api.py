# gpunode_api.py (run on AutoDL GPU instance)
import os
import io
import json
import base64
import tempfile
import subprocess
import threading
from typing import Dict, Any

from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# ====== Config (env) ======
HOST = os.getenv("GPU_NODE_HOST", "0.0.0.0")
PORT = int(os.getenv("GPU_NODE_PORT", "8000"))

# 代码根目录（包含 diverse_synth/）
PROJECT_ROOT = os.getenv("GPU_NODE_PROJECT_ROOT", os.getcwd())
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "diverse_synth", "scripts")

# 模板 checkpoint
TEMPLATE_DIR = os.getenv("GPU_NODE_TEMPLATE_DIR", "/srv/templates")

# 互斥锁：保证单 GPU 串行
JOB_LOCK = threading.Lock()


def _b64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))


def _bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def _template_path(room: str) -> str:
    return os.path.join(TEMPLATE_DIR, f"template_{room}.tar")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/train_one_round", methods=["POST"])
def train_one_round():
    """
    Request JSON:
    {
      "client_id": 1,
      "room": "bedroom" | "livingroom" | "diningroom",
      "round": 1,
      "global_ckpt_b64": "...",     # torch.save 的 checkpoint.tar bytes
      "local_epochs": 1,
      "generator_type": "EnhancedBetaTCVAE",
      "discriminator_type": "UNet3P"
    }

    Response JSON:
    {
      "ok": true,
      "updated_ckpt_b64": "...",
      "updated_path": "...",
      "stdout": "...(optional trimmed)..."
    }
    """
    payload: Dict[str, Any] = request.get_json(force=True)

    client_id = int(payload["client_id"])
    room = str(payload["room"])
    round_idx = int(payload["round"])
    global_b64 = str(payload["global_ckpt_b64"])
    local_epochs = int(payload.get("local_epochs", 1))
    gen_type = str(payload.get("generator_type", "EnhancedBetaTCVAE"))
    dis_type = str(payload.get("discriminator_type", "UNet3P"))

    tpl = _template_path(room)
    if not os.path.isfile(tpl):
        return jsonify({
            "ok": False,
            "error": f"Template checkpoint not found: {tpl}. Please create it with fl_init_global_ckpt.py and place it here."
        }), 400

    # 写 global checkpoint 到临时文件（供 fl_run_one_round.py --global_ckpt_path 使用）
    try:
        global_bytes = _b64_to_bytes(global_b64)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid global_ckpt_b64: {e}"}), 400

    with tempfile.TemporaryDirectory(prefix="fl_job_") as tmpd:
        global_path = os.path.join(tmpd, f"global_r{round_idx}.tar")
        with open(global_path, "wb") as f:
            f.write(global_bytes)

        # 调用 fl_run_one_round.py（它会输出 updated checkpoint 路径）
        cmd = [
            "python", "fl_run_one_round.py",
            "--room", room,
            "--client_id", str(client_id),
            "--round", str(round_idx),
            "--global_ckpt_path", global_path,
            "--template_ckpt_path", tpl,
            "--generator_type", gen_type,
            "--discriminator_type", dis_type,
            "--local_epochs", str(local_epochs),
        ]

        # 单 GPU：加锁串行执行
        with JOB_LOCK:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=SCRIPTS_DIR,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={**os.environ, "WANDB_MODE": "disabled", "WANDB_SILENT": "true"},
                )
                out = proc.stdout
            except subprocess.CalledProcessError as e:
                return jsonify({
                    "ok": False,
                    "error": "Training job failed",
                    "stdout": (e.stdout or "")[-4000:],  # 截断，避免太大
                }), 500

        # fl_run_one_round.py 最后一行会打印 “[OK] updated checkpoint: <path>”
        updated_path = None
        for line in out.splitlines()[::-1]:
            if "updated checkpoint:" in line:
                updated_path = line.split("updated checkpoint:")[-1].strip()
                break
        if not updated_path or not os.path.isfile(updated_path):
            return jsonify({"ok": False, "error": "Cannot locate updated checkpoint from fl_run_one_round output", "stdout": out[-4000:]}), 500

        # 读回 tar bytes 返回给 ECS client
        with open(updated_path, "rb") as f:
            updated_bytes = f.read()

        return jsonify({
            "ok": True,
            "updated_ckpt_b64": _bytes_to_b64(updated_bytes),
            "updated_path": updated_path,
            "stdout": out[-2000:],  # 只返回末尾
        })


if __name__ == "__main__":
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    app.run(host=HOST, port=PORT)
