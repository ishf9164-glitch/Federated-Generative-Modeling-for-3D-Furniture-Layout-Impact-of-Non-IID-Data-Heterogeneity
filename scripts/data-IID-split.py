# iid_split_total_data_stratified.py
# 目标：S0 IID baseline（更严格）
# - 每个客户端样本数尽量相近
# - 每个客户端的 room_type 分布尽量接近全局（分层/stratified）
#
# 目录假设：total_data_dir 下每个房间一个子文件夹，包含 boxes.npz
# 房型获取：尝试从 room 文件夹内的 meta/info/room_type 等文件中解析；解析不到 -> Other

import os
import csv
import json
import shutil
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None


# -------------------------
# Utils
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_template_cfg(path):
    if yaml is None:
        raise SystemExit("Missing dependency: pyyaml (pip install pyyaml)")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    return cfg

def write_yaml(path, obj):
    if yaml is None:
        raise SystemExit("Missing dependency: pyyaml (pip install pyyaml)")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def make_random_splits(room_ids, seed=0, train_ratio=0.8, val_ratio=0.1):
    rng = np.random.default_rng(seed)
    ids = np.array(room_ids, dtype=object)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)
    split = {}
    for rid in ids[:n_train]: split[str(rid)] = "train"
    for rid in ids[n_train:n_train + n_val]: split[str(rid)] = "val"
    for rid in ids[n_train + n_val:]: split[str(rid)] = "test"
    return split

def rel_to_config(path, config_dir):
    rp = os.path.relpath(path, start=config_dir).replace("\\", "/")
    return rp if not rp.startswith("..") else path


# -------------------------
# Room type extraction (robust, best-effort)
# -------------------------
def _norm(s):
    return str(s).strip().lower().replace(" ", "").replace("_", "")

def _deep_find_room_type(obj):
    keys = ["room_type", "roomType", "type", "category", "label", "name", "kind"]
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None and str(obj[k]).strip() != "":
                return str(obj[k])
        for v in obj.values():
            r = _deep_find_room_type(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _deep_find_room_type(v)
            if r is not None:
                return r
    return None

def _load_json_or_yaml(path):
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".yaml", ".yml"] and yaml is not None:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        if ext in [".json", ".txt"]:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None

def infer_room_type_from_folder(room_dir):
    """
    修改后的逻辑：优先检查文件夹名称，再检查配置文件
    """
    folder_name = os.path.basename(room_dir)
    
    # 1. 定义房型关键词映射（用于从文件夹名中提取）
    # 这里的 key 是文件夹名中可能出现的词，value 是归一化后的标准词
    keywords_map = {
        "bedroom": "bedroom",
        "livingroom": "livingroom",
        "livingdining": "livingroom",
        "diningroom": "diningroom",
        "kitchen": "kitchen",
        "bathroom": "bathroom",
        "study": "library",
        "library": "library",
        "storage": "other"
    }
    
    # 将文件夹名转为纯小写并去特殊字符进行匹配
    name_norm = _norm(folder_name)
    
    for kw, target_type in keywords_map.items():
        if kw in name_norm:
            return target_type

    # 2. 如果文件夹名没有匹配到关键词，尝试读取配置文件（保留原有的深层搜索逻辑作为备选）
    candidates = [
        "room_type.json", "meta.json", "info.json", "room.json"
    ]
    for fn in candidates:
        p = os.path.join(room_dir, fn)
        obj = _load_json_or_yaml(p)
        if obj:
            rt = _deep_find_room_type(obj)
            if rt: return rt
            
    # 3. 最终兜底：直接返回原始文件夹名
    return folder_name

def group_room_type(rt_raw):
    # 论文常用的聚合口径（可按你需要再扩展）
    if rt_raw is None:
        return "Other"
    s = _norm(rt_raw)
    if "masterbedroom" in s or "secondbedroom" in s or "bedroom" in s:
        return "Bedroom"
    if "livingdiningroom" in s:
        return "LivingRoom"   # 单标签口径；你若要双计数请不要在这里做
    if "livingroom" in s:
        return "LivingRoom"
    if "diningroom" in s:
        return "DiningRoom"
    if "library" in s:
        return "Library"
    return "Other"


# -------------------------
# Stratified assignment (key fix)
# -------------------------
def stratified_round_robin(labels, K, seed=0):
    """
    labels: list[str] length N
    返回：list[list[int]]，每个 client 一个 index 列表
    思路：按 label 分组 -> 组内 shuffle -> round-robin 发到各 client
    """
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for i, y in enumerate(labels):
        buckets[y].append(i)

    # label 内打乱
    for y in buckets:
        idx = np.array(buckets[y], dtype=int)
        rng.shuffle(idx)
        buckets[y] = idx.tolist()

    client_indices = [[] for _ in range(K)]
    # round-robin 分配，保证每个 client 的 label 比例接近全局
    for y, idxs in buckets.items():
        for t, i in enumerate(idxs):
            client_indices[t % K].append(i)

    # 再把每个 client 内整体打乱（避免 label block）
    for k in range(K):
        arr = np.array(client_indices[k], dtype=int)
        rng.shuffle(arr)
        client_indices[k] = arr.tolist()

    return client_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_data_dir", default=r"E:\diverse_synth\dump\total_data")
    ap.add_argument("--boxes_name", default="boxes.npz")
    ap.add_argument("--num_clients", type=int, default=3)

    ap.add_argument("--out_root", default=r"E:\diverse_synth\dump")
    ap.add_argument("--config_dir", default=r"E:\diverse_synth\config")
    ap.add_argument("--template_config", default=r"E:\diverse_synth\config\bedroom_config.yaml")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--move", action="store_true", help="默认复制；加 --move 改为移动")
    ap.add_argument("--client_prefix", default="IID_client")
    args = ap.parse_args()

    K = args.num_clients
    if K < 2:
        raise SystemExit("--num_clients must be >= 2")

    ensure_dir(args.config_dir)
    ensure_dir(args.out_root)

    # 1) scan valid room folders
    print(f"Step 1: Scanning directory {args.total_data_dir} ...")
    all_entries = os.listdir(args.total_data_dir)

    room_dirs, room_ids, room_labels = [], [], []
    for name in tqdm(all_entries, desc="Scanning"):
        p = os.path.join(args.total_data_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, args.boxes_name)):
            rt_raw = infer_room_type_from_folder(p)
            lab = group_room_type(rt_raw)
            room_dirs.append(p)
            room_ids.append(name)
            room_labels.append(lab)

    if not room_ids:
        raise SystemExit("No valid room folders found (missing boxes.npz?).")

    N = len(room_ids)
    print(f"Total rooms: {N}")

    # 2) stratified IID assignment (fix)
    print("Step 2: Stratified IID assignment (round-robin by room_type) ...")
    client_chunks = stratified_round_robin(room_labels, K, seed=args.seed)

    client_to_ids = {}
    client_to_dirs = {}
    for k in range(K):
        client_name = f"{args.client_prefix}{k+1}"
        idxs = client_chunks[k]
        client_to_ids[client_name] = [room_ids[i] for i in idxs]
        client_to_dirs[client_name] = [room_dirs[i] for i in idxs]

    sizes = [len(v) for v in client_to_ids.values()]
    print(f"Clients: {K}, per-client sizes: min={min(sizes)}, max={max(sizes)}, std={np.std(sizes):.3f}")

    # 3) train/val/test split per client
    splits_by_client = {}
    for client_name, ids_k in client_to_ids.items():
        splits_by_client[client_name] = make_random_splits(
            ids_k, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
        )

    # 4) move/copy folders
    op_name = "Moving" if args.move else "Copying"
    print(f"Step 4: {op_name} room folders to IID client directories ...")
    mover = shutil.move if args.move else shutil.copytree

    for client_name in client_to_ids.keys():
        ensure_dir(os.path.join(args.out_root, client_name))

    for client_name in tqdm(list(client_to_ids.keys()), desc=op_name):
        dst_client_dir = os.path.join(args.out_root, client_name)
        for rid, src_dir in zip(client_to_ids[client_name], client_to_dirs[client_name]):
            dst = os.path.join(dst_client_dir, rid)
            if os.path.exists(dst):
                continue
            mover(src_dir, dst)

    # 5) write configs + splits csv
    print("Step 5: Generating config files and split CSVs ...")
    template = load_template_cfg(args.template_config)

    for client_name in client_to_ids.keys():
        splits_csv = os.path.join(args.config_dir, f"{client_name}_threed_front_splits.csv")
        with open(splits_csv, "w", newline="") as f:
            w = csv.writer(f)
            for rid, sp in splits_by_client[client_name].items():
                w.writerow([rid, sp])

        cfg = {k: template[k] for k in template}
        cfg["data"]["dataset_directory"] = rel_to_config(os.path.join(args.out_root, client_name), args.config_dir)
        cfg["data"]["annotation_file"] = rel_to_config(splits_csv, args.config_dir)
        cfg["data"]["room_type"] = "total"
        cfg["data"]["room_type_filter"] = "no_filtering"
        cfg["training"]["tag"] = f"{client_name}_total"
        write_yaml(os.path.join(args.config_dir, f"{client_name}_config.yaml"), cfg)

    print("\n" + "=" * 40)
    for client_name in client_to_ids.keys():
        print(f"{client_name:12s}: n={len(client_to_ids[client_name])}")
    print("=" * 40)
    print(f"Done! IID client dirs in: {args.out_root}")
    print(f"Configs/splits in: {args.config_dir}")


if __name__ == "__main__":
    main()
