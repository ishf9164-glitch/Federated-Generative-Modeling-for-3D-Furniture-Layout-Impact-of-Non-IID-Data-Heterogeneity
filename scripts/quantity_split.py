# quantity_skew_split_and_stats.py
# S3: Quantity-skew（数据量不均 / 长尾）
# - 3 个客户端：1 个大客户端 + 2 个小客户端（默认比例 0.8/0.1/0.1，可改）
# - 输出房间目录到：
#   ../dump/quantity_client1/3D-FRONT
#   ../dump/quantity_client2/3D-FRONT
#   ../dump/quantity_client3/3D-FRONT
# - 并在：
#   ../dump/quantity_client1/dataset_stats.txt
#   ../dump/quantity_client2/dataset_stats.txt
#   ../dump/quantity_client3/dataset_stats.txt
#   生成每个客户端的 dataset_stats.txt（JSON 一行），格式与示例一致
#
# 假设 total_data_dir 下每个房间一个子文件夹，且包含 boxes.npz（或你指定的 boxes_name）

import os
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm


# =========================
# Robust NPZ readers for dataset_stats
# =========================
def _first_existing_key(d, keys):
    for k in keys:
        if k in d:
            return k
    return None

def _safe_as_array(x):
    try:
        return np.asarray(x)
    except Exception:
        return None

def load_boxes_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    # translations (N,3)
    k_t = _first_existing_key(d, ["translations", "translation", "trans", "t", "centers", "center", "positions", "pos"])
    T = _safe_as_array(d[k_t]) if k_t else None

    # sizes (N,3)
    k_s = _first_existing_key(d, ["sizes", "size", "dims", "dimensions", "scales", "scale"])
    S = _safe_as_array(d[k_s]) if k_s else None

    # angles (N,) or (N,1)
    k_a = _first_existing_key(d, ["angles", "angle", "rotations", "rotation", "rots", "rot", "yaws", "yaw"])
    A = _safe_as_array(d[k_a]) if k_a else None

    # labels / class-related
    k_lbl = _first_existing_key(d, ["class_labels", "labels", "classes", "categories", "cats", "names", "jids", "uids"])
    L = _safe_as_array(d[k_lbl]) if k_lbl else None

    return T, S, A, L, d

def _update_minmax(cur_min, cur_max, arr_2d):
    if arr_2d is None:
        return cur_min, cur_max
    arr = np.asarray(arr_2d, dtype=float)
    if arr.size == 0:
        return cur_min, cur_max
    if not np.isfinite(arr).any():
        return cur_min, cur_max
    a_min = np.nanmin(arr, axis=0)
    a_max = np.nanmax(arr, axis=0)
    if cur_min is None:
        return a_min, a_max
    return np.minimum(cur_min, a_min), np.maximum(cur_max, a_max)

def collect_class_labels(npz_obj, L):
    """
    目标：输出 class_labels(list[str])，并保持稳定排序。
    优先级：
      1) npz 里直接提供 class_labels（最符合你的示例）
      2) L 若为字符串数组：unique
      3) L 若为 int：unique -> str
    """
    if "class_labels" in npz_obj:
        try:
            cl = [str(x) for x in list(npz_obj["class_labels"])]
            cl = [x for x in cl if x not in ("", "None", "nan", "NaN")]
            if len(cl) > 0:
                return sorted(set(cl))
        except Exception:
            pass

    if L is None:
        return []

    L = np.asarray(L)
    if L.dtype.kind in ("U", "S", "O"):
        try:
            vals = [str(x) for x in L.tolist()]
            vals = [v for v in vals if v not in ("", "None", "nan", "NaN")]
            if len(vals) > 0:
                return sorted(set(vals))
        except Exception:
            pass

    if np.issubdtype(L.dtype, np.integer):
        uniq = np.unique(L)
        return [str(int(x)) for x in uniq.tolist()]

    return []

def write_dataset_stats_txt(path, stats_obj):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(stats_obj, ensure_ascii=False))


# =========================
# Quantity-skew split
# =========================
def parse_ratios(s):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    r = np.array([float(x) for x in parts], dtype=float)
    if len(r) != 3:
        raise ValueError("Need exactly 3 ratios, e.g. 0.8,0.1,0.1")
    if (r < 0).any():
        raise ValueError("Ratios must be non-negative.")
    sm = r.sum()
    if sm <= 0:
        raise ValueError("Ratios sum must be > 0.")
    return (r / sm).tolist()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def compute_client_counts(n, ratios):
    # 先 floor，再把余数按小数部分从大到小补齐，保证总和=n
    raw = np.array(ratios, dtype=float) * float(n)
    base = np.floor(raw).astype(int)
    remain = n - int(base.sum())
    frac = raw - base
    order = np.argsort(-frac)  # frac 大的优先补 1
    for i in range(remain):
        base[order[i % len(base)]] += 1
    # 防御：若某个为 0 也允许（但你一般会给 >0）
    return base.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_data_dir", default="../dump/total_data")
    ap.add_argument("--boxes_name", default="boxes.npz")

    ap.add_argument("--out_root", default="../dump")
    ap.add_argument("--client_prefix", default="quantity_client")
    ap.add_argument("--subdir", default="3D-FRONT")

    ap.add_argument("--ratios", default="0.8,0.1,0.1", help="3 clients ratios, e.g. 0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move", action="store_true", help="默认复制；加 --move 改为移动")

    ap.add_argument("--furniture_limit", type=int, default=4)
    args = ap.parse_args()

    ratios = parse_ratios(args.ratios)
    rng = np.random.default_rng(args.seed)

    # 1) scan valid room folders
    print(f"Step 1: Scanning directory {args.total_data_dir}...")
    all_entries = os.listdir(args.total_data_dir)
    room_dirs = []
    room_ids = []

    for name in tqdm(all_entries, desc="Scanning"):
        p = os.path.join(args.total_data_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, args.boxes_name)):
            room_dirs.append(p)
            room_ids.append(name)

    if not room_ids:
        raise SystemExit("No valid room folders found (missing boxes.npz?).")

    n = len(room_ids)
    counts = compute_client_counts(n, ratios)

    print(f"Total rooms: {n}")
    print(f"Quantity-skew ratios: {ratios} -> client counts: {counts}")

    # 2) shuffle and assign
    idx = np.arange(n)
    rng.shuffle(idx)

    client_names = [f"{args.client_prefix}{i}" for i in range(1, 4)]
    client_to_ids = {c: [] for c in client_names}
    client_to_dirs = {c: [] for c in client_names}

    start = 0
    for c, take in zip(client_names, counts):
        sub = idx[start:start + take]
        start += take
        client_to_ids[c] = [room_ids[i] for i in sub]
        client_to_dirs[c] = [room_dirs[i] for i in sub]

    # 3) copy/move into ../dump/quantity_client*/3D-FRONT/<room_id>
    op_name = "Moving" if args.move else "Copying"
    mover = shutil.move if args.move else shutil.copytree

    print(f"Step 2: {op_name} room folders into client directories...")
    for c in client_names:
        ensure_dir(os.path.join(args.out_root, c, args.subdir))

    for c in tqdm(client_names, desc=op_name):
        dst_base = os.path.join(args.out_root, c, args.subdir)
        for rid, src in zip(client_to_ids[c], client_to_dirs[c]):
            dst = os.path.join(dst_base, rid)
            if os.path.exists(dst):
                continue
            mover(src, dst)

    # 4) generate dataset_stats.txt per client (at ../dump/quantity_clientX/dataset_stats.txt)
    print("Step 3: Generating dataset_stats.txt for each client...")
    for c in client_names:
        data_dir = os.path.join(args.out_root, c, args.subdir)
        if not os.path.isdir(data_dir):
            print(f"[Skip] missing {data_dir}")
            continue

        room_entries = os.listdir(data_dir)
        room_paths = []
        for name in room_entries:
            p = os.path.join(data_dir, name)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, args.boxes_name)):
                room_paths.append(p)
        room_paths.sort()

        if not room_paths:
            print(f"[Skip] {data_dir}: no rooms with {args.boxes_name}")
            continue

        t_min = t_max = None
        s_min = s_max = None
        a_min = a_max = None
        class_set = set()

        for rd in tqdm(room_paths, desc=f"Stats {c}", leave=False):
            npz_path = os.path.join(rd, args.boxes_name)
            T, S, A, L, npz_obj = load_boxes_npz(npz_path)

            # translations
            if T is not None:
                T = np.asarray(T, dtype=float)
                if T.ndim == 1 and T.size >= 3:
                    T = T.reshape(1, -1)
                if T.ndim == 2 and T.shape[1] >= 3:
                    T = T[:, :3]
                else:
                    T = None
            t_min, t_max = _update_minmax(t_min, t_max, T)

            # sizes
            if S is not None:
                S = np.asarray(S, dtype=float)
                if S.ndim == 1 and S.size >= 3:
                    S = S.reshape(1, -1)
                if S.ndim == 2 and S.shape[1] >= 3:
                    S = S[:, :3]
                else:
                    S = None
            s_min, s_max = _update_minmax(s_min, s_max, S)

            # angles
            if A is not None:
                A = np.asarray(A, dtype=float)
                if A.ndim == 2 and A.shape[1] >= 1:
                    A = A[:, 0]
                if A.ndim == 1:
                    A = A.reshape(-1, 1)
                else:
                    A = None
            a_min, a_max = _update_minmax(a_min, a_max, A)

            # class labels
            cls = collect_class_labels(npz_obj, L)
            for x in cls:
                class_set.add(str(x))

        # fallbacks to match expected schema
        if t_min is None or t_max is None:
            t_min = np.array([0.0, 0.0, 0.0], dtype=float)
            t_max = np.array([0.0, 0.0, 0.0], dtype=float)
        if s_min is None or s_max is None:
            s_min = np.array([0.0, 0.0, 0.0], dtype=float)
            s_max = np.array([0.0, 0.0, 0.0], dtype=float)
        if a_min is None or a_max is None:
            # 如果角度字段缺失，给默认 [-pi, pi]
            a_min = np.array([-np.pi], dtype=float)
            a_max = np.array([np.pi], dtype=float)

        class_labels = sorted(class_set)
        class_order = {lab: i for i, lab in enumerate(class_labels)}

        stats = {
            "bounds_translations": [
                float(t_min[0]), float(t_min[1]), float(t_min[2]),
                float(t_max[0]), float(t_max[1]), float(t_max[2]),
            ],
            "bounds_sizes": [
                float(s_min[0]), float(s_min[1]), float(s_min[2]),
                float(s_max[0]), float(s_max[1]), float(s_max[2]),
            ],
            "bounds_angles": [float(a_min[0]), float(a_max[0])],
            "class_labels": class_labels,
            "class_order": class_order,
            "furniture_limit": int(args.furniture_limit),
        }

        out_path = os.path.join(args.out_root, c, "dataset_stats.txt")
        write_dataset_stats_txt(out_path, stats)
        print(f"[OK] {c}: rooms={len(room_paths)}, classes={len(class_labels)} -> {out_path}")

    print("\nDone!")
    for c in client_names:
        print(f"- {os.path.join(args.out_root, c, args.subdir)}")
        print(f"- {os.path.join(args.out_root, c, 'dataset_stats.txt')}")


if __name__ == "__main__":
    main()
