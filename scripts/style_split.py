import os
import csv
import shutil
import argparse
import numpy as np
from tqdm import tqdm  # 导入进度条库

try:
    import yaml
except ImportError:
    raise SystemExit("Missing dependency: pyyaml (pip install pyyaml)")

# ... [保留中间的 triangle_area_xz, floor_plan_area_xz, compute_density_from_boxes 函数不变] ...

def triangle_area_xz(v0, v1, v2):
    x0, z0 = float(v0[0]), float(v0[2])
    x1, z1 = float(v1[0]), float(v1[2])
    x2, z2 = float(v2[0]), float(v2[2])
    return 0.5 * abs((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0))

def floor_plan_area_xz(vertices, faces):
    area = 0.0
    for f in faces:
        i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
        area += triangle_area_xz(vertices[i0], vertices[i1], vertices[i2])
    return float(area)

def compute_density_from_boxes(boxes_path):
    d = np.load(boxes_path, allow_pickle=True)
    if "uids" in d:
        num_furn = int(len(d["uids"]))
    elif "jids" in d:
        num_furn = int(len(d["jids"]))
    else:
        return None

    if "floor_plan_vertices" in d and "floor_plan_faces" in d:
        vertices = d["floor_plan_vertices"]
        faces = d["floor_plan_faces"]
    elif "floor_plan" in d:
        fp = d["floor_plan"]
        vertices, faces = fp[0], fp[1]
    else:
        return None

    area = floor_plan_area_xz(vertices, faces)
    if not np.isfinite(area) or area <= 1e-12 or num_furn <= 0:
        return None
    density = float(num_furn) / float(area)
    if not np.isfinite(density) or density <= 0:
        return None
    return density, area, num_furn

def kmeans_3_logspace(densities, seed=0):
    X = np.log(np.asarray(densities, dtype=np.float64) + 1e-12).reshape(-1, 1)
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=3, random_state=seed, n_init=20)
        raw = km.fit_predict(X)
    except Exception:
        q1, q2 = np.quantile(X[:, 0], [1/3, 2/3])
        raw = np.zeros(len(X), dtype=int)
        raw[X[:, 0] >= q1] = 1
        raw[X[:, 0] >= q2] = 2
    centers = np.array([X[raw == k].mean() if np.any(raw == k) else np.inf for k in range(3)])
    order = np.argsort(centers)
    remap = {int(order[i]): i for i in range(3)}
    style = np.array([remap[int(s)] for s in raw], dtype=int)
    return style, centers[order]

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

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_template_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None: cfg = {}
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    return cfg

def write_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_data_dir", default="../dump/total_data")
    ap.add_argument("--boxes_name", default="boxes.npz")
    ap.add_argument("--out_sparse", default="../dump/sparse")
    ap.add_argument("--out_neutral", default="../dump/neutral")
    ap.add_argument("--out_dense", default="../dump/dense")
    ap.add_argument("--config_dir", default="../config")
    ap.add_argument("--template_config", default="../config/bedroom_config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--move", action="store_true", help="默认复制；加 --move 改为移动")
    args = ap.parse_args()

    style_names = ["sparse", "neutral", "dense"]
    out_dirs = {name: getattr(args, f"out_{name}") for name in style_names}

    ensure_dir(args.config_dir)
    for d in out_dirs.values(): ensure_dir(d)

    # 1) scan room folders
    print(f"Step 1: Scanning directory {args.total_data_dir}...")
    all_entries = os.listdir(args.total_data_dir)
    room_dirs = []
    for name in tqdm(all_entries, desc="Scanning"):
        p = os.path.join(args.total_data_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, args.boxes_name)):
            room_dirs.append(p)
    room_dirs.sort()
    
    if not room_dirs:
        raise SystemExit(f"No room folders found.")

    # 2) compute density
    print(f"Step 2: Computing densities for {len(room_dirs)} rooms...")
    room_ids, densities, areas, nums = [], [], [], []
    valid_room_dirs = []
    for rd in tqdm(room_dirs, desc="Processing NPZ"):
        rid = os.path.basename(rd)
        res = compute_density_from_boxes(os.path.join(rd, args.boxes_name))
        if res is None:
            continue
        den, area, num = res
        room_ids.append(rid)
        densities.append(den)
        areas.append(area)
        nums.append(num)
        valid_room_dirs.append(rd)

    if len(room_ids) < 10:
        raise SystemExit(f"Too few valid rooms: {len(room_ids)}")

    # 3) clustering
    style_id, centers = kmeans_3_logspace(densities, seed=args.seed)

    # 4) splits
    splits_by_style = {}
    for sid, sname in enumerate(style_names):
        ids_in_style = [rid for rid, st in zip(room_ids, style_id) if int(st) == sid]
        splits_by_style[sname] = make_random_splits(ids_in_style, args.seed, args.train_ratio, args.val_ratio)

    # 5) move/copy folders
    op_name = "Moving" if args.move else "Copying"
    print(f"Step 5: {op_name} room folders to style directories...")
    mover = shutil.move if args.move else shutil.copytree
    for rd, rid, st in tqdm(zip(valid_room_dirs, room_ids, style_id), total=len(valid_room_dirs), desc=op_name):
        sname = style_names[int(st)]
        dst = os.path.join(out_dirs[sname], rid)
        if os.path.exists(dst):
            continue # 或者报 SystemExit
        mover(rd, dst)

    # 6) write configs
    print("Step 6: Generating config files and split CSVs...")
    template = load_template_cfg(args.template_config)
    def rel_to_config(p):
        rp = os.path.relpath(p, start=args.config_dir).replace("\\", "/")
        return rp if not rp.startswith("..") else p

    for sname in style_names:
        splits_csv = os.path.join(args.config_dir, f"{sname}_threed_front_splits.csv")
        with open(splits_csv, "w", newline="") as f:
            w = csv.writer(f)
            for rid, sp in splits_by_style[sname].items():
                w.writerow([rid, sp])

        cfg = {k: template[k] for k in template}
        cfg["data"]["dataset_directory"] = rel_to_config(out_dirs[sname])
        cfg["data"]["annotation_file"] = rel_to_config(splits_csv)
        cfg["data"]["room_type"] = "total"
        cfg["data"]["room_type_filter"] = "no_filtering"
        cfg["training"]["tag"] = f"{sname}_total"
        write_yaml(os.path.join(args.config_dir, f"{sname}_config.yaml"), cfg)

    # summary [保持不变]
    print("\n" + "="*30)
    dens = np.asarray(densities, dtype=np.float64)
    for sid, sname in enumerate(style_names):
        idx = np.where(style_id == sid)[0]
        if len(idx) > 0:
            print(f"{sname:7}: n={len(idx):4}, mean={dens[idx].mean():.4f}, p50={np.median(dens[idx]):.4f}")
    print("="*30)
    print(f"Done! Configs are in: {args.config_dir}")

if __name__ == "__main__":
    main()