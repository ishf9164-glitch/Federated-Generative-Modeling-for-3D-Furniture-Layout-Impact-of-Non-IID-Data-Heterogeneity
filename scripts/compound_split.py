# compound_skew_split_and_stats.py
# S4: Compound（Label + Style + Quantity）
# - 3 个客户端：数据量不均（长尾 ratios），同时在房型(label)与风格(style=density聚类)上有偏置
# - 输出：
#   ../dump/compound_client1/3D-FRONT/<room_id>
#   ../dump/compound_client2/3D-FRONT/<room_id>
#   ../dump/compound_client3/3D-FRONT/<room_id>
# - 并生成：
#   ../dump/compound_client1/dataset_stats.txt
#   ../dump/compound_client2/dataset_stats.txt
#   ../dump/compound_client3/dataset_stats.txt
#
# 假设：total_data_dir 下每个房间一个子文件夹，且包含 boxes.npz（默认）
# 房型(room_type)获取：脚本会在 room 文件夹内尝试从若干常见 meta 文件读取；读不到则归为 Other

import os
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm

try:
    import yaml
except Exception:
    yaml = None


# =========================
# Geometry: density from boxes.npz (与你原脚本一致)
# =========================
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
    order = np.argsort(centers)             # 低密度 -> 高密度
    remap = {int(order[i]): i for i in range(3)}
    style = np.array([remap[int(s)] for s in raw], dtype=int)
    # style: 0=sparse, 1=neutral, 2=dense
    return style, centers[order]


# =========================
# Room type extraction (尽量鲁棒)
# =========================
def norm(s):
    return str(s).strip().lower().replace(" ", "").replace("_", "")

def deep_find_room_type(obj):
    """在 dict/list 中递归找 room type 字段。"""
    keys = ["room_type", "roomType", "type", "category", "label", "name", "kind"]
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None and str(obj[k]).strip() != "":
                return str(obj[k])
        for v in obj.values():
            r = deep_find_room_type(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = deep_find_room_type(v)
            if r is not None:
                return r
    return None

def load_json_or_yaml(path):
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".yaml", ".yml"] and yaml is not None:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        if ext in [".json", ".jsn", ".txt"]:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None

def infer_room_type_from_folder(room_dir):
    """
    尝试从 room_dir 里的常见文件推断房型。
    找不到返回 None。
    """
    candidates = [
        "room_type.json", "room_type.yaml", "room_type.yml", "room_type.txt",
        "meta.json", "meta.yaml", "meta.yml",
        "info.json", "info.yaml", "info.yml",
        "scene.json", "scene.yaml", "scene.yml",
        "room.json", "annotation.json"
    ]
    for fn in candidates:
        p = os.path.join(room_dir, fn)
        obj = load_json_or_yaml(p)
        if obj is None:
            continue
        rt = deep_find_room_type(obj)
        if rt is not None:
            return rt
    return None

def group_room_type(rt_raw):
    """
    把细粒度房型归并到用于 label-skew 的大类：
    Bedroom / LivingRoom / DiningRoom / Library / Other
    规则与你之前需求一致：LivingDiningRoom 计入 LivingRoom 和 DiningRoom 在“语义上”常见，
    这里为了单标签划分，默认优先归入 LivingRoom（可按需改）。
    """
    if rt_raw is None:
        return "Other"
    s = norm(rt_raw)

    # bedroom family
    if "masterbedroom" in s or "secondbedroom" in s or s == "bedroom":
        return "Bedroom"
    if "bedroom" in s:
        return "Bedroom"

    # living/dining mix
    if "livingdiningroom" in s:
        # 单标签：默认归 LivingRoom（若你想归 DiningRoom 改这里）
        return "LivingRoom"

    if "livingroom" in s:
        return "LivingRoom"
    if "diningroom" in s:
        return "DiningRoom"
    if "library" in s:
        return "Library"

    return "Other"


# =========================
# dataset_stats.txt (与你前面版本一致思路，简化成必需字段)
# =========================
def _first_existing_key(d, keys):
    for k in keys:
        if k in d:
            return k
    return None

def load_boxes_npz_for_stats(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    k_t = _first_existing_key(d, ["translations", "translation", "trans", "t", "centers", "positions", "pos"])
    T = np.asarray(d[k_t]) if k_t else None

    k_s = _first_existing_key(d, ["sizes", "size", "dims", "dimensions", "scales", "scale"])
    S = np.asarray(d[k_s]) if k_s else None

    k_a = _first_existing_key(d, ["angles", "angle", "rotations", "rotation", "rots", "rot", "yaws", "yaw"])
    A = np.asarray(d[k_a]) if k_a else None

    # 这里优先用 class_labels（示例一致）
    L = np.asarray(d["class_labels"]) if "class_labels" in d else None

    return T, S, A, L, d

def _update_minmax(cur_min, cur_max, arr):
    if arr is None:
        return cur_min, cur_max
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or (not np.isfinite(arr).any()):
        return cur_min, cur_max
    a_min = np.nanmin(arr, axis=0)
    a_max = np.nanmax(arr, axis=0)
    if cur_min is None:
        return a_min, a_max
    return np.minimum(cur_min, a_min), np.maximum(cur_max, a_max)

def write_dataset_stats(client_root, data_dir, boxes_name="boxes.npz", furniture_limit=4):
    # room folders
    room_dirs = []
    for name in os.listdir(data_dir):
        p = os.path.join(data_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, boxes_name)):
            room_dirs.append(p)
    room_dirs.sort()
    if not room_dirs:
        raise RuntimeError(f"No rooms with {boxes_name} under {data_dir}")

    t_min = t_max = None
    s_min = s_max = None
    a_min = a_max = None
    class_set = set()

    for rd in tqdm(room_dirs, desc=f"Stats {os.path.basename(client_root)}", leave=False):
        npz_path = os.path.join(rd, boxes_name)
        T, S, A, L, npz_obj = load_boxes_npz_for_stats(npz_path)

        # translations (N,3)
        if T is not None:
            T = np.asarray(T, dtype=float)
            if T.ndim == 1 and T.size >= 3:
                T = T.reshape(1, -1)
            if T.ndim == 2 and T.shape[1] >= 3:
                T = T[:, :3]
            else:
                T = None
        t_min, t_max = _update_minmax(t_min, t_max, T)

        # sizes (N,3)
        if S is not None:
            S = np.asarray(S, dtype=float)
            if S.ndim == 1 and S.size >= 3:
                S = S.reshape(1, -1)
            if S.ndim == 2 and S.shape[1] >= 3:
                S = S[:, :3]
            else:
                S = None
        s_min, s_max = _update_minmax(s_min, s_max, S)

        # angles (N,1)
        if A is not None:
            A = np.asarray(A, dtype=float)
            if A.ndim == 2 and A.shape[1] >= 1:
                A = A[:, 0]
            if A.ndim == 1:
                A = A.reshape(-1, 1)
            else:
                A = None
        a_min, a_max = _update_minmax(a_min, a_max, A)

        # class labels（示例字段）
        if "class_labels" in npz_obj:
            try:
                for x in list(npz_obj["class_labels"]):
                    s = str(x).strip()
                    if s:
                        class_set.add(s)
            except Exception:
                pass

    # fallbacks
    if t_min is None or t_max is None:
        t_min = np.array([0.0, 0.0, 0.0]); t_max = np.array([0.0, 0.0, 0.0])
    if s_min is None or s_max is None:
        s_min = np.array([0.0, 0.0, 0.0]); s_max = np.array([0.0, 0.0, 0.0])
    if a_min is None or a_max is None:
        a_min = np.array([-np.pi]); a_max = np.array([np.pi])

    class_labels = sorted(class_set)
    class_order = {lab: i for i, lab in enumerate(class_labels)}

    stats = {
        "bounds_translations": [float(t_min[0]), float(t_min[1]), float(t_min[2]),
                               float(t_max[0]), float(t_max[1]), float(t_max[2])],
        "bounds_sizes": [float(s_min[0]), float(s_min[1]), float(s_min[2]),
                         float(s_max[0]), float(s_max[1]), float(s_max[2])],
        "bounds_angles": [float(a_min[0]), float(a_max[0])],
        "class_labels": class_labels,
        "class_order": class_order,
        "furniture_limit": int(furniture_limit),
    }

    out_path = os.path.join(client_root, "dataset_stats.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(stats, ensure_ascii=False))
    return out_path, len(room_dirs), len(class_labels)


# =========================
# Compound assignment
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_ratios(s):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    r = np.array([float(x) for x in parts], dtype=float)
    if len(r) != 3:
        raise ValueError("Need exactly 3 ratios, e.g. 0.7,0.2,0.1")
    if (r < 0).any():
        raise ValueError("Ratios must be non-negative.")
    sm = r.sum()
    if sm <= 0:
        raise ValueError("Ratios sum must be > 0.")
    return (r / sm).tolist()

def compute_client_counts(n, ratios):
    raw = np.array(ratios, dtype=float) * float(n)
    base = np.floor(raw).astype(int)
    remain = n - int(base.sum())
    frac = raw - base
    order = np.argsort(-frac)
    for i in range(remain):
        base[order[i % len(base)]] += 1
    return base.tolist()

def style_name_from_id(sid):
    return ["sparse", "neutral", "dense"][int(sid)]

def pick_from_pool(available_idx, mask_idx, need, rng):
    """从 available_idx ∩ mask_idx 中随机抽 need 个，返回 picked, remaining_available"""
    cand = np.array([i for i in available_idx if mask_idx[i]], dtype=int)
    if len(cand) == 0 or need <= 0:
        return [], available_idx
    rng.shuffle(cand)
    take = cand[:min(need, len(cand))].tolist()
    remain = [i for i in available_idx if i not in set(take)]
    return take, remain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_data_dir", default="../dump/total_data")
    ap.add_argument("--boxes_name", default="boxes.npz")

    ap.add_argument("--out_root", default="../dump")
    ap.add_argument("--client_prefix", default="compound_client")
    ap.add_argument("--subdir", default="3D-FRONT")

    # Quantity-skew
    ap.add_argument("--ratios", default="0.7,0.2,0.1", help="client data ratios, e.g. 0.7,0.2,0.1")

    # Label+Style bias profiles（默认：各客户端偏向不同组合）
    ap.add_argument("--client1_room", default="Bedroom")
    ap.add_argument("--client2_room", default="LivingRoom")
    ap.add_argument("--client3_room", default="DiningRoom")

    ap.add_argument("--client1_style", default="sparse", choices=["sparse", "neutral", "dense"])
    ap.add_argument("--client2_style", default="neutral", choices=["sparse", "neutral", "dense"])
    ap.add_argument("--client3_style", default="dense", choices=["sparse", "neutral", "dense"])

    # 偏置强度：每个客户端至少这么多比例来自“room+style 同时匹配”的子集（不够会自动降级放宽条件）
    ap.add_argument("--bias_strength", type=float, default=0.8)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move", action="store_true")

    ap.add_argument("--furniture_limit", type=int, default=4)
    args = ap.parse_args()

    ratios = parse_ratios(args.ratios)
    rng = np.random.default_rng(args.seed)

    client_names = [f"{args.client_prefix}{i}" for i in range(1, 4)]
    client_pref = {
        client_names[0]: (args.client1_room, args.client1_style),
        client_names[1]: (args.client2_room, args.client2_style),
        client_names[2]: (args.client3_room, args.client3_style),
    }

    # 1) scan rooms
    print(f"Step 1: Scanning {args.total_data_dir}...")
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

    # 2) compute density + room_type
    print(f"Step 2: Computing density + room_type for {len(room_dirs)} rooms...")
    densities = []
    room_type_raw = []
    valid_dirs = []
    valid_ids = []
    for rd, rid in tqdm(list(zip(room_dirs, room_ids)), desc="Processing"):
        res = compute_density_from_boxes(os.path.join(rd, args.boxes_name))
        if res is None:
            continue
        den, _, _ = res
        rt = infer_room_type_from_folder(rd)
        densities.append(den)
        room_type_raw.append(rt)
        valid_dirs.append(rd)
        valid_ids.append(rid)

    if len(valid_ids) < 50:
        raise SystemExit(f"Too few valid rooms after density parsing: {len(valid_ids)}")

    densities = np.asarray(densities, dtype=np.float64)
    room_type_group = np.array([group_room_type(rt) for rt in room_type_raw], dtype=object)

    # 3) style clustering
    print("Step 3: KMeans style clustering on log-density...")
    style_id, centers = kmeans_3_logspace(densities.tolist(), seed=args.seed)
    style_id = np.asarray(style_id, dtype=int)
    style_str = np.array([style_name_from_id(s) for s in style_id], dtype=object)

    # 4) decide client sizes (quantity skew)
    n = len(valid_ids)
    counts = compute_client_counts(n, ratios)
    print(f"Total valid rooms: {n}")
    print(f"Ratios {ratios} -> counts {counts}")
    print("Client preferences:")
    for c in client_names:
        print(f"  {c}: room={client_pref[c][0]}, style={client_pref[c][1]}")

    # 5) compound assignment (no overlap)
    available = list(range(n))

    # precompute masks
    def mask_room(target_room):
        return np.array([rt == target_room for rt in room_type_group], dtype=bool)

    def mask_style(target_style):
        return np.array([st == target_style for st in style_str], dtype=bool)

    assignments = {c: [] for c in client_names}

    for c, target_n in zip(client_names, counts):
        pref_room, pref_style = client_pref[c]
        need_main = int(round(target_n * float(args.bias_strength)))
        need_main = min(max(need_main, 0), target_n)

        m_room = mask_room(pref_room)
        m_style = mask_style(pref_style)
        m_both = m_room & m_style

        # 5.1 先拿 room+style 都匹配的
        picked, available = pick_from_pool(available, m_both, need_main, rng)
        assignments[c].extend(picked)

        # 5.2 若不够，放宽到 room 匹配
        if len(assignments[c]) < need_main:
            picked2, available = pick_from_pool(available, m_room, need_main - len(assignments[c]), rng)
            assignments[c].extend(picked2)

        # 5.3 若还不够，放宽到 style 匹配
        if len(assignments[c]) < need_main:
            picked3, available = pick_from_pool(available, m_style, need_main - len(assignments[c]), rng)
            assignments[c].extend(picked3)

        # 5.4 余下 quota 从剩余中随机补齐（保持数量不均 + 但仍是 IID 混入）
        remaining_need = target_n - len(assignments[c])
        if remaining_need > 0:
            rng.shuffle(available)
            take = available[:min(remaining_need, len(available))]
            assignments[c].extend(take)
            available = available[min(remaining_need, len(available)):]

    # 如果还有未分配（通常因为 rounding），塞到 client1
    if len(available) > 0:
        assignments[client_names[0]].extend(available)
        available = []

    # 汇总打印
    print("\n" + "=" * 40)
    for c in client_names:
        idxs = np.array(assignments[c], dtype=int)
        print(f"{c}: n={len(idxs)} | "
              f"room_major={client_pref[c][0]} frac={np.mean(room_type_group[idxs]==client_pref[c][0]):.3f} | "
              f"style_major={client_pref[c][1]} frac={np.mean(style_str[idxs]==client_pref[c][1]):.3f}")
    print("=" * 40)

    # 6) copy/move to output dirs
    op_name = "Moving" if args.move else "Copying"
    mover = shutil.move if args.move else shutil.copytree

    print(f"Step 4: {op_name} room folders to compound_client*/3D-FRONT ...")
    for c in client_names:
        ensure_dir(os.path.join(args.out_root, c, args.subdir))

    for c in tqdm(client_names, desc=op_name):
        dst_base = os.path.join(args.out_root, c, args.subdir)
        for i in assignments[c]:
            rid = valid_ids[i]
            src = valid_dirs[i]
            dst = os.path.join(dst_base, rid)
            if os.path.exists(dst):
                continue
            mover(src, dst)

    # 7) dataset_stats.txt per client (at ../dump/compound_clientX/dataset_stats.txt)
    print("Step 5: Generating dataset_stats.txt per client...")
    for c in client_names:
        client_root = os.path.join(args.out_root, c)
        data_dir = os.path.join(client_root, args.subdir)
        out_path, n_rooms, n_cls = write_dataset_stats(
            client_root=client_root,
            data_dir=data_dir,
            boxes_name=args.boxes_name,
            furniture_limit=args.furniture_limit
        )
        print(f"[OK] {c}: rooms={n_rooms}, classes={n_cls} -> {out_path}")

    print("\nDone!")
    print("Outputs:")
    for c in client_names:
        print(f"- {os.path.join(args.out_root, c, args.subdir)}")
        print(f"- {os.path.join(args.out_root, c, 'dataset_stats.txt')}")


if __name__ == "__main__":
    main()
