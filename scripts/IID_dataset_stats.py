# make_dataset_stats_iid.py
# 作用：
# 对已划分好的 IID 客户端数据目录（例如 ../dump/IID_client1/3D-FRONT/ 下的房间文件夹）
# 逐个生成 dataset_stats.txt（JSON 一行），格式与你给的示例一致：
# {
#   "bounds_translations": [...6...],
#   "bounds_sizes": [...6...],
#   "bounds_angles": [...2...],
#   "class_labels": [...],
#   "class_order": {...},
#   "furniture_limit": 4
# }
#
# 说明：
# - 自动从每个 room 文件夹里的 boxes.npz 读取 translations / sizes / angles / labels（字段名多候选，兼容不同版本）
# - class_labels：优先读取 npz 里自带的类名列表；否则从每个物体的类别字符串集合构建；再否则退化为整数标签字符串
# - furniture_limit：默认写 4（可用参数修改）
#
# 用法示例：
# python make_dataset_stats_iid.py --iid_root ../dump --clients IID_client1 IID_client2 IID_client3 --subdir 3D-FRONT --boxes_name boxes.npz --furniture_limit 4
#
# 或者自动扫描：
# python make_dataset_stats_iid.py --iid_root ../dump --client_prefix IID_client --subdir 3D-FRONT

import os
import json
import argparse
import numpy as np
from tqdm import tqdm


# -------------------------
# Robust NPZ field readers
# -------------------------
def _first_existing_key(d, keys):
    for k in keys:
        if k in d:
            return k
    return None

def _as_array(x):
    try:
        return np.asarray(x)
    except Exception:
        return None

def load_boxes_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())

    # translations (N,3)
    k_t = _first_existing_key(d, [
        "translations", "translation", "trans", "t", "centers", "center", "positions", "pos"
    ])
    T = _as_array(d[k_t]) if k_t else None

    # sizes (N,3)
    k_s = _first_existing_key(d, [
        "sizes", "size", "dims", "dimensions", "scales", "scale"
    ])
    S = _as_array(d[k_s]) if k_s else None

    # angles (N,) or (N,1)
    k_a = _first_existing_key(d, [
        "angles", "angle", "rotations", "rotation", "rots", "rot", "yaws", "yaw"
    ])
    A = _as_array(d[k_a]) if k_a else None

    # labels: could be strings or ints
    k_lbl = _first_existing_key(d, [
        "class_labels", "labels", "classes", "categories", "cats", "names", "jids", "uids"
    ])
    L = _as_array(d[k_lbl]) if k_lbl else None

    return keys, T, S, A, L, d


def update_bounds(minv, maxv, arr, expect_dim):
    """arr shape (N, expect_dim)"""
    if arr is None:
        return minv, maxv
    arr = np.asarray(arr)
    if arr.ndim == 1 and expect_dim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2 and arr.shape[1] != expect_dim:
        return minv, maxv
    if arr.ndim != 2:
        return minv, maxv
    if arr.shape[0] == 0:
        return minv, maxv
    if not np.isfinite(arr).any():
        return minv, maxv

    a_min = np.nanmin(arr, axis=0)
    a_max = np.nanmax(arr, axis=0)
    if minv is None:
        minv = a_min
        maxv = a_max
    else:
        minv = np.minimum(minv, a_min)
        maxv = np.maximum(maxv, a_max)
    return minv, maxv


def collect_class_names(L, npz_obj):
    """
    返回 class_labels(list[str])。
    优先级：
    1) npz 里直接有 "class_labels" 且是 list[str]
    2) L 若是字符串数组（每个物体类别），则取 unique
    3) L 若是 int 标签且 npz 里有 "id_to_class"/"classes"/"class_names" 之类映射则用
    4) 退化：把 int 标签转成字符串
    """
    # 1) direct class_labels
    if "class_labels" in npz_obj:
        cl = npz_obj["class_labels"]
        try:
            cl_list = [str(x) for x in list(cl)]
            if len(cl_list) > 0:
                return sorted(set(cl_list))
        except Exception:
            pass

    if L is None:
        return []

    L = np.asarray(L)

    # 2) string labels per object
    if L.dtype.kind in ("U", "S", "O"):
        try:
            # 可能是每个物体一个类别字符串
            vals = [str(x) for x in L.tolist()]
            # 过滤明显的 uid/jid（如果你想保留也可以删掉这段过滤）
            vals = [v for v in vals if v not in ("None", "nan", "NaN", "")]
            if len(vals) > 0:
                return sorted(set(vals))
        except Exception:
            pass

    # 3) int labels + mapping
    if np.issubdtype(L.dtype, np.integer):
        for map_key in ["id_to_class", "id2class", "class_names", "classes", "categories"]:
            if map_key in npz_obj:
                try:
                    mapping = npz_obj[map_key]
                    # mapping 可能是 dict 或 list
                    if isinstance(mapping, dict):
                        return [str(mapping[i]) for i in sorted(mapping.keys())]
                    else:
                        return [str(x) for x in list(mapping)]
                except Exception:
                    pass
        # 4) fallback: int -> str
        uniq = np.unique(L)
        return [str(int(x)) for x in uniq.tolist()]

    return []


def write_dataset_stats(out_path, stats_obj):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(stats_obj, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iid_root", default="../dump", help="包含 IID_client* 的父目录，例如 ../dump")
    ap.add_argument("--clients", nargs="*", default=None, help="显式指定客户端目录名列表，例如 IID_client1 IID_client2")
    ap.add_argument("--client_prefix", default="IID_client", help="自动扫描时使用的前缀")
    ap.add_argument("--subdir", default="3D-FRONT", help="客户端下的数据子目录名（你的例子是 3D-FRONT）")
    ap.add_argument("--boxes_name", default="boxes.npz")
    ap.add_argument("--furniture_limit", type=int, default=4)
    args = ap.parse_args()

    # resolve client dirs
    if args.clients:
        client_dirs = [os.path.join(args.iid_root, c) for c in args.clients]
    else:
        all_entries = os.listdir(args.iid_root)
        client_dirs = [
            os.path.join(args.iid_root, x)
            for x in all_entries
            if x.startswith(args.client_prefix) and os.path.isdir(os.path.join(args.iid_root, x))
        ]
        client_dirs.sort()

    if not client_dirs:
        raise SystemExit("No client directories found. Use --clients or check --iid_root/--client_prefix")

    for cdir in client_dirs:
        data_dir = os.path.join(cdir, args.subdir)
        if not os.path.isdir(data_dir):
            print(f"[Skip] {cdir}: missing subdir {args.subdir}")
            continue

        # room folders: any directory containing boxes.npz
        room_entries = os.listdir(data_dir)
        room_dirs = []
        for name in room_entries:
            p = os.path.join(data_dir, name)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, args.boxes_name)):
                room_dirs.append(p)
        room_dirs.sort()

        if not room_dirs:
            print(f"[Skip] {data_dir}: no room folders with {args.boxes_name}")
            continue

        # collect bounds + class labels
        t_min = t_max = None
        s_min = s_max = None
        a_min = a_max = None
        all_class_names = set()

        for rd in tqdm(room_dirs, desc=f"Scanning {os.path.basename(cdir)}", leave=False):
            npz_path = os.path.join(rd, args.boxes_name)
            keys, T, S, A, L, npz_obj = load_boxes_npz(npz_path)

            # translations
            if T is not None:
                T = np.asarray(T, dtype=float)
                if T.ndim == 1 and T.size == 3:
                    T = T.reshape(1, 3)
                if T.ndim == 2 and T.shape[1] >= 3:
                    T = T[:, :3]
                else:
                    T = None
            t_min, t_max = update_bounds(t_min, t_max, T, 3)

            # sizes
            if S is not None:
                S = np.asarray(S, dtype=float)
                if S.ndim == 1 and S.size == 3:
                    S = S.reshape(1, 3)
                if S.ndim == 2 and S.shape[1] >= 3:
                    S = S[:, :3]
                else:
                    S = None
            s_min, s_max = update_bounds(s_min, s_max, S, 3)

            # angles
            if A is not None:
                A = np.asarray(A, dtype=float)
                if A.ndim == 2 and A.shape[1] >= 1:
                    A = A[:, 0]
                if A.ndim == 1:
                    A = A.reshape(-1, 1)
                else:
                    A = None
            a_min, a_max = update_bounds(a_min, a_max, A, 1)

            # class names
            cls = collect_class_names(L, npz_obj)
            for x in cls:
                all_class_names.add(str(x))

        # finalize bounds
        if t_min is None or t_max is None:
            # fallback to zeros
            t_min = np.array([0.0, 0.0, 0.0], dtype=float)
            t_max = np.array([0.0, 0.0, 0.0], dtype=float)
        if s_min is None or s_max is None:
            s_min = np.array([0.0, 0.0, 0.0], dtype=float)
            s_max = np.array([0.0, 0.0, 0.0], dtype=float)
        if a_min is None or a_max is None:
            # default yaw range
            a_min = np.array([-np.pi], dtype=float)
            a_max = np.array([np.pi], dtype=float)

        class_labels = sorted(all_class_names)
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

        out_path = os.path.join(data_dir, "dataset_stats.txt")
        write_dataset_stats(out_path, stats)
        print(f"[OK] Wrote {out_path}  (rooms={len(room_dirs)}, classes={len(class_labels)})")


if __name__ == "__main__":
    main()
