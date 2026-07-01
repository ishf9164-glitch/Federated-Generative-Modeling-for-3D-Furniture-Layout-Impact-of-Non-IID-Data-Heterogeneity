# style_cluster_room_distribution.py
# 目标：
# 1) 从 3D-FRONT JSON 读取房间类型
# 2) 用“家具排列密集程度”定义风格特征（density）
# 3) 对全数据聚类成 3 类风格：宽敞 / 中性 / 精致
# 4) 画柱状图：每个风格组下各房间类型分布（堆叠柱状图）
#
# 说明：
# - 如果 JSON 中 room 没有独立 bbox/面积信息，则退化为 scene-level 面积（同一 scene 内 rooms 共享 area）
# - 如果没有任何 bbox，则退化为 density = furniture_count（仍可聚类，但风格解释会弱一些）

import os, json
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATA_DIR = r"E:\diverse_synth\dump\3D-FRONT"

N_STYLE = 3
STYLE_NAMES = ["Spacious", "Neutral", "Compact"]  # 最终会按密度从低到高映射为：宽敞/中性/精致
STYLE_NAMES_CN = ["宽敞", "中性", "精致"]

TOP_ROOM_TYPES = 12     # 图上最多显示多少种房型，其余合并为 Other
MIN_SAMPLES_PER_STYLE = 200  # 风格组太小就说明聚类/特征有问题


# -------------------------
# JSON field helpers
# -------------------------
def extract_furniture_items(data: dict):
    furn = data.get("furniture", [])
    return furn if isinstance(furn, list) else []

def get_room_type(room: dict):
    if not isinstance(room, dict):
        return "UnknownRoom"
    for k in ["type", "category", "room_type", "roomType", "name", "label", "kind"]:
        if k in room and room[k] is not None:
            return str(room[k])
    return "UnknownRoom"

def get_rooms(data: dict):
    scene = data.get("scene", {})
    room_obj = scene.get("room", None)
    if room_obj is None:
        return []
    if isinstance(room_obj, dict):
        return [room_obj]
    if isinstance(room_obj, list):
        return room_obj
    return []

def bbox_area_from_obj(obj: dict):
    """从可能的 boundingBox / bbox / box 中推断 2D 面积。"""
    if not isinstance(obj, dict):
        return None

    # 可能字段名
    for k in ["boundingBox", "bbox", "box"]:
        bb = obj.get(k, None)
        if bb is None:
            continue

        # 常见形式1：{"min":[x,y,z], "max":[x,y,z]}
        if isinstance(bb, dict) and "min" in bb and "max" in bb:
            mn, mx = bb["min"], bb["max"]
            if isinstance(mn, (list, tuple)) and isinstance(mx, (list, tuple)) and len(mn) >= 2 and len(mx) >= 2:
                w = float(mx[0]) - float(mn[0])
                h = float(mx[1]) - float(mn[1])
                area = abs(w * h)
                return area if np.isfinite(area) and area > 0 else None

        # 常见形式2：[minx, miny, maxx, maxy] 或 [x, y, w, h]
        if isinstance(bb, (list, tuple)):
            arr = list(bb)
            if len(arr) >= 4:
                a, b, c, d = map(float, arr[:4])
                # 猜测是 minx,miny,maxx,maxy
                area1 = abs((c - a) * (d - b))
                if np.isfinite(area1) and area1 > 0:
                    return area1
                # 或者 x,y,w,h
                area2 = abs(c * d)
                if np.isfinite(area2) and area2 > 0:
                    return area2

    return None


# -------------------------
# 1) Build per-room samples: (density, room_type)
# -------------------------
density_list = []
roomtype_list = []

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
if not files:
    raise RuntimeError(f"No .json files found in {DATA_DIR}")

for fn in tqdm(files, desc="Parsing 3D-FRONT JSON"):
    with open(os.path.join(DATA_DIR, fn), "r", encoding="utf-8") as fp:
        data = json.load(fp)

    rooms = get_rooms(data)
    if not rooms:
        continue

    furn_list = extract_furniture_items(data)
    n_furn = len(furn_list)

    scene = data.get("scene", {})
    scene_area = bbox_area_from_obj(scene)  # scene-level area

    # 对每个 room 生成一个样本（如果 room 有 bbox，则用 room bbox；否则用 scene bbox）
    for room in rooms:
        rt = get_room_type(room)

        room_area = bbox_area_from_obj(room)
        area = room_area if room_area is not None else scene_area

        # density 定义：furniture_count / area
        # 如果没有 area，就退化为 furniture_count（仍能聚类）
        if area is None:
            density = float(n_furn)
        else:
            density = float(n_furn) / (float(area) + 1e-12)

        if np.isfinite(density):
            density_list.append(density)
            roomtype_list.append(rt)

density = np.array(density_list, dtype=float)
roomtypes = np.array(roomtype_list, dtype=object)

print("Total room samples:", len(density))
print("Unique room types:", len(np.unique(roomtypes)))
print("Density stats:",
      f"min={density.min():.4g}, p50={np.median(density):.4g}, p95={np.percentile(density,95):.4g}, max={density.max():.4g}")

# -------------------------
# 2) Cluster into 3 styles by density (1D)
# -------------------------
X = density.reshape(-1, 1)

# 标准化 + KMeans（可复现）
Xz = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=N_STYLE, random_state=0, n_init=20)
style_raw = km.fit_predict(Xz)

# 让风格标签按“密度从低到高”排序：低=宽敞，中=中性，高=精致
centers = []
for s in range(N_STYLE):
    centers.append(np.mean(density[style_raw == s]))
centers = np.array(centers)

order = np.argsort(centers)  # 低密度 -> 高密度
remap = {int(order[i]): i for i in range(N_STYLE)}  # 原簇id -> 排序后id(0..2)

style_id = np.array([remap[int(s)] for s in style_raw], dtype=int)

# 显示每个风格组的密度中心/样本量
for i in range(N_STYLE):
    idx = np.where(style_id == i)[0]
    print(f"Style {i} ({STYLE_NAMES_CN[i]}): n={len(idx)}, mean_density={density[idx].mean():.6g}")

# 基本 sanity check
if np.min([np.sum(style_id == i) for i in range(N_STYLE)]) < MIN_SAMPLES_PER_STYLE:
    print("[Warning] Some style cluster is very small; density feature or bbox area may be missing/unstable.")

# -------------------------
# 3) Count room-type distribution within each style
# -------------------------
# 先挑 TOP_ROOM_TYPES，其余合并为 "Other"
global_room_counts = Counter(roomtypes.tolist())
top_rooms = [r for r, _ in global_room_counts.most_common(TOP_ROOM_TYPES)]
top_rooms_set = set(top_rooms)

def canonical_room(rt):
    return rt if rt in top_rooms_set else "Other"

# 统计：style -> room_type -> count
style_room_counts = [Counter() for _ in range(N_STYLE)]
for s, rt in zip(style_id, roomtypes):
    style_room_counts[int(s)][canonical_room(rt)] += 1

# 统一 room_type 顺序：Top rooms + Other
room_order = top_rooms + (["Other"] if "Other" in set(canonical_room(r) for r in roomtypes) else [])

# 构造堆叠柱状图矩阵：rows=room_type, cols=style
M = np.zeros((len(room_order), N_STYLE), dtype=int)
for i, r in enumerate(room_order):
    for s in range(N_STYLE):
        M[i, s] = style_room_counts[s][r]

# 同时输出比例（可写论文）
print("\nRoom-type proportions within each style:")
for s in range(N_STYLE):
    total = sum(style_room_counts[s].values())
    print(f"Style {s} ({STYLE_NAMES_CN[s]}) total={total}")
    for r in room_order[:min(8, len(room_order))]:
        print(f"  {r:16s}: {style_room_counts[s][r]/(total+1e-12):.3f}")
    print("  ...")

# -------------------------
# 4) Plot: stacked bar chart (styles on x-axis, stacked by room types)
# -------------------------
x = np.arange(N_STYLE)
bottom = np.zeros(N_STYLE, dtype=int)

plt.figure(figsize=(10, 6))

for i, r in enumerate(room_order):
    vals = M[i, :]
    plt.bar(x, vals, bottom=bottom, label=r)
    bottom += vals

plt.xticks(x, [f"{STYLE_NAMES_CN[i]}\n(mean dens={centers[order[i]]:.3g})" for i in range(N_STYLE)])
plt.ylabel("Number of room samples")
plt.title("Room-type Distribution under Density-based Style Clusters (3D-FRONT)")
plt.legend(ncol=2, fontsize=9, frameon=False)
plt.tight_layout()
plt.show()

# -------------------------
# 5) Optional: plot style cluster density distribution
# -------------------------
plt.figure(figsize=(10, 4))
for i in range(N_STYLE):
    d = density[style_id == i]
    # 裁剪极端值，避免长尾影响观感
    clip = np.percentile(d, 99)
    plt.hist(np.clip(d, 0, clip), bins=60, alpha=0.5, label=f"{STYLE_NAMES_CN[i]} (n={len(d)})", density=True)

plt.xlabel("Density = furniture_count / area  (clipped at 99th percentile)")
plt.ylabel("Density (normalized)")
plt.title("Density Distributions of Style Clusters")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 为了观感，对 density 取 log（避免长尾压扁）
log_density = np.log10(density + 1e-6)

# 为每个 style 一个 y 轴抖动
y = style_id + np.random.uniform(-0.15, 0.15, size=len(style_id))

plt.figure(figsize=(10, 5))
plt.scatter(log_density, y, s=6, alpha=0.25)

plt.yticks([0, 1, 2], ["宽敞", "中性", "精致"])
plt.xlabel("log10(Density)")
plt.title("Scatter of Room Density across Style Clusters")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

for i, name in enumerate(["宽敞", "中性", "精致"]):
    d = density[style_id == i]
    clip = np.percentile(d, 99)
    plt.hist(np.clip(d, 0, clip), bins=80, density=True,
             alpha=0.45, label=f"{name} (n={len(d)})")

plt.xlabel("Density (clipped at 99th percentile)")
plt.ylabel("Probability density")
plt.title("Density Distributions of Style Clusters")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 为了观感，对 density 取 log（避免长尾压扁）
log_density = np.log10(density + 1e-6)

# 为每个 style 一个 y 轴抖动
y = style_id + np.random.uniform(-0.15, 0.15, size=len(style_id))

plt.figure(figsize=(10, 5))
plt.scatter(log_density, y, s=6, alpha=0.25)

plt.yticks([0, 1, 2], ["宽敞", "中性", "精致"])
plt.xlabel("log10(Density)")
plt.title("Scatter of Room Density across Style Clusters")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
data = [density[style_id == i] for i in range(3)]
plt.violinplot(data, showmeans=True, showmedians=True)

plt.xticks([1, 2, 3], ["Spacious", "Neutral", "Elegant"])
plt.ylabel("Density")
plt.title("Violin Plot of Density by Style")
plt.tight_layout()
plt.show()

# 给房型编码
room_ids = {r: i for i, r in enumerate(np.unique(roomtypes))}
room_y = np.array([room_ids[r] for r in roomtypes])

plt.figure(figsize=(12, 6))
plt.scatter(log_density, room_y, c=style_id, s=4, alpha=0.3)

plt.yticks(list(room_ids.values())[::3],
           list(room_ids.keys())[::3])
plt.xlabel("log10(Density)")
plt.ylabel("Room Type")
plt.title("Room Type vs Density Colored by Style")
plt.tight_layout()
plt.show()
