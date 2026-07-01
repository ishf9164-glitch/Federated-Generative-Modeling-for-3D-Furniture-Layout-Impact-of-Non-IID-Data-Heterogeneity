import os, json
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from scipy.stats import wasserstein_distance

DATA_DIR = r"E:\diverse_synth\dump\3D-FRONT"
TOP_FURN = 80          # 家具类别词表大小
MIN_ROOMS_PER_CLIENT = 50

# -------------------------
# Utils
# -------------------------
def extract_furniture_items(data: dict):
    furn = data.get("furniture", [])
    return furn if isinstance(furn, list) else []

def get_furn_category(f: dict):
    for k in ["category", "type", "name", "model", "modelId", "jid"]:
        if isinstance(f, dict) and k in f and f[k] is not None:
            return str(f[k])
    return "Unknown"

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

# 1D EMD on discrete vocab positions
def emd_discrete(p, q, positions):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return wasserstein_distance(positions, positions, u_weights=p, v_weights=q)

# -------------------------
# Load file list
# -------------------------
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
if not files:
    raise RuntimeError(f"No .json files found in {DATA_DIR}")
print("Num JSON files:", len(files))

# -------------------------
# Pass-1: Build furniture vocab (Top-K)
# -------------------------
cat_counter = Counter()
for fn in tqdm(files, desc="Pass-1: build furniture vocab"):
    with open(os.path.join(DATA_DIR, fn), "r", encoding="utf-8") as fp:
        data = json.load(fp)
    for furn in extract_furniture_items(data):
        cat_counter[get_furn_category(furn)] += 1

vocab = [c for c, _ in cat_counter.most_common(TOP_FURN)]
vocab_index = {c: i for i, c in enumerate(vocab)}
positions = np.arange(TOP_FURN, dtype=float)

print("Vocab size:", len(vocab), "Top-5:", vocab[:5])

# -------------------------
# Pass-2: Accumulate per-roomType histograms (client = room type)
# We compute each scene's normalized furniture hist, then assign it to each room in that scene.
# -------------------------
sum_hist_global = np.zeros(TOP_FURN, dtype=float)
n_rooms_global = 0

sum_hist_by_roomtype = defaultdict(lambda: np.zeros(TOP_FURN, dtype=float))
count_by_roomtype = Counter()

for fn in tqdm(files, desc="Pass-2: accumulate hist per room"):
    with open(os.path.join(DATA_DIR, fn), "r", encoding="utf-8") as fp:
        data = json.load(fp)

    rooms = get_rooms(data)
    if not rooms:
        continue

    furn_list = extract_furniture_items(data)

    # scene-level normalized histogram over TOP_FURN
    hist = np.zeros(TOP_FURN, dtype=float)
    for furn in furn_list:
        cat = get_furn_category(furn)
        idx = vocab_index.get(cat, None)
        if idx is not None:
            hist[idx] += 1.0

    total = hist.sum()
    if total > 0:
        hist = hist / total  # normalize to probability mass on vocab

    # assign to each room sample in this scene
    for room in rooms:
        rt = get_room_type(room)

        sum_hist_by_roomtype[rt] += hist
        count_by_roomtype[rt] += 1

        sum_hist_global += hist
        n_rooms_global += 1

print("\nTotal room samples:", n_rooms_global)
print("Num room types:", len(count_by_roomtype))

# Global distribution p(c)
p_global = sum_hist_global / (n_rooms_global + 1e-12)
p_global = p_global / (p_global.sum() + 1e-12)

# -------------------------
# Compute EMD per client (room type)
# -------------------------
emd_per_client = {}
for rt, cnt in count_by_roomtype.items():
    if cnt < MIN_ROOMS_PER_CLIENT:
        continue
    p_k = sum_hist_by_roomtype[rt] / (cnt + 1e-12)
    p_k = p_k / (p_k.sum() + 1e-12)
    emd_per_client[rt] = emd_discrete(p_k, p_global, positions)

emd_sorted = sorted(emd_per_client.items(), key=lambda x: x[1], reverse=True)

print("\nTop-10 room clients by EMD (label-skew on furniture categories):")
for k, v in emd_sorted[:10]:
    print(f"{k:20s}  rooms={count_by_roomtype[k]:6d}  EMD={v:.4f}")

vals = np.array(list(emd_per_client.values()), dtype=float)
print("\nMean EMD:", float(vals.mean()) if len(vals) else None)
print("Std  EMD:", float(vals.std()) if len(vals) else None)
print("Num clients used (>=MIN_ROOMS_PER_CLIENT):", len(emd_per_client))

import matplotlib.pyplot as plt
import numpy as np

# ===== 你前面已经有：emd_per_client, count_by_roomtype, MIN_ROOMS_PER_CLIENT =====
# emd_per_client: dict {room_type: emd_value}
# count_by_roomtype: Counter {room_type: num_rooms}

# ---------- 1) Top-K EMD bar chart ----------
TOPK = 10
emd_sorted = sorted(emd_per_client.items(), key=lambda x: x[1], reverse=True)
top = emd_sorted[:TOPK]

top_labels = [k for k, _ in top]
top_vals = [v for _, v in top]
top_counts = [count_by_roomtype[k] for k in top_labels]

plt.figure(figsize=(10, 5))
bars = plt.bar(range(TOPK), top_vals)
plt.xticks(range(TOPK), top_labels, rotation=35, ha="right")
plt.ylabel("EMD (Wasserstein-1)")
plt.title(f"Top-{TOPK} Room-type Clients by EMD (label-skew)")

# 在柱子上标 EMD + rooms 数
for i, (b, emd_v, cnt) in enumerate(zip(bars, top_vals, top_counts)):
    h = b.get_height()
    plt.text(
        b.get_x() + b.get_width() / 2,
        h,
        f"{emd_v:.2f}\n(n={cnt})",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.show()

# ---------- 2) EMD distribution (hist + mean line) ----------
all_vals = np.array([v for _, v in emd_sorted], dtype=float)

plt.figure(figsize=(8, 5))
plt.hist(all_vals, bins=12, density=False)
mean_v = float(all_vals.mean()) if len(all_vals) else 0.0
plt.axvline(mean_v, linewidth=2)  # 默认颜色即可
plt.xlabel("EMD (Wasserstein-1)")
plt.ylabel("Number of clients")
plt.title(f"Distribution of EMD Across Clients (mean={mean_v:.2f}, std={all_vals.std():.2f})")
plt.tight_layout()
plt.show()

# ---------- 3) Optional: sorted curve (Lorenz-like view) ----------
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(all_vals) + 1), np.sort(all_vals)[::-1], marker="o", linewidth=1)
plt.xlabel("Client rank (sorted by EMD)")
plt.ylabel("EMD (Wasserstein-1)")
plt.title("EMD Ranking Curve Across Clients")
plt.tight_layout()
plt.show()
