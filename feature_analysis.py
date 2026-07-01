# analyze_3dfront_density.py
# Density = (sum furniture footprint areas) / (floor area in XZ plane)
#
# Assumptions aligned with the uploaded 3D-FRONT json example:
# - scene["mesh"] contains multiple meshes, including type "Floor" (preferred).
# - each mesh has "xyz" (flat list of floats), "faces" (flat list of ints, triangles).
# - furniture footprint uses bbox[0]*bbox[1] if bbox is [a,b,c] numeric; else size[0]*size[1].
# - skips furniture with valid==False.

import argparse
import json
import math
import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def triangle_area_xz(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    x0, z0 = float(v0[0]), float(v0[2])
    x1, z1 = float(v1[0]), float(v1[2])
    x2, z2 = float(v2[0]), float(v2[2])
    return 0.5 * abs((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0))


def mesh_area_xz(mesh: Dict[str, Any]) -> float:
    xyz = np.asarray(mesh.get("xyz", []), dtype=float)
    faces = np.asarray(mesh.get("faces", []), dtype=int)

    if xyz.size % 3 != 0 or faces.size % 3 != 0 or xyz.size == 0 or faces.size == 0:
        return 0.0

    verts = xyz.reshape(-1, 3)
    tri = faces.reshape(-1, 3)

    # guard face indices
    if tri.max(initial=-1) >= len(verts) or tri.min(initial=0) < 0:
        return 0.0

    area = 0.0
    for a, b, c in tri:
        area += triangle_area_xz(verts[a], verts[b], verts[c])
    return float(area)


def floor_area_from_scene(scene: Dict[str, Any]) -> float:
    meshes = scene.get("mesh", [])
    if not isinstance(meshes, list):
        return 0.0

    def sum_by_type(t: str) -> float:
        return sum(mesh_area_xz(m) for m in meshes if isinstance(m, dict) and m.get("type") == t)

    # Prefer explicit Floor meshes; fallback to SlabBottom if absent.
    area_floor = sum_by_type("Floor")
    if area_floor > 0:
        return area_floor

    area_slab = sum_by_type("SlabBottom")
    return area_slab


def furniture_footprint_area(item: Dict[str, Any]) -> Optional[float]:
    if item.get("valid") is False:
        return None

    dims = None
    bbox = item.get("bbox", None)
    if isinstance(bbox, list) and len(bbox) == 3 and all(isinstance(x, (int, float)) for x in bbox):
        dims = bbox
    else:
        size = item.get("size", None)
        if isinstance(size, list) and len(size) >= 2 and all(isinstance(x, (int, float)) for x in size[:2]):
            dims = size

    if dims is None:
        return None

    a = float(dims[0]) * float(dims[1])
    if not math.isfinite(a) or a <= 0:
        return None
    return a


def density_from_scene(scene: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, float]]:
    floor_area = floor_area_from_scene(scene)
    if not math.isfinite(floor_area) or floor_area <= 0:
        return None, {"floor_area": float(floor_area) if math.isfinite(floor_area) else 0.0,
                      "furn_area": 0.0}

    furn = scene.get("furniture", [])
    furn_area = 0.0
    if isinstance(furn, list):
        for it in furn:
            if not isinstance(it, dict):
                continue
            a = furniture_footprint_area(it)
            if a is not None:
                furn_area += a

    dens = furn_area / floor_area
    return float(dens), {"floor_area": float(floor_area), "furn_area": float(furn_area)}


def find_json_files(root: str, pattern: str, recursive: bool) -> List[str]:
    if os.path.isfile(root) and root.lower().endswith(".json"):
        return [root]

    if recursive:
        return sorted(glob(os.path.join(root, "**", pattern), recursive=True))
    return sorted(glob(os.path.join(root, pattern), recursive=False))


def save_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    import csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = ["path", "uid", "density", "floor_area", "furn_area"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def plot_hist(d: np.ndarray, q1: float, q2: float, out_png: str) -> None:
    plt.figure()
    plt.hist(d, bins=50)
    plt.axvline(q1, linestyle="--")
    plt.axvline(q2, linestyle="--")
    plt.title("Density distribution (hist)")
    plt.xlabel("density = furniture_area / floor_area")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_cdf(d: np.ndarray, q1: float, q2: float, out_png: str) -> None:
    xs = np.sort(d)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.figure()
    plt.plot(xs, ys)
    plt.axvline(q1, linestyle="--")
    plt.axvline(q2, linestyle="--")
    plt.title("Density CDF")
    plt.xlabel("density")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_box(d: np.ndarray, out_png: str) -> None:
    plt.figure()
    plt.boxplot(d, vert=True, showfliers=True)
    plt.title("Density boxplot")
    plt.ylabel("density")
    plt.xticks([1], ["all"])
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_hist_and_cdf(d: np.ndarray, q1: float, q2: float, out_png: str) -> None:
    d = np.asarray(d, dtype=float)
    # 使用科学出版物常见的配置
    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 1. 绘制直方图并分段着色，体现非独立同分布的分布特性
    bins = 60
    n, b, patches = ax1.hist(d, bins=bins, alpha=0.5, color='steelblue', label='Frequency')
    
    # 根据 q1, q2 对直方图分段着色
    for i in range(len(patches)):
        if b[i] < q1: patches[i].set_facecolor('#7fb3d5') # 稀疏
        elif b[i] < q2: patches[i].set_facecolor('#566573') # 中等
        else: patches[i].set_facecolor('#1b2631')          # 密集

    ax1.set_xlabel(r"Density: $\frac{\sum \text{footprint}}{\text{floor area}}$")
    ax1.set_ylabel("Count of Scenes")

    # 2. 绘制 CDF 曲线
    ax2 = ax1.twinx()
    xs = np.sort(d)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax2.plot(xs, ys, color='#a04000', linewidth=2, label='CDF')
    ax2.set_ylabel("Cumulative Distribution Function (CDF)")
    ax2.set_ylim(0.0, 1.05)

    # 3. 绘制分位线
    for x, label in zip([q1, q2], ['$q_{low}$', '$q_{high}$']):
        ax1.axvline(x, color='gray', linestyle="--", alpha=0.8)
        ax1.text(x + 0.01, ax1.get_ylim()[1]*0.9, label, color='black', fontweight='bold')

    # 图例设置
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, ["Histogram", "CDF"], loc='upper right')

    plt.title("Furniture Density Distribution Across Scenes")
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches='tight') # 提高分辨率
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Folder containing 3D-FRONT json files (or a single json file).")
    ap.add_argument("--pattern", type=str, default="*.json")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--outdir", type=str, default="output_analysis")
    ap.add_argument("--q_low", type=float, default=1/3)
    ap.add_argument("--q_high", type=float, default=2/3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = find_json_files(args.data_root, args.pattern, args.recursive)
    if len(files) == 0:
        raise SystemExit(f"No json found under: {args.data_root}")

    rows: List[Dict[str, Any]] = []
    densities: List[float] = []

    for fp in tqdm(files, desc="Computing density", unit="scene"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                scene = json.load(f)
        except Exception:
            continue

        dens, aux = density_from_scene(scene)
        if dens is None or not math.isfinite(dens):
            continue

        uid = scene.get("uid", "")
        rows.append({
            "path": fp,
            "uid": uid,
            "density": dens,
            "floor_area": aux["floor_area"],
            "furn_area": aux["furn_area"],
        })
        densities.append(dens)
    if len(densities) == 0:
        raise SystemExit("No valid densities computed. Check Floor meshes / bbox|size fields.")

    d = np.asarray(densities, dtype=float)

    q1 = float(np.quantile(d, args.q_low))
    q2 = float(np.quantile(d, args.q_high))

    # group counts
    sparse = int(np.sum(d < q1))
    neutral = int(np.sum((d >= q1) & (d < q2)))
    dense = int(np.sum(d >= q2))

    # summary
    summary_txt = os.path.join(args.outdir, "density_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"n={len(d)}\n")
        f.write(f"mean={float(np.mean(d))}\n")
        f.write(f"std={float(np.std(d))}\n")
        f.write(f"min={float(np.min(d))}\n")
        f.write(f"p25={float(np.quantile(d, 0.25))}\n")
        f.write(f"median={float(np.median(d))}\n")
        f.write(f"p75={float(np.quantile(d, 0.75))}\n")
        f.write(f"max={float(np.max(d))}\n\n")
        f.write("group boundaries (quantile-based):\n")
        f.write(f"q_low={args.q_low} -> {q1}\n")
        f.write(f"q_high={args.q_high} -> {q2}\n\n")
        f.write("group counts:\n")
        f.write(f"sparse (<q1): {sparse}\n")
        f.write(f"neutral ([q1,q2)): {neutral}\n")
        f.write(f"dense (>=q2): {dense}\n")

    # per-scene csv
    save_csv(rows, os.path.join(args.outdir, "density_per_scene.csv"))

    # plots
    plot_hist(d, q1, q2, os.path.join(args.outdir, "density_hist.png"))
    plot_cdf(d, q1, q2, os.path.join(args.outdir, "density_cdf.png"))
    plot_box(d, os.path.join(args.outdir, "density_box.png"))
    plot_hist_and_cdf(d, q1, q2, os.path.join(args.outdir, "density_hist_cdf.png"))
    print("Done.")
    print(f"Saved to: {args.outdir}")
    print(f"Boundaries: q1={q1}, q2={q2}")
    print(f"Counts: sparse={sparse}, neutral={neutral}, dense={dense}")


if __name__ == "__main__":
    main()