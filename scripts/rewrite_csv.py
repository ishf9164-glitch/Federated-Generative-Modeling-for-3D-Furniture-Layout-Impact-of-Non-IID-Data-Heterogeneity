import os
import argparse
import random
from collections import Counter

def detect_delim(line: str):
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return None

def parse_example_ratios(example_csv: str):
    cnt = Counter()
    with open(example_csv, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            delim = detect_delim(s)
            if delim is None:
                continue
            parts = [p.strip() for p in s.split(delim) if p.strip() != ""]
            if len(parts) < 2:
                continue
            split = parts[-1].lower()
            if split in ("train", "val", "test"):
                cnt[split] += 1
    total = cnt["train"] + cnt["val"] + cnt["test"]
    if total == 0:
        raise RuntimeError(f"example_csv 里没读到 train/val/test: {example_csv}")
    return cnt, (cnt["train"]/total, cnt["val"]/total, cnt["test"]/total)

def extract_scene_id(folder_name: str):
    # UUID_Tag -> Tag
    if "_" not in folder_name:
        return None
    return folder_name.split("_", 1)[1].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="e.g. /dump/bedroom/3D-FRONT")
    ap.add_argument("--example_csv", required=True, help="e.g. /mnt/data/neutral_threed_front_splits.csv")
    ap.add_argument("--out_csv", required=True, help="e.g. ../config/bedroom_threed_front_splits.csv")
    ap.add_argument("--seed", type=int, default=2023)
    args = ap.parse_args()

    cnt, (r_tr, r_val, r_te) = parse_example_ratios(args.example_csv)

    if not os.path.isdir(args.scene_dir):
        raise FileNotFoundError(args.scene_dir)

    folders = [f for f in os.listdir(args.scene_dir) if os.path.isdir(os.path.join(args.scene_dir, f))]
    scene_ids = []
    for f in folders:
        sid = extract_scene_id(f)
        if sid:
            scene_ids.append(sid)

    scene_ids = sorted(set(scene_ids))
    if not scene_ids:
        raise RuntimeError(f"No scene_ids extracted from {args.scene_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(scene_ids)

    n = len(scene_ids)
    n_train = int(n * r_tr)
    n_val = int(n * r_val)
    n_test = n - n_train - n_val

    train_list = scene_ids[:n_train]
    val_list = scene_ids[n_train:n_train + n_val]
    test_list = scene_ids[n_train + n_val:]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8") as w:
        for sid in train_list:
            w.write(f"{sid},train\n")
        for sid in val_list:
            w.write(f"{sid},val\n")
        for sid in test_list:
            w.write(f"{sid},test\n")

    print("[OK] example counts:", dict(cnt))
    print(f"[OK] ratios: train={r_tr:.6f}, val={r_val:.6f}, test={r_te:.6f}")
    print(f"[OK] wrote: {args.out_csv}")
    print(f"[OK] total={n} train={len(train_list)} val={len(val_list)} test={len(test_list)}")
    print(f"[OK] sample: {train_list[0]},train")

if __name__ == "__main__":
    main()
