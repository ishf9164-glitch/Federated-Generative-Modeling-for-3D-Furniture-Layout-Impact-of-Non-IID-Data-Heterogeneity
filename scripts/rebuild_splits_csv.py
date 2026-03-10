# scripts/rebuild_splits_csv.py
# 作用：
# 1) 扫描 scene_dir 下的场景文件夹（形如 UUID_Bedroom-43072）
# 2) 提取 tag（Bedroom-43072）
# 3) 随机但可复现地划分 train/val/test
# 4) 覆盖写回 csv_path（不加 header，格式：tag<TAB>split）

import os
import argparse
import random

def extract_tag(folder_name: str) -> str | None:
    # 期望格式：<uuid>_<tag>
    # 例如：00110bde-..._Bedroom-43072 -> Bedroom-43072
    if "_" not in folder_name:
        return None
    tag = folder_name.split("_", 1)[1]
    return tag.strip() if tag.strip() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="e.g. ../dump/bedroom/3D-FRONT")
    ap.add_argument("--csv_path", required=True, help="existing splits csv path to overwrite")
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--ext", default="", help="optional: only keep folders endingwith this string")
    args = ap.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    if not os.path.isdir(args.scene_dir):
        raise FileNotFoundError(f"scene_dir not found: {args.scene_dir}")

    # 列出所有场景文件夹
    folders = [f for f in os.listdir(args.scene_dir) if os.path.isdir(os.path.join(args.scene_dir, f))]
    if args.ext:
        folders = [f for f in folders if f.endswith(args.ext)]

    tags = []
    bad = []
    for f in folders:
        tag = extract_tag(f)
        if tag is None:
            bad.append(f)
        else:
            tags.append(tag)

    tags = sorted(set(tags))  # 去重 + 稳定排序
    if len(tags) == 0:
        raise RuntimeError(
            f"No valid tags found in {args.scene_dir}. "
            f"Example bad folders: {bad[:5]}"
        )

    # 可复现随机划分
    rng = random.Random(args.seed)
    rng.shuffle(tags)

    n = len(tags)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    # 剩余都给 test，避免四舍五入丢样本
    n_test = n - n_train - n_val

    train_tags = tags[:n_train]
    val_tags = tags[n_train:n_train + n_val]
    test_tags = tags[n_train + n_val:]

    assert len(test_tags) == n_test

    # 写 CSV：两列，无 header，tab 分隔（兼容你现有格式）
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_path)), exist_ok=True)
    with open(args.csv_path, "w", encoding="utf-8") as w:
        for t in train_tags:
            w.write(f"{t}\ttrain\n")
        for t in val_tags:
            w.write(f"{t}\tval\n")
        for t in test_tags:
            w.write(f"{t}\ttest\n")

    print(f"[OK] scene_dir: {args.scene_dir}")
    print(f"[OK] wrote csv: {args.csv_path}")
    print(f"[OK] counts: train={len(train_tags)}, val={len(val_tags)}, test={len(test_tags)}, total={n}")
    if bad:
        print(f"[WARN] {len(bad)} folders skipped (no '_' found). Example: {bad[:3]}")

if __name__ == "__main__":
    main()
