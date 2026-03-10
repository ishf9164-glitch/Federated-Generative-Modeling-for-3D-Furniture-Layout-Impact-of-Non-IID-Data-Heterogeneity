import os
import csv
import json
import argparse
import numpy as np
import sys
from tqdm import tqdm

# 路径修复
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthesis.datasets.FRONT import Front

def read_scene_ids_from_splits(csv_path, use_splits=("train", "val")):
    ids = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row: continue
            sid = row[0].strip()
            sp = row[1].strip() if len(row) > 1 else ""
            if sp in use_splits and sid:
                ids.add(sid)
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", required=True)
    ap.add_argument("--dataset_directory", required=True, help="指向包含 3D-FRONT 子目录的父目录")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threed_future_dataset_directory", default="../dump/3D-FUTURE-model")
    ap.add_argument("--model_info", default="../dump/3D-FUTURE-model/model_info.json")
    ap.add_argument("--train_stats_name", default="dataset_stats.txt")
    ap.add_argument("--without_lamps", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 加载 CSV IDs
    target_ids = read_scene_ids_from_splits(args.splits_csv, use_splits=("train", "val"))
    print(f"\n[Step 1] Loaded {len(target_ids)} IDs from CSV.")
    if len(target_ids) > 0:
        print(f"Sample CSV ID: '{list(target_ids)[0]}'")

    # 调试计数器
    debug_info = {"checked": 0, "matched": 0}

    # 2) 重新定义的过滤器
    def room_filter(room):
        debug_info["checked"] += 1
        
        rid = room.scene_id
        raw_uid = getattr(room, "uid", "")
        
        # --- 核心修正逻辑：处理 Windows 路径前缀 ---
        # 1. 将反斜杠统一替换为斜杠
        normalized_uid = raw_uid.replace("\\", "/")
        # 2. 提取路径最后一部分（即真正的文件夹 ID），去除 "3D-FRONT/" 这种前缀
        clean_uid = normalized_uid.split("/")[-1]
        
        # 调试打印：确认清洗后的 ID 是否匹配 CSV 格式
        if debug_info["checked"] <= 5:
            print(f"DEBUG: Checking Room -> scene_id: '{rid}', raw_uid: '{raw_uid}', clean_uid: '{clean_uid}'")

        # 使用清洗后的 clean_uid 进行匹配
        is_match = (clean_uid in target_ids) or (rid in target_ids)
        
        if is_match:
            debug_info["matched"] += 1
            if args.without_lamps:
                if any(getattr(b, "label", "") == "lamp" for b in room.bboxes):
                    return False
            return True
        return False

    # 3) 加载
    print(f"\n[Step 2] Loading dataset from: {args.dataset_directory} ...")
    
    # 检查 3D-FRONT 目录是否存在
    check_path = os.path.join(args.dataset_directory, "3D-FRONT")
    if not os.path.exists(check_path):
        print(f"WARNING: '{check_path}' not found! The loader might fail.")

    dataset = Front.load_dataset(
        dataset_directory=args.dataset_directory,
        path_to_model_info=args.model_info,
        path_to_models=args.threed_future_dataset_directory,
        room_type_filter=room_filter
    )

    print(f"\n[Step 3] Filter Stats: Checked {debug_info['checked']}, Matched {debug_info['matched']}")

    # 4) 保存数据 (保持原有逻辑)
    if len(dataset) == 0:
        print("\nERROR: Dataset is still empty. Please check the 'clean_uid' in DEBUG output above.")
        print("Does it match the CSV Sample ID exactly?")
        return

    trans_bounds = dataset.bounds["translations"]
    size_bounds = dataset.bounds["sizes"]
    angle_bounds = dataset.bounds["angles"]

    dataset_stats = {
        "bounds_translations": trans_bounds[0].tolist() + trans_bounds[1].tolist(),
        "bounds_sizes": size_bounds[0].tolist() + size_bounds[1].tolist(),
        "bounds_angles": angle_bounds[0].tolist() + angle_bounds[1].tolist(),
        "class_labels": dataset.class_labels,
        "class_order": dataset.class_order,
        "furniture_limit": dataset.furniture_limit,
    }

    out_path = os.path.join(args.out_dir, args.train_stats_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f)

    print(f"\nSuccessfully saved stats to: {out_path}")
    print(f"Total rooms included in stats: {len(dataset)}")

if __name__ == "__main__":
    main()