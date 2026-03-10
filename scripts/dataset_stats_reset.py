import os
import json

def overwrite_all_stats_files():
    # 根目录
    base_path = "/root/diverse_synth/dump"
    
    # 你的 23 类标准模板
    stats_template = {
        "bounds_translations": [-5.671399570951715, 0.0375, -5.716401580065309, 5.290714577596304, 3.6750878746899414, 5.4048500000000015],
        "bounds_sizes": [0.03998999999999994, 0.020000020334800084, 0.009999999999999884, 2.38027, 1.42123, 1.4137885],
        "bounds_angles": [-3.141592653589793, 3.141592653589793],
        "class_labels": ["desk", "nightstand", "bed", "wardrobe", "shelf", "tv_stand", "chair", "classic_chair", "dining_chair", "dressing_chair", "dressing_table", "dining_table", "coffee_table/tea_table", "end_table", "bar", "cabinet", "children_cabinet", "wine_cabinet", "shoe_cabinet", "stool", "sofa", "floor_lamp", "lamp"],
        "class_order": {"desk": 0, "nightstand": 1, "bed": 2, "wardrobe": 3, "shelf": 4, "tv_stand": 5, "chair": 6, "classic_chair": 7, "dining_chair": 8, "dressing_chair": 9, "dressing_table": 10, "dining_table": 11, "coffee_table/tea_table": 12, "end_table": 13, "bar": 14, "cabinet": 15, "children_cabinet": 16, "wine_cabinet": 17, "shoe_cabinet": 18, "stool": 19, "sofa": 20, "floor_lamp": 21, "lamp": 22},
        "furniture_limit": 4
    }

    if not os.path.exists(base_path):
        print(f"路径不存在: {base_path}")
        return

    count = 0
    # 递归查找所有名为 dataset_stats.txt 的文件
    for root, dirs, files in os.walk(base_path):
        if "dataset_stats.txt" in files:
            target_file = os.path.join(root, "dataset_stats.txt")
            
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(stats_template, f, ensure_ascii=False)
            
            print(f"已重置 ({count+1}): {target_file}")
            count += 1

    print(f"\n任务完成！共重置了 {count} 个统计文件。")

if __name__ == "__main__":
    overwrite_all_stats_files()