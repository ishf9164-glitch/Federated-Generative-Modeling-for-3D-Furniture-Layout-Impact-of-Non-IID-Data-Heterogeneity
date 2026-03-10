import os
import csv
import yaml
import copy
from tqdm import tqdm

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def generate_client_configs():
    # --- 1. 路径设置 ---
    # client 文件夹所在的根目录 (例如 E:\diverse_synth\dump)
    root_dir = r"E:/diverse_synth/dump"          
    # 配置文件生成的存放目录 (例如 E:\diverse_synth\config)
    config_out_dir = r"E:/diverse_synth/config"   
    
    # --- 2. 模板配置 ---
    template_data = {
        "data": {
            "dataset_type": "cached_threedfront",
            "room_type": "total",
            "encoding_type": "cached",
            "dataset_directory": "",  
            "annotation_file": "",    
            "room_type_filter": "no_filtering",
            "train_stats": "dataset_stats.txt",
            "room_layout_size": "256,256",
            "num_each_class": 4,
            "num_class": 23,
            "data_dim": 16,
            "half_range": 6,
            "interval": 0.3
        },
        "training": {
            "splits": ["train", "val"],
            "tag": "",                
            "checkpoint_dir": "../savepoint/",
            "epochs": 100,
            "batch_size": 64,
            "save_frequency": 100,
            "optimizer": "Adam",
            "lr": 0.001,
            "weight_decay": 0,
            "adjust_kl_divergence": False
        },
        "validation": {
            "splits": ["test"],
            "frequency": 5,
            "batch_size": 16
        },
        "generate": {
            "output_path": "../output/"
        },
        "EnhancedBetaTCVAE": {
            "input_dim": 16,
            "latent_dimension": 32,
            "bn_momentum": 0.0005,
            "kld_weight": 0.0001,
            "kld_interval": 50,
            "sparse_num": 4,
            "embedding_dim1": 32,
            "embedding_dim2": 48,
            "embedding_dim3": 64,
            "sparse_embedding1": 256,
            "sparse_embedding2": 128,
            "sparse_embedding3": 64,
            "linear_embedding1": 32,
            "linear_embedding2": 16,
            "linear_embedding3": 4
        },
        "UNet3Plus": {
            "input_dim": 16,
            "bn_momentum": 0.01,
            "embedding_dim1": 64,
            "embedding_dim2": 128,
            "embedding_dim3": 256,
            "embedding_dim4": 512
        }
    }

    # 3. 扫描目录获取所有 client 文件夹
    if not os.path.exists(root_dir):
        print(f"错误: 找不到目录 {root_dir}")
        return

    client_dirs = [d for d in os.listdir(root_dir) 
                   if os.path.isdir(os.path.join(root_dir, d)) and "compound_client1" in d.lower()]

    if not client_dirs:
        print("未发现包含 'client' 字样的文件夹。")
        return

    print(f"检测到 {len(client_dirs)} 个客户端，正在生成配置...")
    ensure_dir(config_out_dir)

    for client_name in tqdm(client_dirs):
        # 实际房间所在的路径：E:\diverse_synth\dump\IID_client1\3D-FRONT
        client_front_path = os.path.join(root_dir, client_name, "3D-FRONT")
        
        if not os.path.exists(client_front_path):
            print(f"警告: 跳过 {client_name}，因为找不到子文件夹 3D-FRONT")
            continue
        
        # --- A. 生成 CSV 分拆文件 ---
        csv_filename = f"{client_name}_threed_front_splits.csv"
        csv_path = os.path.join(config_out_dir, csv_filename)
        
        # 扫描 3D-FRONT 下的所有房间子目录
        rooms = [r for r in os.listdir(client_front_path) 
                 if os.path.isdir(os.path.join(client_front_path, r))]
        
        # 按照 8:1:1 划分
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            n = len(rooms)
            for i, rid in enumerate(rooms):
                if i < int(n * 0.8):
                    split = "train"
                elif i < int(n * 0.9):
                    split = "val"
                else:
                    split = "test"
                writer.writerow([rid, split])

        # --- B. 生成 YAML 配置文件 ---
        yaml_filename = f"{client_name}_config.yaml"
        yaml_path = os.path.join(config_out_dir, yaml_filename)
        
        client_cfg = copy.deepcopy(template_data)
        
        # 修改数据路径
        # 根据你的需求，数据目录应指向 3D-FRONT 这一层
        # 计算从 config 目录到 dump/client/3D-FRONT 的相对路径
        rel_data_dir = os.path.relpath(client_front_path, config_out_dir).replace("\\", "/")
        
        client_cfg["data"]["dataset_directory"] = rel_data_dir
        # 根据示例，标注文件路径指向生成的 CSV
        client_cfg["data"]["annotation_file"] = f"../config/{csv_filename}"
        client_cfg["training"]["tag"] = f"{client_name}_total"

        # 写入 YAML
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(client_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\n成功！CSV 和 YAML 已保存在: {config_out_dir}")

if __name__ == "__main__":
    generate_client_configs()