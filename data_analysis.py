import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def run_full_analysis(root_dir, output_dir='output_analysis'):
    os.makedirs(output_dir, exist_ok=True)
    room_data = [] # 用于存储所有房间类型的列表

    all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) 
                 for f in filenames if f.endswith('.json')]
    
    for file_path in tqdm(all_files, desc="Parsing all rooms"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                scene = json.load(f)
            
            # 【关键点】找到房间列表，并遍历其中的每一个房间
            # 兼容不同层级结构
            rooms = []
            if 'scene' in scene and 'room' in scene['scene']:
                rooms = scene['scene']['room']
            elif 'room' in scene:
                rooms = scene['room']
            
            for room in rooms:
                room_type = room.get('type', 'unknown')
                if room_type != 'unknown':
                    room_data.append(room_type)
                    
        except Exception:
            continue

    # 统计数据
    df_rooms = pd.Series(room_data).value_counts()
    
    # --- 绘图 ---
    plt.figure(figsize=(16, 8))
    # 使用 seaborn 绘制柱状图，效果更美观
    sns.barplot(x=df_rooms.index, y=df_rooms.values, palette="viridis")
    
    plt.title("Full Distribution of Room Types")
    plt.ylabel("Number of Samples")
    plt.xlabel("Room Type")
    plt.xticks(rotation=45, ha='right') # 旋转标签以防重叠
    plt.tight_layout() # 自动调整布局，防止标签被截断
    plt.savefig(os.path.join(output_dir, "full_room_type_distribution.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_full_analysis('dump/3D-FRONT')