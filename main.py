import blenderproc as bproc
import argparse
import sys
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('front_dir', help="Path to 3D-Front JSON folder or single file")
parser.add_argument('future_folder', help="Path to 3D-Future model folder")
parser.add_argument('front_3D_texture_path', help="Path to 3D-Front texture folder")
parser.add_argument('output_dir', help="Path to output directory")
args = parser.parse_args()

bproc.init()

# 1. 自动处理目录：获取目录下所有的 JSON 文件
if os.path.isdir(args.front_dir):
    json_files = [os.path.join(args.front_dir, f) for f in os.listdir(args.front_dir) if f.endswith('.json')]
else:
    json_files = [args.front_dir]

# 2. 标签映射 (示例使用默认映射)
mapping = bproc.utility.LabelIdMapping.from_dict({"background": 0})

# 全局渲染设置（只需设置一次）
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# 3. 循环处理每个场景
total_files = len(json_files)
for index, json_path in enumerate(json_files):
    
    # 打印进度提示
    print(f"--- Processing Scene {index + 1}/{total_files}: {os.path.basename(json_path)} ---")
    
    # 清理上一个场景的对象和相机姿态
    bproc.clean_up()
    
    try:
        # 加载场景
        loaded_objects = bproc.loader.load_front3d(
            json_path=json_path,
            future_model_path=args.future_folder,
            front_3D_texture_path=args.front_3D_texture_path,
            label_mapping=mapping
        )

        # 构建 BVH 树用于遮挡检测
        bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if o.is_mesh])
        
        # 场景采样点准备 (在房间内部采样)
        point_sampler = bproc.sampler.UpperRegionSampler(loaded_objects)

        # 相机采样逻辑
        proximity_checks = {"min": 1.0, "avg": {"min": 2.5, "max": 3.5}, "no_background": True}
        tries = 0
        poses = 0
        
        while tries < 1000 and poses < 5: # 每个场景尝试生成5个视角
            height = np.random.uniform(1.4, 1.8)
            location = point_sampler.sample(height)
            rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

            # 验证姿态：检查障碍物距离和背景可见性
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
                bproc.camera.add_camera_pose(cam2world_matrix)
                poses += 1
            tries += 1

        if poses == 0:
            print(f"Warning: Could not find valid camera poses for {json_path}")
            continue

        # 渲染并写入
        data = bproc.renderer.render()
        
        # 自动根据 JSON 文件名创建输出子路径
        scene_name = os.path.basename(json_path).replace(".json", "")
        scene_output_dir = os.path.join(args.output_dir, scene_name)
        
        # 确保输出目录存在
        if not os.path.exists(scene_output_dir):
            os.makedirs(scene_output_dir)
            
        bproc.writer.write_hdf5(scene_output_dir, data)
        print(f"Successfully saved to {scene_output_dir}")

    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        continue

bproc.clean_up()
print("All tasks finished.")