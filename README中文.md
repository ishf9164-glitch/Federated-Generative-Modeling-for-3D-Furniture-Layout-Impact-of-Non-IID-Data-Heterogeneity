# 使用文档

本项目基于 3D-FRONT 和 3D-FUTURE 数据集，实现了一套完整的 3D 场景家具布局生成框架。项目支持单机训练以及基于联邦学习（FedAVG）的多客户端协同训练方案，涵盖了数据预处理、模型训练、场景生成、后处理及多维度指标评估。

## 环境依赖

环境依赖如下：

- numpy
- cython
- pillow
- pyyaml
- pyrr
- torch && torchvision
- trimesh
- tqdm
- matlab
- cleanfid
- scipy
- wandb
- simple-3dviz

## 数据集（下载地址：[dataset](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) )

- 3D-FRONT 
- 3D-FUTURE dataset

<!-- 查看数据集的布局示意图 -->
blenderproc run front3d_single_bedroom_render.py "E:/diverse_synth/dump/3D-FRONT/0af309fd-4a34-4aa2-bbe9-610edadf747e.json" "E:/diverse_synth/dump/3D-FUTURE-model" "./output_single_bedroom_paperview" --bedroom_index 0 --resolution 1600 --samples 128 --camera_mode paper_view --margin 0.8

<!-- 主视角的示意图 -->
blenderproc run front3d_bedroom_render.py "E:/diverse_synth/dump/3D-FRONT/0a8d471a-2587-458a-9214-586e003e9cf9.json" "E:/diverse_synth/dump/3D-FUTURE-model" "E:/diverse_synth/dump/3D-Front-texture" "./output_render_bedroom_bedfront" --mapping_file "E:/diverse_synth/front_3d_label_mapping.csv" --bedroom_index 0 --margin 1.2 --resolution 1600 --samples 128 --camera_mode bed_front

blenderproc run main.py "E:/diverse_synth/dump/3D-FRONT" "E:/diverse_synth/dump/3D-FUTURE-model" "E:/diverse_synth/dump/3D-Front-texture" "./output"

### 数据预处理 

-- 数据划分

- Compound混合偏置
    '''python compound_split.py --total_data_dir ../dump/total_data --out_root ../dump --client_prefix compound_client --subdir 3D-FRONT --ratios 0.7,0.2,0.1 --client1_room Bedroom --client1_style sparse --client2_room LivingRoom --client2_style neutral --client3_room DiningRoom --client3_style dense --bias_strength 0.8 --seed 0'''

-- 数据异质性分析
    '''python data_analysis.py --root E:\diverse_synth\dump --subdir 3D-FRONT --setting S0=IID_client1,IID_client2,IID_client3 --setting S1=bedroom,livingroom,diningroom --setting S2=sparse,neutral,dense --setting S3=quantity_client1,quantity_client2,quantity_client3 --setting S4=compound_client1,compound_client2,compound_client3'''

------

**（注：preprocess_data.py、pickle_future_data.py内的数据集文件路径以及config的文件路径需根据实际存放路径改变，即--threed_front_dataset_directory、--threed_future_dataset_directory、--model_info）**

- bedroom
    
    ```python
    python preprocess_data.py --annotation_file ../config/bedroom_threed_front_splits.csv --dataset_filtering bedroom --render
```
    
    ```python
    python pickle_future_data.py --annotation_file ../config/bedroom_threed_front_splits.csv --dataset_filtering bedroom
    ```
    
- livingroom
    ```python
    python preprocess_data.py --annotation_file ../config/livingroom_threed_front_splits.csv --dataset_filtering livingroom --render
    ```
    ```python
    python pickle_future_data.py --annotation_file ../config/library_threed_front_splits.csv --dataset_filtering library
    ```
    
- diningroom
    ```python
    python preprocess_data.py  --annotation_file ../config/diningroom_threed_front_splits.csv --dataset_filtering diningroom --render
    ```
    ```python
    python pickle_future_data.py --annotation_file ../config/diningroom_threed_front_splits.csv --dataset_filtering diningroom
    ```

## 网络训练

### 训练

**（注：train_with_wandb.py内的wandb相关配置根据自己的wandb账号进行更改，若不使用wandb直接注释相关代码即可，wandb是一个可视化训练过程损失函数的插件，直接搜索即可注册使用）**

- bedroom
    ```python
    python train_with_wandb.py --config_file ../config/bedroom_config.yaml --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P
    ```

- livingroom
    ```python 
    python train_with_wandb.py --config_file ../config/livingroom_config.yaml --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P
    ```

- diningroom
    ```python 
    python train_with_wandb.py --config_file ../config/diningroom_config.yaml --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P
    ```

- dense
    '''python train_with_wandb.py --config_file ../config/dense_config.yaml --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P'''

- neutral
    '''python train_with_wandb.py --config_file ../config/neutral_config.yaml --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P'''

- sparse
    '''python train_with_wandb.py --config_file ../config/sparse_config.yaml --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P'''

-- train_fedAVG联邦流程

- Room-based Label Shift
    '''python train_fedAVG.py --config_bedroom ../config/bedroom_config.yaml --config_diningroom ../config/diningroom_config.yaml --config_livingroom  ../config/livingroom_config.yaml --rounds 50 --local_epochs 1 --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P --wandb_entity 2745302895- --run_name fedavg_bedroom_livingroom_diningroom'''
    
- Style-based Feature Shift
    '''python train_fedAVG.py --config_dense ../config/dense_config.yaml --config_neutral ../config/neutral_config.yaml --config_sparse  ../config/sparse_config.yaml --rounds 50 --local_epochs 1 --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P --wandb_entity 2745302895- --run_name fedavg_dense_neutral_sparse'''

- IID-based CLIENT
    '''python train_fedAVG.py --config_IID_client1 ../config/IID_client1_config.yaml --config_IID_client2 ../config/IID_client2_config.yaml --config_IID_client3  ../config/IID_client3_config.yaml --rounds 50 --local_epochs 1 --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P --wandb_entity 2745302895- --run_name fedavg_IID_client'''
 
     '''python train_fedAVG.py --config_IID_client1 ../config/IID_client1_config.yaml --config_IID_client2 ../config/IID_client2_config.yaml --config_IID_client3  ../config/IID_client3_config.yaml --rounds 100 --local_epochs 1 --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P --wandb_entity 2745302895- --run_name fedavg_IID_client'''

- Quantity-based Shift
    '''python train_fedAVG.py --config_quantity_client1 ../config/quantity_client1_config.yaml --config_quantity_client2 ../config/quantity_client2_config.yaml --config_quantity_client3  ../config/quantity_client3_config.yaml --rounds 50 --local_epochs 1 --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P --wandb_entity 2745302895- --run_name fedavg_quantity_client'''
    
    '''python train_fedAVG.py --config_quantity_client1 ../config/quantity_client1_config.yaml --config_quantity_client2 ../config/quantity_client2_config.yaml --config_quantity_client3  ../config/quantity_client3_config.yaml --rounds 100 --local_epochs 1 --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P --wandb_entity 2745302895- --run_name fedavg_quantity_client'''
    
- compound-based Shift
    '''python train_fedAVG.py --config_compound_client1 ../config/compound_client1_config.yaml --config_compound_client2 ../config/compound_client2_config.yaml --config_compound_client3  ../config/compound_client3_config.yaml --rounds 50 --local_epochs 1 --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P --wandb_entity 2745302895- --run_name fedavg_compound_client'''

    '''python train_fedAVG.py --config_compound_client1 ../config/compound_client1_config.yaml --config_compound_client2 ../config/compound_client2_config.yaml --config_compound_client3  ../config/compound_client3_config.yaml --rounds 100 --local_epochs 1 --generator_type EnhancedBetaTCVAE --discriminator_type UNet3P --wandb_entity 2745302895- --run_name fedavg_compound_client'''

## 场景家具布局数据生成和后处理
### 场景家具布局生成
- bedroom
    ```python
    python generate_scene.py --output_directory ../render_scene/bedroom/ --room_type bedroom --annotation_file ../config/bedroom_threed_front_splits.csv --weight_file ../savepoint/bedroom_ebvae_h32_mss --tag fedavg_bedroom --render
    ```
- livingroom
    ```python
    python generate_scene.py --output_directory ../render_scene/bedroom/ --room_type livingroom --annotation_file ../config/livingroom_threed_front_splits.csv --weight_file ../savepoint/bedroom_ebvae_h32_mss --tag fedavg_livingroom --render
    ```
- diningroom
    ```python
    python generate_scene.py --output_directory ../render_scene/bedroom/ --room_type diningroom --annotation_file ../config/diningroom_threed_front_splits.csv --weight_file ../savepoint/bedroom_ebvae_h32_mss --tag fedavg_diningroom --render
    ```
- dense
    '''python generate_scene.py --output_directory ../render_scene/fedavg/ --room_type dense --annotation_file ../config/dense_threed_front_splits.csv --weight_file ../savepoint/dense_total --tag fedavg_dense --render'''

- neutral
    '''python generate_scene.py --output_directory ../render_scene/fedavg/ --room_type neutral --annotation_file ../config/neutral_threed_front_splits.csv --weight_file ../savepoint/dense_total --tag fedavg_neutral --render'''

- sparse
    '''python generate_scene.py --output_directory ../render_scene/fedavg/ --room_type sparse --annotation_file ../config/sparse_threed_front_splits.csv --weight_file ../savepoint/dense_total --tag fedavg_sparse --render'''

- IID_client1
    '''python generate_scene.py --output_directory ../render_scene/fedavg/ --room_type IID_client1 --annotation_file ../config/IID_client1_threed_front_splits.csv --weight_file ../savepoint/IID_client1_total --tag fedavg_IID_client1 --render'''

- compound_client1
    '''python generate_scene.py --output_directory ../render_scene/fedavg/ --room_type compound_client1 --annotation_file ../config/compound_client1_threed_front_splits.csv --weight_file ../savepoint/compound_client1_total --tag fedavg_compound_client1 --render'''

- quantity_client1
    '''python generate_scene.py --output_directory ../render_scene/fedavg/ --room_type quantity_client1 --annotation_file ../config/quantity_client1_threed_front_splits.csv --weight_file ../savepoint/quantity_client1_total --tag fedavg_quantity_client1 --render'''

### 使用matlab进行后处理（暂时可以先不进行处理）
如果像可视化后处理过程, 在 matlab上运行`test.m` 

## 指标评估

- 计算 fid 分数
- bedroom
    ```python
    python compute_fid_scores.py --dataset_type bedroom --tag fedavg_bedroom --path_to_renderings ../render_scene --path_to_annotations ../config/bedroom_threed_front_splits.csv --output_directory ../metrics/fedavg_bedroom
    ```
- IID_client
    '''python compute_fid_scores.py --tag fedavg_IID_client1 --path_to_renderings ../render_scene --path_to_annotations ../config/IID_client1_threed_front_splits.csv --output_directory ../metrics/fedavg_IID_client1'''
    
- dense FID
    '''python compute_fid_scores.py --tag fedavg_dense --path_to_renderings ../render_scene --path_to_annotations ../config/dense_threed_front_splits.csv --output_directory ../metrics/fedavg_dense'''

- neutral FID
    '''python compute_fid_scores.py --tag fedavg_neutral --path_to_renderings ../render_scene --path_to_annotations ../config/neutral_threed_front_splits.csv --output_directory ../metrics/fedavg_neutral'''

- sparse FID
    '''python compute_fid_scores.py --tag fedavg_sparse --path_to_renderings ../render_scene --path_to_annotations ../config/sparse_threed_front_splits.csv --output_directory ../metrics/fedavg_sparse'''

- quantity FID
    '''python compute_fid_scores.py --tag fedavg_quantity_client1 --path_to_renderings ../render_scene --path_to_annotations ../config/quantity_client1_threed_front_splits.csv --output_directory ../metrics/fedavg_quantity_client1'''

- compound FID
    '''python compute_fid_scores.py --tag fedavg_compound_client1 --path_to_renderings ../render_scene --path_to_annotations ../config/compound_client1_threed_front_splits.csv --output_directory ../metrics/fedavg_compound_client1'''

- 计算 iou
- bedroom
    ```python
    python compute_iou.py --tag fedavg_bedroom --path_to_renderings ../render_scene
    ```

- dense IOU
    '''python compute_iou.py --tag fedavg_dense --path_to_renderings ../render_scene'''

- neutral IOU
    '''python compute_iou.py --tag fedavg_neutral --path_to_renderings ../render_scene'''

- sparse IOU
    '''python compute_iou.py --tag fedavg_sparse --path_to_renderings ../render_scene'''

- IID_client1 IOU
    '''python compute_iou.py --tag fedavg_IID_client1 --path_to_renderings ../render_scene'''

- quantity_client1 IOU
    '''python compute_iou.py --tag fedavg_quantity_client1 --path_to_renderings ../render_scene'''

- compound_client1 IOU
    '''python compute_iou.py --tag fedavg_compound_client1 --path_to_renderings ../render_scene'''

- 计算 KL 散度
    ```python
    python valuate_kl_divergence_object_category.py --output_directory xxx --path_to_renderings xxx --dataset_type bedroom --tag xxx
    ```

KL散度没跑通。
- dense KL
    '''python preprocess_render.py --room_type dense --annotation_file ../config/dense_threed_front_splits.csv --output_directory ../render_scene/fedavg --tag raw_dense --scene_path ../dump/bedroom/--render'''

    ```
    python evaluate_kl_divergence_object_category.py --output_directory ../metrics/fedavg_dense --path_to_renderings ../render_scene/fedavg --dataset_type dense --tag fedavg_dense
    ```

- neutral KL

- sparse KL


