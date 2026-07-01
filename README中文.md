# Federated Generative Modeling for 3D Furniture Layout

This repository implements a federated generative framework for 3D indoor furniture layout generation under non-IID client data. The framework studies how room-type label skew, style-based feature skew, quantity skew, and compound skew affect federated optimization, convergence, and layout quality.

The model combines an Enhanced β-TCVAE generator with a UNet3+-based relational constraint module. Federated training is implemented with FedAvg.

## Requirements

```text
numpy
cython
pillow
pyyaml
pyrr
torch
torchvision
trimesh
tqdm
scipy
cleanfid
wandb
simple-3dviz
matlab
```

## Datasets

The project uses:

* [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset): structured indoor scenes and furniture layouts
* 3D-FUTURE: furniture assets associated with 3D-FRONT

Before running the code, update dataset-related paths in:

* `preprocess_data.py`
* `pickle_future_data.py`
* configuration files under `config/`

Required path arguments include:

```text
--threed_front_dataset_directory
--threed_future_dataset_directory
--model_info
```

## Data Preprocessing

### Room-Type Splits

```bash
python preprocess_data.py \
  --annotation_file ../config/bedroom_threed_front_splits.csv \
  --dataset_filtering bedroom \
  --render

python pickle_future_data.py \
  --annotation_file ../config/bedroom_threed_front_splits.csv \
  --dataset_filtering bedroom
```

Replace `bedroom` and the corresponding annotation file with `livingroom` or `diningroom` when preparing other room categories.

### Compound Non-IID Partition

```bash
python compound_split.py \
  --total_data_dir ../dump/total_data \
  --out_root ../dump \
  --client_prefix compound_client \
  --subdir 3D-FRONT \
  --ratios 0.7,0.2,0.1 \
  --client1_room Bedroom \
  --client1_style sparse \
  --client2_room LivingRoom \
  --client2_style neutral \
  --client3_room DiningRoom \
  --client3_style dense \
  --bias_strength 0.8 \
  --seed 0
```

### Heterogeneity Analysis

```bash
python data_analysis.py \
  --root ../dump \
  --subdir 3D-FRONT \
  --setting S0=IID_client1,IID_client2,IID_client3 \
  --setting S1=bedroom,livingroom,diningroom \
  --setting S2=sparse,neutral,dense \
  --setting S3=quantity_client1,quantity_client2,quantity_client3 \
  --setting S4=compound_client1,compound_client2,compound_client3
```

## Centralized Training

```bash
python train_with_wandb.py \
  --config_file ../config/bedroom_config.yaml \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P
```

Use the corresponding configuration file for other room types or style partitions.

## Federated Training

### IID Baseline

```bash
python train_fedAVG.py \
  --config_IID_client1 ../config/IID_client1_config.yaml \
  --config_IID_client2 ../config/IID_client2_config.yaml \
  --config_IID_client3 ../config/IID_client3_config.yaml \
  --rounds 50 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --run_name fedavg_iid
```

### Room-Type Label Skew

```bash
python train_fedAVG.py \
  --config_bedroom ../config/bedroom_config.yaml \
  --config_diningroom ../config/diningroom_config.yaml \
  --config_livingroom ../config/livingroom_config.yaml \
  --rounds 50 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --run_name fedavg_label_skew
```

### Style-Based Feature Skew

```bash
python train_fedAVG.py \
  --config_dense ../config/dense_config.yaml \
  --config_neutral ../config/neutral_config.yaml \
  --config_sparse ../config/sparse_config.yaml \
  --rounds 50 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --run_name fedavg_style_skew
```

### Quantity Skew

```bash
python train_fedAVG.py \
  --config_quantity_client1 ../config/quantity_client1_config.yaml \
  --config_quantity_client2 ../config/quantity_client2_config.yaml \
  --config_quantity_client3 ../config/quantity_client3_config.yaml \
  --rounds 50 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --run_name fedavg_quantity_skew
```

### Compound Skew

```bash
python train_fedAVG.py \
  --config_compound_client1 ../config/compound_client1_config.yaml \
  --config_compound_client2 ../config/compound_client2_config.yaml \
  --config_compound_client3 ../config/compound_client3_config.yaml \
  --rounds 50 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --run_name fedavg_compound_skew
```

## Layout Generation

```bash
python generate_scene.py \
  --output_directory ../render_scene/fedavg \
  --room_type bedroom \
  --annotation_file ../config/bedroom_threed_front_splits.csv \
  --weight_file ../savepoint/model_checkpoint \
  --tag fedavg_bedroom \
  --render
```

Modify `room_type`, `annotation_file`, `weight_file`, and `tag` according to the evaluated setting.

## Evaluation

### FID

```bash
python compute_fid_scores.py \
  --dataset_type bedroom \
  --tag fedavg_bedroom \
  --path_to_renderings ../render_scene \
  --path_to_annotations ../config/bedroom_threed_front_splits.csv \
  --output_directory ../metrics/fedavg_bedroom
```

### IoU

```bash
python compute_iou.py \
  --tag fedavg_bedroom \
  --path_to_renderings ../render_scene
```

## Notes

* Raw 3D-FRONT and 3D-FUTURE data are not included in this repository.
* Weights & Biases is optional. Remove or disable related logging code when not required.
* MATLAB-based post-processing is optional and is not required for core training or evaluation.
