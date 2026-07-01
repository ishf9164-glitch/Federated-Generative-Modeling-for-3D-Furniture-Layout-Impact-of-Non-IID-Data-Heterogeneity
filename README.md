## Datasets

Based on [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)  and 3D-Future darasets.

---

## Table of Contents

- [Environment](#environment)
- [Datasets](#datasets)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Federated Training](#federated-training)
- [Scene Generation](#scene-generation)
- [Post-processing](#post-processing)
- [Evaluation](#evaluation)
- [Notes](#notes)

---

## Environment

### Dependencies

The project depends on the following packages:

- `numpy`
- `cython`
- `pillow`
- `pyyaml`
- `pyrr`
- `torch`
- `torchvision`
- `trimesh`
- `tqdm`
- `matlab`
- `cleanfid`
- `scipy`
- `wandb`
- `simple-3dviz`

Example installation:

```bash
pip install numpy cython pillow pyyaml pyrr trimesh tqdm cleanfid scipy wandb simple-3dviz
```

Install PyTorch and torchvision according to your CUDA version:

```bash
pip install torch torchvision
```

---

## Datasets

This project uses:

- [3D-FRONT Dataset](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)
- 3D-FUTURE Dataset

Before running the project, configure the dataset paths in the preprocessing scripts and configuration files.

Expected directory structure example:

```text
project_root/
├── config/
├── dump/
│   ├── 3D-FRONT/
│   ├── 3D-FUTURE-model/
│   └── 3D-FRONT-texture/
├── render_scene/
├── savepoint/
├── metrics/
└── ...
```

---

## Dataset Visualization

### Render a Single Bedroom Layout

```bash
blenderproc run front3d_single_bedroom_render.py \
  "E:/diverse_synth/dump/3D-FRONT/0af309fd-4a34-4aa2-bbe9-610edadf747e.json" \
  "E:/diverse_synth/dump/3D-FUTURE-model" \
  "./output_single_bedroom_paperview" \
  --bedroom_index 0 \
  --resolution 1600 \
  --samples 128 \
  --camera_mode paper_view \
  --margin 0.8
```

### Render Bedroom Front View

```bash
blenderproc run front3d_bedroom_render.py \
  "E:/diverse_synth/dump/3D-FRONT/0a8d471a-2587-458a-9214-586e003e9cf9.json" \
  "E:/diverse_synth/dump/3D-FUTURE-model" \
  "E:/diverse_synth/dump/3D-FRONT-texture" \
  "./output_render_bedroom_bedfront" \
  --mapping_file "E:/diverse_synth/front_3d_label_mapping.csv" \
  --bedroom_index 0 \
  --margin 1.2 \
  --resolution 1600 \
  --samples 128 \
  --camera_mode bed_front
```

### Batch Rendering

```bash
blenderproc run main.py \
  "E:/diverse_synth/dump/3D-FRONT" \
  "E:/diverse_synth/dump/3D-FUTURE-model" \
  "E:/diverse_synth/dump/3D-FRONT-texture" \
  "./output"
```

---

## Data Preprocessing

> Before preprocessing, update the dataset paths in `preprocess_data.py`, `pickle_future_data.py`, and the configuration files as needed.
>
> Key parameters include:
>
> - `--threed_front_dataset_directory`
> - `--threed_future_dataset_directory`
> - `--model_info`

### Dataset Splitting

#### Compound Bias Split

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

### Data Heterogeneity Analysis

```bash
python data_analysis.py \
  --root E:\diverse_synth\dump \
  --subdir 3D-FRONT \
  --setting S0=IID_client1,IID_client2,IID_client3 \
  --setting S1=bedroom,livingroom,diningroom \
  --setting S2=sparse,neutral,dense \
  --setting S3=quantity_client1,quantity_client2,quantity_client3 \
  --setting S4=compound_client1,compound_client2,compound_client3
```

### Bedroom

```bash
python preprocess_data.py \
  --annotation_file ../config/bedroom_threed_front_splits.csv \
  --dataset_filtering bedroom \
  --render
```

```bash
python pickle_future_data.py \
  --annotation_file ../config/bedroom_threed_front_splits.csv \
  --dataset_filtering bedroom
```

### Living Room

```bash
python preprocess_data.py \
  --annotation_file ../config/livingroom_threed_front_splits.csv \
  --dataset_filtering livingroom \
  --render
```

```bash
python pickle_future_data.py \
  --annotation_file ../config/livingroom_threed_front_splits.csv \
  --dataset_filtering livingroom
```

### Dining Room

```bash
python preprocess_data.py \
  --annotation_file ../config/diningroom_threed_front_splits.csv \
  --dataset_filtering diningroom \
  --render
```

```bash
python pickle_future_data.py \
  --annotation_file ../config/diningroom_threed_front_splits.csv \
  --dataset_filtering diningroom
```

---

## Model Training

> `train_with_wandb.py` contains Weights & Biases configuration.
>
> Replace the W&B account-related configuration with your own account information. If W&B is not required, comment out the related code.

### Bedroom

```bash
python train_with_wandb.py \
  --config_file ../config/bedroom_config.yaml \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P
```

### Living Room

```bash
python train_with_wandb.py \
  --config_file ../config/livingroom_config.yaml \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P
```

### Dining Room

```bash
python train_with_wandb.py \
  --config_file ../config/diningroom_config.yaml \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P
```

### Dense Style

```bash
python train_with_wandb.py \
  --config_file ../config/dense_config.yaml \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P
```

### Neutral Style

```bash
python train_with_wandb.py \
  --config_file ../config/neutral_config.yaml \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P
```

### Sparse Style

```bash
python train_with_wandb.py \
  --config_file ../config/sparse_config.yaml \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P
```

---

## Federated Training

The project supports FedAvg-based collaborative training under different data heterogeneity settings.

Replace `YOUR_WANDB_ENTITY` with your own W&B entity name.

### Room-Based Label Shift

```bash
python train_fedAVG.py \
  --config_bedroom ../config/bedroom_config.yaml \
  --config_diningroom ../config/diningroom_config.yaml \
  --config_livingroom ../config/livingroom_config.yaml \
  --rounds 50 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --wandb_entity YOUR_WANDB_ENTITY \
  --run_name fedavg_bedroom_livingroom_diningroom
```

### Style-Based Feature Shift

```bash
python train_fedAVG.py \
  --config_dense ../config/dense_config.yaml \
  --config_neutral ../config/neutral_config.yaml \
  --config_sparse ../config/sparse_config.yaml \
  --rounds 50 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --wandb_entity YOUR_WANDB_ENTITY \
  --run_name fedavg_dense_neutral_sparse
```

### IID Clients

```bash
python train_fedAVG.py \
  --config_IID_client1 ../config/IID_client1_config.yaml \
  --config_IID_client2 ../config/IID_client2_config.yaml \
  --config_IID_client3 ../config/IID_client3_config.yaml \
  --rounds 50 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --wandb_entity YOUR_WANDB_ENTITY \
  --run_name fedavg_IID_client
```

```bash
python train_fedAVG.py \
  --config_IID_client1 ../config/IID_client1_config.yaml \
  --config_IID_client2 ../config/IID_client2_config.yaml \
  --config_IID_client3 ../config/IID_client3_config.yaml \
  --rounds 100 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --wandb_entity YOUR_WANDB_ENTITY \
  --run_name fedavg_IID_client
```

### Quantity-Based Shift

```bash
python train_fedAVG.py \
  --config_quantity_client1 ../config/quantity_client1_config.yaml \
  --config_quantity_client2 ../config/quantity_client2_config.yaml \
  --config_quantity_client3 ../config/quantity_client3_config.yaml \
  --rounds 50 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --wandb_entity YOUR_WANDB_ENTITY \
  --run_name fedavg_quantity_client
```

```bash
python train_fedAVG.py \
  --config_quantity_client1 ../config/quantity_client1_config.yaml \
  --config_quantity_client2 ../config/quantity_client2_config.yaml \
  --config_quantity_client3 ../config/quantity_client3_config.yaml \
  --rounds 100 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --wandb_entity YOUR_WANDB_ENTITY \
  --run_name fedavg_quantity_client
```

### Compound-Based Shift

```bash
python train_fedAVG.py \
  --config_compound_client1 ../config/compound_client1_config.yaml \
  --config_compound_client2 ../config/compound_client2_config.yaml \
  --config_compound_client3 ../config/compound_client3_config.yaml \
  --rounds 50 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --wandb_entity YOUR_WANDB_ENTITY \
  --run_name fedavg_compound_client
```

```bash
python train_fedAVG.py \
  --config_compound_client1 ../config/compound_client1_config.yaml \
  --config_compound_client2 ../config/compound_client2_config.yaml \
  --config_compound_client3 ../config/compound_client3_config.yaml \
  --rounds 100 \
  --local_epochs 1 \
  --generator_type EnhancedBetaTCVAE \
  --discriminator_type UNet3P \
  --wandb_entity YOUR_WANDB_ENTITY \
  --run_name fedavg_compound_client
```

---

## Scene Generation

Replace the checkpoint path in `--weight_file` with the trained model corresponding to the current experiment.

### Bedroom

```bash
python generate_scene.py \
  --output_directory ../render_scene/bedroom/ \
  --room_type bedroom \
  --annotation_file ../config/bedroom_threed_front_splits.csv \
  --weight_file ../savepoint/bedroom_ebvae_h32_mss \
  --tag fedavg_bedroom \
  --render
```

### Living Room

```bash
python generate_scene.py \
  --output_directory ../render_scene/livingroom/ \
  --room_type livingroom \
  --annotation_file ../config/livingroom_threed_front_splits.csv \
  --weight_file ../savepoint/livingroom_ebvae_h32_mss \
  --tag fedavg_livingroom \
  --render
```

### Dining Room

```bash
python generate_scene.py \
  --output_directory ../render_scene/diningroom/ \
  --room_type diningroom \
  --annotation_file ../config/diningroom_threed_front_splits.csv \
  --weight_file ../savepoint/diningroom_ebvae_h32_mss \
  --tag fedavg_diningroom \
  --render
```

### Dense Style

```bash
python generate_scene.py \
  --output_directory ../render_scene/fedavg/ \
  --room_type dense \
  --annotation_file ../config/dense_threed_front_splits.csv \
  --weight_file ../savepoint/dense_total \
  --tag fedavg_dense \
  --render
```

### Neutral Style

```bash
python generate_scene.py \
  --output_directory ../render_scene/fedavg/ \
  --room_type neutral \
  --annotation_file ../config/neutral_threed_front_splits.csv \
  --weight_file ../savepoint/neutral_total \
  --tag fedavg_neutral \
  --render
```

### Sparse Style

```bash
python generate_scene.py \
  --output_directory ../render_scene/fedavg/ \
  --room_type sparse \
  --annotation_file ../config/sparse_threed_front_splits.csv \
  --weight_file ../savepoint/sparse_total \
  --tag fedavg_sparse \
  --render
```

### IID Client 1

```bash
python generate_scene.py \
  --output_directory ../render_scene/fedavg/ \
  --room_type IID_client1 \
  --annotation_file ../config/IID_client1_threed_front_splits.csv \
  --weight_file ../savepoint/IID_client1_total \
  --tag fedavg_IID_client1 \
  --render
```

### Compound Client 1

```bash
python generate_scene.py \
  --output_directory ../render_scene/fedavg/ \
  --room_type compound_client1 \
  --annotation_file ../config/compound_client1_threed_front_splits.csv \
  --weight_file ../savepoint/compound_client1_total \
  --tag fedavg_compound_client1 \
  --render
```

### Quantity Client 1

```bash
python generate_scene.py \
  --output_directory ../render_scene/fedavg/ \
  --room_type quantity_client1 \
  --annotation_file ../config/quantity_client1_threed_front_splits.csv \
  --weight_file ../savepoint/quantity_client1_total \
  --tag fedavg_quantity_client1 \
  --render
```

---

## Post-processing

MATLAB-based post-processing is optional.

To visualize the post-processing procedure, run:

```matlab
test.m
```

---

## Evaluation

### FID Score

#### Bedroom

```bash
python compute_fid_scores.py \
  --dataset_type bedroom \
  --tag fedavg_bedroom \
  --path_to_renderings ../render_scene \
  --path_to_annotations ../config/bedroom_threed_front_splits.csv \
  --output_directory ../metrics/fedavg_bedroom
```

#### IID Client

```bash
python compute_fid_scores.py \
  --tag fedavg_IID_client1 \
  --path_to_renderings ../render_scene \
  --path_to_annotations ../config/IID_client1_threed_front_splits.csv \
  --output_directory ../metrics/fedavg_IID_client1
```

#### Dense Style

```bash
python compute_fid_scores.py \
  --tag fedavg_dense \
  --path_to_renderings ../render_scene \
  --path_to_annotations ../config/dense_threed_front_splits.csv \
  --output_directory ../metrics/fedavg_dense
```

#### Neutral Style

```bash
python compute_fid_scores.py \
  --tag fedavg_neutral \
  --path_to_renderings ../render_scene \
  --path_to_annotations ../config/neutral_threed_front_splits.csv \
  --output_directory ../metrics/fedavg_neutral
```

#### Sparse Style

```bash
python compute_fid_scores.py \
  --tag fedavg_sparse \
  --path_to_renderings ../render_scene \
  --path_to_annotations ../config/sparse_threed_front_splits.csv \
  --output_directory ../metrics/fedavg_sparse
```

#### Quantity Client

```bash
python compute_fid_scores.py \
  --tag fedavg_quantity_client1 \
  --path_to_renderings ../render_scene \
  --path_to_annotations ../config/quantity_client1_threed_front_splits.csv \
  --output_directory ../metrics/fedavg_quantity_client1
```

#### Compound Client

```bash
python compute_fid_scores.py \
  --tag fedavg_compound_client1 \
  --path_to_renderings ../render_scene \
  --path_to_annotations ../config/compound_client1_threed_front_splits.csv \
  --output_directory ../metrics/fedavg_compound_client1
```

### IoU Score

#### Bedroom

```bash
python compute_iou.py \
  --tag fedavg_bedroom \
  --path_to_renderings ../render_scene
```

#### Dense Style

```bash
python compute_iou.py \
  --tag fedavg_dense \
  --path_to_renderings ../render_scene
```

#### Neutral Style

```bash
python compute_iou.py \
  --tag fedavg_neutral \
  --path_to_renderings ../render_scene
```

#### Sparse Style

```bash
python compute_iou.py \
  --tag fedavg_sparse \
  --path_to_renderings ../render_scene
```

#### IID Client 1

```bash
python compute_iou.py \
  --tag fedavg_IID_client1 \
  --path_to_renderings ../render_scene
```

#### Quantity Client 1

```bash
python compute_iou.py \
  --tag fedavg_quantity_client1 \
  --path_to_renderings ../render_scene
```

#### Compound Client 1

```bash
python compute_iou.py \
  --tag fedavg_compound_client1 \
  --path_to_renderings ../render_scene
```

### KL Divergence

General command:

```bash
python evaluate_kl_divergence_object_category.py \
  --output_directory xxx \
  --path_to_renderings xxx \
  --dataset_type bedroom \
  --tag xxx
```

#### Dense Style Example

```bash
python preprocess_render.py \
  --room_type dense \
  --annotation_file ../config/dense_threed_front_splits.csv \
  --output_directory ../render_scene/fedavg \
  --tag raw_dense \
  --scene_path ../dump/bedroom/ \
  --render
```

```bash
python evaluate_kl_divergence_object_category.py \
  --output_directory ../metrics/fedavg_dense \
  --path_to_renderings ../render_scene/fedavg \
  --dataset_type dense \
  --tag fedavg_dense
```

> The KL-divergence evaluation pipeline may require additional adjustment depending on the rendered output format and dataset configuration.

---

## Notes

- All dataset paths, checkpoint paths, and output paths should be modified according to the local environment.
- Commands assume execution from the corresponding script directory. Adjust relative paths when running from the project root.
- Checkpoints generated by centralized training and federated training may use different naming conventions.
- W&B is optional. Remove or comment out W&B-related code when experiment tracking is not needed.
