import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import numpy as np
import torch.utils.data.dataset
from PIL import Image
from cleanfid import fid


class ThreedFrontRenderDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, path, dataset_type='fedavg'):
        self.path = f"{path}/{dataset_type}/furniture_only"
        self.images = sorted([
            f"{self.path}/{f}"
            for f in os.listdir(self.path)
            if any(f.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return Image.open(self.images[idx])


class ImageFolderDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, path, dataset_type='bedroom', tag=None):
        self.path = f"{path}/{dataset_type}/{tag}/furniture_only"
        self.images = sorted([
            f"{self.path}/{f}"
            for f in os.listdir(self.path)
            if any(f.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images")
    )
    parser.add_argument(
        "--dataset_type",
        default="fedavg",
        choices=[
            "bedroom",
            "livingroom",
            "diningroom",
            "library",
            "dense",
            "sparse",
            "neutral",
            "IID_client1",
            "compound_client1",
            "quantity_client1"
        ],
        help="The type of dataset filtering to be used"
    )

    parser.add_argument(
        "--tag",
        default="bedroom_ebvae_h32_mss",
    )
    parser.add_argument(
        "--path_to_renderings",
        default="../render_scene",
        help="Path to the folder containing the synthesized"
    )
    parser.add_argument(
        "--path_to_annotations",
        default="../config/bedroom_threed_front_splits.csv",
        help="Path to the folder containing the annotations"
    )
    
    parser.add_argument(
        "--output_directory",
        default="/data/compute_fid/",
        help="Path to the folder containing the annotations"
    )
    parser.add_argument(
        "--path_to_save",
        default="../render_scene/fedavg",
        help="Path to the save"
    )

    args = parser.parse_args(argv)

    print("Generating temporary a folder with test_real images...")
    path_to_test_real = f"{args.path_to_save}/{args.tag}/test_real/"
    if not os.path.exists(path_to_test_real):
        os.makedirs(path_to_test_real)
    dataset_images = [
        f"{args.path_to_renderings}/{args.dataset_type}/{args.tag}/furniture_only/{f}"
        for f in os.listdir(f"{args.path_to_renderings}/{args.dataset_type}/{args.tag}/furniture_only")
        if any(f.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
    ]

    print("Generating temporary a folder with test_fake images...")
    path_to_test_fake = f"{args.path_to_save}/{args.tag}/test_fake/"
    if not os.path.exists(path_to_test_fake):
        os.makedirs(path_to_test_fake)

    synthesized_images = [
            f"{args.path_to_renderings}/{args.dataset_type}/{args.tag}/furniture_only/{f}"
            for f in os.listdir(f"{args.path_to_renderings}/{args.dataset_type}/{args.tag}/furniture_only")
            if any(f.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
    ]

    for i, fi in enumerate(synthesized_images):
        shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))

    fake_length = len(synthesized_images)

    scores = []
    for _ in range(10):
        np.random.shuffle(synthesized_images)
        dataset_subset = np.random.choice(dataset_images, fake_length)
        for i, fi in enumerate(dataset_subset):
            shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_real, i))

        # Compute the FID score
        # 💡 核心修复：明确指定 device='cpu' 以绕过 CUDA 编译检查
        fid_score = fid.compute_fid(
            path_to_test_real,
            path_to_test_fake,
            device='cpu', # <--- 修改这里
            num_workers = 0
        )
        scores.append(fid_score)
        print(fid_score)
    print(sum(scores) / len(scores))
    print(np.std(scores))


if __name__ == "__main__":
    main(sys.argv[1:])
