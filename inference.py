# python native
import argparse
import os
from glob import glob

# external library
import albumentations as A
import pandas as pd

# torch
import torch
from torch.utils.data import DataLoader

# utils
from datasets.test_dataset import XRayInferenceDataset
from runner.test_runner import test
from utils import read_json


def main(config):
    # Load Model
    model = torch.load(os.path.join(config.save_model_dir, config.save_model_fname))

    # Load Data
    pngs = glob(os.path.join(config.test_image_dir, "*", "*.png"))

    # Augmentation
    tf = A.Resize(config.input_size, config.input_size)

    # Dataset
    test_dataset = XRayInferenceDataset(config, pngs, transforms=tf)

    # Dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    rles, filename_and_class = test(config, model, test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])

    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )

    df.to_csv(os.path.join(config.model_dir, "output.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="./config.json", type=str, help="config file path (default: None)"
    )
    args = parser.parse_args()
    config = read_json(args.config)
    main(config)
