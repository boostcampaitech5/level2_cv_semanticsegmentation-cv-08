# python native
import argparse
import os

# external library
import pandas as pd

# torch
import torch
from torch.utils.data import DataLoader

import augmentations

# utils
from datasets import XRayInferenceDataset
from runner import test
from utils import read_json


def main(config):
    # Load Model
    model = torch.load(os.path.normpath(config.inference_model_dir))

    # Augmentation
    tf = getattr(augmentations, config.test_augmentations.name)(**config.test_augmentations.parameters)

    # Dataset
    test_dataset = XRayInferenceDataset(config, transforms=tf)

    # Dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.test_num_workers,
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
