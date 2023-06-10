# python native
import argparse
import os

# external library
import pandas as pd

# torch
import torch
from torch.utils.data import DataLoader

from augmentations import augmentation

# utils
from datasets.test_dataset import XRayInferenceDataset
from runner.test_runner import test
from utils import read_json


def main(config):
    # Load Model
    model = torch.load(os.path.normpath(config.inference_model_dir))

    # Augmentation
    tf = getattr(augmentation, "base_augmentation")(config.input_size, mean=0.13189, std=0.17733)

    # Dataset
    test_dataset = XRayInferenceDataset(config, transforms=tf)

    # Dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
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
