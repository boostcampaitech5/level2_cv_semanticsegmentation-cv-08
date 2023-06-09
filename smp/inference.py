# python native
import os
from argparse import ArgumentParser
from glob import glob

# external library
import albumentations as A
import pandas as pd

# torch
import torch
import yaml

# utils
from datasets.test_dataset import XRayInferenceDataset
from runner.test_runner import test
from torch.utils.data import DataLoader
from utils.util import AttributeDict, set_seed


def main(args):
    # Load Model
    model = torch.load(os.path.join(args.save_model_dir, args.save_model_fname))

    # Load Data
    pngs = glob(os.path.join(args.test_image_dir, "*", "*.png"))

    # Augmentation
    tf = A.Resize(512, 512)

    # Dataset
    test_dataset = XRayInferenceDataset(args, pngs, transforms=tf)

    # Dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    rles, filename_and_class = test(args, model, test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])

    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )

    print(df.head(10))

    df.to_csv(os.path.join(args.save_submit_dir, args.save_submit_fname), index=False)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        default="./config.yaml",
        type=str,
        help="config file path (default: ./config.yaml)",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return AttributeDict(config)


if __name__ == "__main__":
    args = parse_args()

    # set seed
    set_seed(args.seed)

    # check save_dir
    assert os.path.isdir(args.save_model_dir), "please check save dir"

    main(args)
