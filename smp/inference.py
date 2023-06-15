# python native
import os
from argparse import ArgumentParser
from glob import glob

import albumentations as A

# external library
import pandas as pd
import segmentation_models_pytorch as smp

# torch
import torch
import yaml
from torch.utils.data import DataLoader

# utils
import models
from datasets.test_dataset import XRayInferenceDataset
from runner.test_runner import test
from utils.util import CLASSES, AttributeDict, check_directory, set_seed


def main(args):
    # Model Define
    if args.model.use == "smp":
        model = getattr(smp, args.model.smp.architectures)(
            encoder_name=args.encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=args.encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=len(CLASSES),  # model output channels (number of classes in your dataset)
        )
    else:
        model = getattr(models, args.model.pytorch.architectures)(len(CLASSES))

    # Load Model
    model.load_state_dict(torch.load(os.path.join(args.save_model_dir, args.save_model_fname)))

    # Load Data
    pngs = glob(os.path.join(args.test_image_dir, "*", "*.png"))

    # Augmentation
    test_tf = A.Compose(
        [getattr(A, aug["type"])(**aug["parameters"]) for aug in args.test.augmentation]
    )
    # Dataset
    test_dataset = XRayInferenceDataset(args, pngs, transforms=test_tf)

    # Dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test.batch_size,
        shuffle=False,
        num_workers=args.test.num_workers,
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
    print(f"save csv directory : {os.path.join(args.save_submit_dir, args.save_submit_fname)}")
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

    # check save model dir
    args.inference = True
    args = check_directory(args)

    main(args)
