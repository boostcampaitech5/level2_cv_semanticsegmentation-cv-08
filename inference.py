# python native
import argparse
import os

# external library
import pandas as pd
import segmentation_models_pytorch as smp

# torch
import torch
from torch.utils.data import DataLoader

import augmentations

# utils
import models
from datasets import XRayInferenceDataset
from runner import test
from utils import CLASSES, read_json


def main(config):
    # Load Model
    if config.base.use == "smp":
        model = getattr(smp, config.base.smp.model)(
            encoder_name=config.base.smp.encoder_name,
            encoder_weights=config.base.smp.encoder_weights,
            in_channels=3,
            classes=len(CLASSES),
        )
    else:
        model = getattr(models, config.base.pytorch.model)(len(CLASSES))

    if config.resume_from:
        print(f"Load {config.resume_from}")
        if os.path.splitext(config.resume_from)[1] == ".pt":
            model = torch.load(config.resume_from)
    else:
        model.load_state_dict(
            torch.load(os.path.join(config.save_model_dir, config.model_file_name))
        )

    # Augmentation
    tf = getattr(augmentations, config.test.augmentations.name)(
        **config.test.augmentations.parameters
    )

    # Dataset
    test_dataset = XRayInferenceDataset(config, transforms=tf)

    # Dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
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

    print(df.head(10))
    print(f"save csv directory : {os.path.join(config.save_model_dir, 'output.csv')}")
    df.to_csv(os.path.join(config.save_model_dir, "output.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", default="./config.json", type=str, help="config file path (default: None)"
    )

    args = parser.parse_args()

    config = read_json(args.config)

    main(config)
