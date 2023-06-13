import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
IMAGE_ROOT = "/opt/ml/data/train/DCM"
METADATA_PATH = "/opt/ml/data/meta_data.xlsx"


def get_img_paths():
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    return list(pngs)


def get_groups(img_paths):
    groups = [os.path.dirname(fname) for fname in img_paths]
    return groups


def get_train_meta():
    meta_df = pd.read_excel(METADATA_PATH)  # contains all (train+test) metadata

    train_ids = [int(x[2:]) for x in os.listdir(IMAGE_ROOT)]
    meta_df = meta_df.loc[meta_df["ID"].isin(train_ids)]  # train metadata only

    return meta_df


def get_bmi(x):
    weight = x["체중(몸무게)"]
    height = x["키(신장)"]
    bmi = weight / (height * height / 10000)

    return round(bmi, 1)


def describe_bmi(bmi):
    if bmi <= 18.5:
        desc = "underweight"
    elif 18.5 < bmi <= 23:
        desc = "normal"
    elif 23 < bmi <= 25:
        desc = "overweight"
    else:
        desc = "obese"

    return desc


def get_class_bmi(meta_df, img_paths):
    meta_df["bmi"] = meta_df.apply(lambda x: get_bmi(x), axis=1)
    meta_df["bmi_desc"] = meta_df.apply(lambda x: describe_bmi(x["bmi"]), axis=1)

    y = [
        meta_df[meta_df["ID"] == int(os.path.dirname(fname).split("ID")[-1])]["bmi_desc"].values[0]
        for fname in img_paths
    ]

    return y


def split_data(args):
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, random_state=SEED, shuffle=True)

    img_paths = get_img_paths()
    groups = get_groups(img_paths)
    meta_df = get_train_meta()
    get_class = globals()[f"get_class_{args.class_by}"]
    y = get_class(meta_df, img_paths)

    train_data = defaultdict(list)
    valid_data = defaultdict(list)

    for train_idx, valid_idx in sgkf.split(img_paths, y, groups):
        for tidx in train_idx:
            fname = img_paths[tidx]
            train_data[fname.split("/")[0]].append(fname.split(".")[0])

        for fidx in valid_idx:
            fname = img_paths[fidx]
            valid_data[fname.split("/")[0]].append(fname.split(".")[0])

        break  # use Fold 0 as train/valid split

    with open(args.train_json_path, "w") as f:
        json.dump(train_data, f)

    with open(args.valid_json_path, "w") as f:
        json.dump(valid_data, f)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds.")
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="/opt/ml/data/train_split.json",
        help="Path to save json file containing train filenames.",
    )
    parser.add_argument(
        "--valid_json_path",
        type=str,
        default="/opt/ml/data/valid_split.json",
        help="Path to save json file containing validation filenames.",
    )
    parser.add_argument(
        "--class_by",
        type=str,
        default="bmi",
        choices=["bmi"],
        help="Class label for stratified data split.",
    )

    args = parser.parse_args()

    split_data(args)
