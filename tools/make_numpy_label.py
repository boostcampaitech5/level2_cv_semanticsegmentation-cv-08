import argparse
import gzip
import json
import os
import pickle
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append("..")
import augmentations
from datasets import XRayDataset
from utils import read_json


def main(args):
    target = args.target
    np_mode = args.mode
    
    if np_mode not in ["npz", "npy"]:
        raise ValueError(f'{np_mode} is not supported. Only "npz", "npy" can be used.')
    LABEL_ROOT = os.path.join(target, "train/outputs_json")
    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
    
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
        for root, _dirs, files in os.walk(LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
    jsons = sorted(jsons)
    
    labelnames = np.array(jsons)
    
    print("Converting json label into numpy mask array. This may take a while...")
    
    for label_name in tqdm(labelnames):
        label_path = os.path.join(LABEL_ROOT, label_name)

        label_shape = (2048, 2048) + (len(CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)

        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for ann in annotations:
            c = ann["label"]
            # CLASS2IND
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros((2048, 2048), dtype=np.uint8)

            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
            
        changed_label = label_name.split('.')
        changed_label[-1] = "npy"
        changed_label = ".".join(changed_label)
        
        label = np.packbits(label.astype(bool), axis=None)
        
        with open(os.path.join(LABEL_ROOT, changed_label), 'wb') as f:
            if np_mode == "npy":
                np.save(f, label)
            elif np_mode == "npz":
                np.savez_compressed(f, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        type=str,
        help="target dataset root directory",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="npz",
        help='numpy conversion mode. "npy" to save as uncompressed numpy array, "npz" to save as compressed numpy array(default: "npz")'
    )
    args = parser.parse_args()
    main(args)
