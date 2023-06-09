import pickle
import argparse
from datasets import XRayDataset
from tqdm import tqdm
import gzip
from utils import read_json
import os

def main(config, target):
    if not os.path.exists(target):
        os.makedirs(os.path.join(target, "train"))
        os.makedirs(os.path.join(target, "valid"))

    train_path = os.path.join(target, "train")
    valid_path = os.path.join(target, "valid")
    train_dataset = XRayDataset(
        config
    )
    for i in tqdm(range(len(train_dataset))):
        g = train_dataset[i]
        with gzip.open(os.path.join(train_path, f"{i}.pkl"), mode='wb') as f:
            pickle.dump(g, f)   

    valid_dataset = XRayDataset(
        config, is_train=False
    )

    for i in tqdm(range(len(valid_dataset))):
        g = valid_dataset[i]
        with gzip.open(os.path.join(valid_path, f"{i}.pkl"), mode='wb') as f:
            pickle.dump(g, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='./config.json', type=str,
                    help='config file path (default: None)')
    parser.add_argument('-t', '--target', default='/opt/ml/cache_data', type=str,
                    help='cache dataset root directory')    
    args = parser.parse_args()
    config = read_json(args.config)
    main(config, args.target)
    