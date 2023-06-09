import pickle
import argparse
from dataset import XRayDataset
from parse_config import ConfigParser
from tqdm import tqdm
import gzip

args = argparse.ArgumentParser()
args.add_argument('-c', '--config', default='./config.json')
config = ConfigParser.from_args(args)

train_dataset = XRayDataset(
    config
)

for i in tqdm(range(len(train_dataset))):
    g = train_dataset[i]
    with gzip.open(f'/opt/ml/input/code/train_cache/{i}.pkl', mode='wb') as f:
        pickle.dump(g, f)   

valid_dataset = XRayDataset(
    config, is_train=False
)

for i in tqdm(range(len(valid_dataset))):
    g = valid_dataset[i]
    with gzip.open(f'/opt/ml/input/code/valid_cache/{i}.pkl', mode='wb') as f:
        pickle.dump(g, f)   