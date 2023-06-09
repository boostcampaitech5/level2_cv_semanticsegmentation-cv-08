from pathlib import Path
import json
import os
import torch
import numpy as np
import random

def read_json(fname):
    fname = Path(fname)
    with fname.open(encoding='utf-8') as handle:
        return json.load(handle)

def set_seed(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)