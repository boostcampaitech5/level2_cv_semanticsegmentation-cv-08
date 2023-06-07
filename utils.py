from pathlib import Path
import json
import os
import torch

def read_json(fname):
    fname = Path(fname)
    with fname.open(encoding='utf-8') as handle:
        return json.load(handle)
