import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
# from logger import setup_logging
from utils import read_json

class ConfigParser:
    def __init__(self, args):
       self._config = read_json(args.config)
       
    @classmethod 
    def from_args(cls, args):
        args = args.parse_args()
        return cls(args)

    def __getitem__(self, name):
        if self._config.get(name):
            return self._config[name]
        else:
            None
        # return self._config[name]