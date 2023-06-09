from torch.utils.data import Dataset
import os
import pickle
import gzip

class CacheDataset(Dataset):
    def __init__(self, config, is_train=True):
        super().__init__()
        
        if is_train:
            self.datadir = config['train_cache_data_dir']
            self._filename = [n for n in os.listdir(self.datadir) if n.endswith('pkl')]
        else:
            self.datadir = config['valid_cache_data_dir']
            self._filename = [n for n in os.listdir(self.datadir) if n.endswith('pkl')]
        
    def __getitem__(self, idx):
        with gzip.open(os.path.join(self.datadir, self._filename[idx]), mode='rb') as f:
            p = pickle.load(f)
        return p[0], p[1]
    
    def __len__(self):
        return len(self._filename)  