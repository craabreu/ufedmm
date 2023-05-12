from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class Dataset(Dataset):
    def __init__(self, feature_file, cv_labels, force_labels):
        super().__init__()
        assert isinstance(cv_labels, list)
        assert isinstance(force_labels, list)
        assert len(cv_labels) > 0
        assert len(force_labels) > 0
        assert len(cv_labels) == len(force_labels)
        df = pd.read_csv(feature_file)
        self.cvs = np.array(df[cv_labels])
        self.forces = np.array(df[force_labels])
        self.cv_mean = np.mean(self.cvs, axis=0, keepdims=True)
        self.cv_std = np.std(self.cvs, axis=0, keepdims=True)
        self.force_mean = np.mean(self.forces, axis=0, keepdims=True)
        self.force_std = np.std(self.forces, axis=0, keepdims=True)

    def __len__(self):
        return len(self.cvs)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.cvs[idx]), torch.Tensor(self.forces[idx])