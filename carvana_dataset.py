import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class NOAATornadoDataset(Dataset):
    def __init__(self, root_path, test = False):
        self.root_path = root_path
        folder_prefix = "manual_test" if test else "train"
        ids = sorted(os.listdir(os.path.join(root_path, folder_prefix, "cape")))
        self.cape_paths = [os.path.join(root_path, folder_prefix, "cape", i) for i in ids]
        self.cin_paths = [os.path.join(root_path, folder_prefix, "cin", i) for i in ids]
        self.geo_paths = [os.path.join(root_path, folder_prefix, "geo", i) for i in ids]
        self.tor_paths = [os.path.join(root_path, f"{folder_prefix}_masks", "tornado", i) for i in ids]
        self.sigtor_paths = [os.path.join(root_path, f"{folder_prefix}_masks", "sigtor", i) for i in ids]
        assert len(self.cape_paths) == len(self.cin_paths) == len(self.geo_paths) == len(self.tor_paths) == len(self.sigtor_paths)

    def __getitem__(self, index):
        
        cape = np.load(self.cape_paths[index])
        cin = np.load(self.cin_paths[index])
        geo = np.load(self.geo_paths[index])

        tor_prob = np.load(self.tor_paths[index])
        sigtor_prob = np.load(self.sigtor_paths[index])

        x = torch.from_numpy(np.stack([cape, cin, geo], axis = 0)).float()
        y = torch.from_numpy(np.stack([tor_prob, sigtor_prob], axis = 0)).float()

        x = x / (x.abs().amax(dim=(1, 2), keepdim=True) + 1e-8)  # per channel

        x = TF.resize(x, [256, 256], antialias=True)
        y = TF.resize(y, [256, 256], interpolation=TF.InterpolationMode.NEAREST)       
        return x , y

    def __len__(self):
        return len(self.cape_paths)