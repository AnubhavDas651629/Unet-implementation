import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

class NOAATornadoDataset(Dataset):
    def __init__(self, root_path, test=False, file_list=None):
        self.root_path = root_path
        folder_prefix = "manual_test" if test else "train"
        
        # Allows for clean chronological splitting in main.py
        if file_list is None:
            ids = sorted(os.listdir(os.path.join(root_path, folder_prefix, "cape")))
        else:
            ids = sorted(file_list)
            
        self.cape_paths = [os.path.join(root_path, folder_prefix, "cape", i) for i in ids]
        self.cin_paths = [os.path.join(root_path, folder_prefix, "cin", i) for i in ids]
        self.geo_paths = [os.path.join(root_path, folder_prefix, "geo", i) for i in ids]
        self.tor_paths = [os.path.join(root_path, f"{folder_prefix}_masks", "tornado", i) for i in ids]
        self.sigtor_paths = [os.path.join(root_path, f"{folder_prefix}_masks", "sigtor", i) for i in ids]
        
        assert len(self.cape_paths) == len(self.cin_paths) == len(self.geo_paths) == len(self.tor_paths) == len(self.sigtor_paths), \
            f"Mismatch in file counts! Cape: {len(self.cape_paths)}, Cin: {len(self.cin_paths)}"

    def __getitem__(self, index):
        # 1. Load the pre-scaled 256x256 numpy matrices directly
        cape = np.load(self.cape_paths[index])
        cin = np.load(self.cin_paths[index])
        geo = np.load(self.geo_paths[index])

        tor_prob = np.load(self.tor_paths[index])
        sigtor_prob = np.load(self.sigtor_paths[index])

        # 2. Stack them along Axis 0 to create explicit channel depths
        x = torch.from_numpy(np.stack([cape, cin, geo], axis=0)).float()       # Shape: [3, 256, 256]
        y = torch.from_numpy(np.stack([tor_prob, sigtor_prob], axis=0)).float() # Shape: [2, 256, 256]

        # 3. Secure clean absolute max normalization per channel
        channel_maxes = x.abs().amax(dim=(1, 2), keepdim=True)
        x = x / (channel_maxes + 1e-8)
     
        # Notice: TF.resize is completely removed!
        return x, y

    def __len__(self):
        return len(self.cape_paths)