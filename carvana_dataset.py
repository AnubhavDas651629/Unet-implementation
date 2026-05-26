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
        self.cape_paths = sorted([os.path.join(root_path, folder_prefix, "cape", i) for i in os.listdir(os.path.join(root_path, folder_prefix, "cape"))])
        self.cin_paths = sorted([os.path.join(root_path, folder_prefix, "cin", i) for i in os.listdir(os.path.join(root_path, folder_prefix, "cin"))])
        self.geo_paths = sorted([os.path.join(root_path, folder_prefix, "geo", i) for i in os.listdir(os.path.join(root_path, folder_prefix, "geo"))])

        self.tor_paths = sorted([os.path.join(root_path, f"{folder_prefix}_masks", "tornado", i) for i in os.listdir(os.path.join(root_path, f"{folder_prefix}_masks", "tornado"))])
        self.sigtor_paths = sorted([os.path.join(root_path, f"{folder_prefix}_masks", "sigtor", i) for i in os.listdir(os.path.join(root_path, f"{folder_prefix}_masks", "sigtor"))])
       

    def __getitem__(self, index):
        
        cape = np.load(self.cape_paths[index])
        cin = np.load(self.cin_paths[index])
        geo = np.load(self.geo_paths[index])

        tor_prob = np.load(self.tor_paths[index])
        sigtor_prob = np.load(self.sigtor_paths[index])

        x = torch.from_numpy(np.stack([cape, cin, geo], axis = 0)).float()
        y = torch.from_numpy(np.stack([tor_prob, sigtor_prob], axis = 0)).float()

        x = x / (torch.max(torch.abs(x))+1e-8)

        x = TF.resize(x, [256, 256], antialias=True)
        y = TF.resize(y, [256, 256], antialias=True)

        return x , y

    def __len__(self):
        return len(self.cape_paths)