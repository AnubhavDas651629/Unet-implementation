import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

class NOAATornadoDataset(Dataset):
    # Added optional file_list parameter to accept our 1-year slices cleanly
    def __init__(self, root_path, test=False, file_list=None):
        self.root_path = root_path
        folder_prefix = "manual_test" if test else "train"
        
        # If no specific subset is provided, read all files from the directory automatically
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
            f"Mismatch in file counts! Cape: {len(self.cape_paths)}, Cin: {len(self.cin_paths)}, Geo: {len(self.geo_paths)}, Tor: {len(self.tor_paths)}, SigTor: {len(self.sigtor_paths)}"

    def __getitem__(self, index):
        cape = np.load(self.cape_paths[index])
        cin = np.load(self.cin_paths[index])
        geo = np.load(self.geo_paths[index])

        tor_prob = np.load(self.tor_paths[index])
        sigtor_prob = np.load(self.sigtor_paths[index])

        x = torch.from_numpy(np.stack([cape, cin, geo], axis = 0)).float()
        y = torch.from_numpy(np.stack([tor_prob, sigtor_prob], axis = 0)).float()

        # Min-max / Max absolute normalization scaling per channel
        x = x / (x.abs().amax(dim=(1, 2), keepdim=True) + 1e-8)  # per channel

        # Double check spatial dimensions to ensure perfect 256x256 grid layouts
        x = TF.resize(x, [256, 256], antialias=True)
        y = TF.resize(y, [256, 256], interpolation=TF.InterpolationMode.NEAREST)       
        return x , y

    def __len__(self):
        return len(self.cape_paths)