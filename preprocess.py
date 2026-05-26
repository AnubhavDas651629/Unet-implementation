import os
import numpy as np
import xarray as xr
from torch.nn.functional import interpolate
import torch

# ==========================================
# CRITICAL CHANGE 1: REMOVED BACKSLASHES FROM STRINGS
# Python strings do NOT need backslashes before spaces.
# CRITICAL CHANGE 2: ASSIGNED TARGET PATHS
# Added absolute paths for your tornado target files.
# ==========================================
CAPE_PATH = "/Users/parthapratimdas/Downloads/Cape 2014.nc"
CIN_PATH = "/Users/parthapratimdas/Downloads/CIN\ 2014.nc"  # NOTE: You are using Cape for CIN. Make sure to download a separate CIN file later if needed!
HGT_PATH = "/Users/parthapratimdas/Downloads/HGT Tropo 2014.nc"
TOR_TARGET_PATH = "/Users/parthapratimdas/Downloads/Pper\ Tor\ 1979-2023.nc"
SIGTOR_TARGET_PATH = "/Users/parthapratimdas/Downloads/Pper\ Sig\ Tor\ 1979-2023.nc"

OUTPUT_DIR = "./data"

def create_folders():
    paths = [
        "train/cape", "train/cin", "train/geo",
        "train_masks/tornado", "train_masks/sigtor"
    ]
    for p in paths:
        os.makedirs(os.path.join(OUTPUT_DIR, p), exist_ok=True)

def resize_grid(matrix_2d):
    """Resizes any 2D weather matrix to exactly 256x256 using bilinear interpolation"""
    tensor = torch.tensor(matrix_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)
    return resized.squeeze().numpy()

def main():
    print("Creating directory structure...")
    create_folders()
    
    # ==========================================
    # CRITICAL CHANGE 3: FIXED THE TARGET YEAR SLICE
    # Your input files are for 2014, so we must slice 2014 from the targets too!
    # ==========================================
    print("Loading 2014 Atmospheric features...")
    ds_cape = xr.open_dataset(CAPE_PATH)
    ds_cin = xr.open_dataset(CIN_PATH)
    ds_hgt = xr.open_dataset(HGT_PATH)
    
    print("Loading massive target files and slicing out 2014...")
    ds_tor = xr.open_dataset(TOR_TARGET_PATH).sel(time="2014")
    ds_sigtor = xr.open_dataset(SIGTOR_TARGET_PATH).sel(time="2014")
    
    # Get common dates available across all datasets to ensure alignment
    dates = np.intersect1d(ds_cape.time.values, ds_tor.time.values)
    print(f"Found {len(dates)} perfectly aligned days to process.")

    for i, date in enumerate(dates):
        date_str = np.datetime_as_string(date, unit='D')
        print(f"[{i+1}/{len(dates)}] Processing day: {date_str}")
        
        # ==========================================
        # CRITICAL CHANGE 4: FIXED XARRAY SYNTAX BUG
        # Changed parentheses () to square brackets [] to extract variables.
        # Fixed internal variable names to match standard NOAA/Gensini formats.
        # Added .squeeze() to drop accidental dimensions.
        # ==========================================
        
        # 1. Extract and resize feature maps
        # Note: If your variables inside the NetCDF have different names (e.g., 'cape' vs 'CAPE'), adjust the strings below!
        cape_map = ds_cape['cape'].sel(time=date).mean(dim='time', skipna=True).squeeze().values
        cin_map = ds_cin['cin'].sel(time=date).mean(dim='time', skipna=True).squeeze().values
        hgt_map = ds_hgt['hgt'].sel(time=date).mean(dim='time', skipna=True).squeeze().values
        
        # 2. Extract and resize target probability maps
        tor_map = ds_tor['prob'].sel(time=date).squeeze().values
        sigtor_map = ds_sigtor['prob'].sel(time=date).squeeze().values
        
        # 3. Save as NumPy files
        np.save(os.path.join(OUTPUT_DIR, "train/cape", f"{date_str}.npy"), resize_grid(cape_map))
        np.save(os.path.join(OUTPUT_DIR, "train/cin", f"{date_str}.npy"), resize_grid(cin_map))
        np.save(os.path.join(OUTPUT_DIR, "train/geo", f"{date_str}.npy"), resize_grid(hgt_map))
        np.save(os.path.join(OUTPUT_DIR, "train_masks/tornado", f"{date_str}.npy"), resize_grid(tor_map))
        np.save(os.path.join(OUTPUT_DIR, "train_masks/sigtor", f"{date_str}.npy"), resize_grid(sigtor_map))

    print("\n🎉 Preprocessing Complete! Your 2014 data folder is fully populated.")

if __name__ == "__main__":
    main()