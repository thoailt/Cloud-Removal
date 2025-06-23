# codes/dataloader.py
import os
import numpy as np
import random # Keep if random.randint is used for cropping
import rasterio # For reading .tif files
import pandas as pd # To handle CSV filelist more robustly

import torch
from torch.utils.data import Dataset

from feature_detectors import get_cloud_cloudshadow_mask # Import the cloud/shadow detection logic

class AlignedDataset(Dataset):
    """
    A dataset class for paired image data (cloudy, SAR, cloud-free, mask) from Sentinel-2/1.
    It handles loading, normalization, and cropping.
    """

    def __init__(self, opts, filelist_df): # Expect filelist as a pandas DataFrame now
        """Initialize this dataset class.

        Args:
            opts (argparse.Namespace): Stores all the experiment flags.
            filelist_df (pd.DataFrame): DataFrame containing file paths and split info.
        """
        self.opts = opts
        self.filelist_df = filelist_df # Store as DataFrame
        self.n_images = len(self.filelist_df)

        # Define clipping/normalization parameters based on your data types
        # SAR (data_type=1) clip parameters
        self.sar_clip_min = np.array([-25.0, -32.5]) # For 2 SAR channels
        self.sar_clip_max = np.array([0.0, 0.0]) # For 2 SAR channels

        # Optical (data_type=2 or 3) clip parameters (for 13 bands)
        # Assuming you meant 13 channels of 0 and 10000 respectively
        self.opt_clip_min = np.array([0] * 13)
        self.opt_clip_max = np.array([10000] * 13)
        
        self.max_val = 1.0 # Max value after normalization (e.g., 1.0 for [0, 1] range)
        self.opt_scale_factor = 10000.0 # Scaling factor for optical data

        # Check for consistency of options
        if not hasattr(self.opts, 'is_test'):
            self.opts.is_test = False # Default to False if not present (for random cropping)


    def __getitem__(self, index):
        """
        Returns a dictionary of data for a given index.
        """
        sample_row = self.filelist_df.iloc[index] # Get row as a Series from DataFrame

        # Construct full paths using column names (more readable and robust)
        # Adjust these column names based on your actual data.csv header
        # Assuming data.csv has columns like 's1_path_col', 's2_cloudfree_path_col', etc.
        # Based on your get_train_val_test_filelists, fileID[1] is s1_folder, fileID[4] is file_name.
        # Let's infer column names for now based on previous `fileID` usage:
        # fileID[1] -> s1_folder_name
        # fileID[2] -> s2_cloudfree_folder_name
        # fileID[3] -> s2_cloudy_folder_name
        # fileID[4] -> image_file_name (e.g., "tile_X_Y.tif")

        # Let's assume your CSV is structured something like:
        # split,s1_folder,s2_cloudfree_folder,s2_cloudy_folder,file_name
        # 1,s1,s2_cloudfree,s2_cloudy,image_001.tif
        
        s1_folder_name = sample_row.iloc[1] # Assuming 2nd column
        s2_cloudfree_folder_name = sample_row.iloc[2] # Assuming 3rd column
        s2_cloudy_folder_name = sample_row.iloc[3] # Assuming 4th column
        image_file_name = sample_row.iloc[4] # Assuming 5th column (file_name)

        s1_path = os.path.join(self.opts.input_data_folder, s1_folder_name, image_file_name)
        s2_cloudfree_path = os.path.join(self.opts.input_data_folder, s2_cloudfree_folder_name, image_file_name)
        s2_cloudy_path = os.path.join(self.opts.input_data_folder, s2_cloudy_folder_name, image_file_name)

        s1_data = self._load_image(s1_path, band_count=2).astype(np.float32) # Load 2-band SAR
        s2_cloudfree_data = self._load_image(s2_cloudfree_path, band_count=13).astype(np.float32) # Load 13-band Optical
        s2_cloudy_data = self._load_image(s2_cloudy_path, band_count=13).astype(np.float32) # Load 13-band Optical

        # Generate cloud mask
        cloud_mask = None
        if self.opts.is_use_cloudmask:
            # get_cloud_cloudshadow_mask now returns 0 (clear), 1 (cloud), -1 (shadow)
            cloud_mask = get_cloud_cloudshadow_mask(s2_cloudy_data.copy(), self.opts.cloud_threshold)
            # IMPORTANT: Do not convert -1 to 1 here. Keep -1 for shadow as used in ModelCRNet.
            # cloud_mask[cloud_mask != 0] = 1 # REMOVED: This line would overwrite shadow info

        # Normalize data (SAR and Optical use different methods)
        s1_data = self._normalize_sar(s1_data)
        s2_cloudfree_data = self._normalize_optical(s2_cloudfree_data)
        s2_cloudy_data = self._normalize_optical(s2_cloudy_data)

        # Convert to PyTorch tensors (from numpy: C, H, W -> tensor)
        s1_data = torch.from_numpy(s1_data)
        s2_cloudfree_data = torch.from_numpy(s2_cloudfree_data)
        s2_cloudy_data = torch.from_numpy(s2_cloudy_data)
        
        if self.opts.is_use_cloudmask:
            cloud_mask = torch.from_numpy(cloud_mask).long() # Mask is int, use .long()

        # Apply random cropping for training, or center crop for testing
        if self.opts.load_size > self.opts.crop_size:
            H, W = s2_cloudy_data.shape[1], s2_cloudy_data.shape[2] # Assume all images have same H, W
            
            if not self.opts.is_test:
                y = random.randint(0, H - self.opts.crop_size)
                x = random.randint(0, W - self.opts.crop_size)
            else:
                y = (H - self.opts.crop_size) // 2
                x = (W - self.opts.crop_size) // 2
            
            s1_data = s1_data[:, y:y+self.opts.crop_size, x:x+self.opts.crop_size]
            s2_cloudfree_data = s2_cloudfree_data[:, y:y+self.opts.crop_size, x:x+self.opts.crop_size]
            s2_cloudy_data = s2_cloudy_data[:, y:y+self.opts.crop_size, x:x+self.opts.crop_size]
            if self.opts.is_use_cloudmask:
                cloud_mask = cloud_mask[y:y+self.opts.crop_size, x:x+self.opts.crop_size]
        
        # Prepare results dictionary
        results = {
            'cloudy_data': s2_cloudy_data,
            'cloudfree_data': s2_cloudfree_data,
            'SAR_data': s1_data,
            'file_name': image_file_name, # Use actual file name for logging/debugging
            'paths': (s2_cloudy_path, s1_path, s2_cloudfree_path) # For debugging/logging full paths
        }
        if self.opts.is_use_cloudmask:
            results['cloud_mask'] = cloud_mask

        return results

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_images

    def _load_image(self, path, band_count):
        """
        Loads a multi-band image using rasterio and handles NaN values.
        """
        try:
            with rasterio.open(path, 'r', driver='GTiff') as src:
                image = src.read() # Reads as (C, H, W)
            # Fill holes and artifacts with channel-wise nanmean
            for c in range(image.shape[0]):
                if np.any(np.isnan(image[c])):
                    image[c][np.isnan(image[c])] = np.nanmean(image[c])
            return image
        except rasterio.errors.RasterioIOError as e:
            print(f"Error loading image {path}: {e}")
            # Return a dummy black image or raise error based on your error handling strategy
            # For now, return zeros array
            dummy_image = np.zeros((band_count, self.opts.load_size, self.opts.load_size), dtype=np.float32)
            return dummy_image
        except Exception as e:
            print(f"An unexpected error occurred loading image {path}: {e}")
            dummy_image = np.zeros((band_count, self.opts.load_size, self.opts.load_size), dtype=np.float32)
            return dummy_image


    def _normalize_sar(self, data_image):
        """Normalize SAR data (assumes 2 channels) based on clip_min/max."""
        normalized_data = np.zeros_like(data_image)
        for channel in range(data_image.shape[0]): # Iterate over channels
            # Clip values for each channel
            clipped_channel = np.clip(data_image[channel], self.sar_clip_min[channel], self.sar_clip_max[channel])
            # Shift to positive range
            shifted_channel = clipped_channel - self.sar_clip_min[channel]
            # Normalize to [0, max_val]
            range_val = self.sar_clip_max[channel] - self.sar_clip_min[channel]
            if range_val > 1e-8: # Avoid division by zero
                normalized_data[channel] = self.max_val * (shifted_channel / range_val)
            else: # If min == max, set to 0 (or constant if that's desired)
                normalized_data[channel] = 0.0 # Or self.max_val / 2 if middle point is preferred
        return normalized_data

    def _normalize_optical(self, data_image):
        """Normalize Optical data (assumes 13 channels) by dividing by scale_factor."""
        # Clip values (channel-wise clip for optical based on your old code)
        normalized_data = np.zeros_like(data_image)
        for channel in range(data_image.shape[0]):
            normalized_data[channel] = np.clip(data_image[channel], self.opt_clip_min[channel], self.opt_clip_max[channel])
        
        # Scale all channels by the common factor
        normalized_data /= self.opt_scale_factor
        return normalized_data

def get_train_val_test_filelists(listpath):
    """
    Loads the data list from CSV using pandas and splits into train, val, test DataFrames.
    Assumes listpath is a CSV with a 'split' column (e.g., 1 for train, 2 for val, 3 for test)
    and other columns like 's1_folder', 's2_cloudfree_folder', 's2_cloudy_folder', 'file_name'.
    """
    column_names = ['split', 's1_folder', 's2_cloudfree_folder', 's2_cloudy_folder', 'file_name'] 
    
    try:
        df = pd.read_csv(listpath)

        # print(f"DEBUG: Successfully read CSV from: {listpath}")
        # print(f"DEBUG: Total rows in CSV (after header parsing): {len(df)}")
        # print(f"DEBUG: First 5 rows of DataFrame (after header parsing):\n{df.head()}") 
        # print(f"DEBUG: Unique split types found in 'split' column: {df['split'].unique()}")

        df['split'] = pd.to_numeric(df['split'], errors='coerce') 
        df.dropna(subset=['split'], inplace=True)

        train_df = df[df['split'] == 1].reset_index(drop=True)
        val_df = df[df['split'] == 2].reset_index(drop=True)
        test_df = df[df['split'] == 3].reset_index(drop=True)

        # print(f"DEBUG: Train samples: {len(train_df)}")
        # print(f"DEBUG: Val samples: {len(val_df)}")
        # print(f"DEBUG: Test samples: {len(test_df)}")

        return train_df, val_df, test_df
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {listpath}")
        raise # Re-raise the exception after printing
    except Exception as e:
        print(f"ERROR: An error occurred while reading or processing the CSV: {e}")
        raise # Re-raise for full traceback

# Example usage for self-contained testing
if __name__ == "__main__":
    # Import necessary utilities for standalone test
    import argparse
    from utils.misc import print_options, seed_torch # Assuming these are in utils/misc.py

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='../../data') # Adjust path for testing from codes/dataloader.py
    parser.add_argument('--data_list_filepath', type=str, default='../../data/data.csv') # Adjust path
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=True)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)
    parser.add_argument('--batch_sz', type=int, default=1) # Needed for DataLoader
    # Add dummy gpu_ids if opts is passed to model/trainer, to avoid error.
    parser.add_argument('--gpu_ids', type=str, default='0') 

    opts = parser.parse_args() 
    print_options(opts)
    seed_torch() # Set seed for reproducibility in testing

    # Get file lists (using the pandas version)
    train_filelist_df, val_filelist_df, test_filelist_df = get_train_val_test_filelists(opts.data_list_filepath)
    
    # Create dataset for testing (e.g., using test_filelist_df)
    data = AlignedDataset(opts, test_filelist_df)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=opts.batch_sz, shuffle=False)

    print(f"\nTesting dataloader with {len(test_filelist_df)} samples...")
    _iter = 0
    for results in dataloader:
        cloudy_data = results['cloudy_data']
        cloudfree_data = results['cloudfree_data']
        SAR = results['SAR_data']
        cloud_mask = results.get('cloud_mask', None) # Use .get for cloud_mask
        file_name = results['file_name']
        
        print(f"\n--- Sample {_iter} ---")
        print(f"File: {file_name}")
        print(f"Cloudy data shape: {cloudy_data.shape}, dtype: {cloudy_data.dtype}, min: {cloudy_data.min():.4f}, max: {cloudy_data.max():.4f}")
        print(f"Cloud-free data shape: {cloudfree_data.shape}, dtype: {cloudfree_data.dtype}, min: {cloudfree_data.min():.4f}, max: {cloudfree_data.max():.4f}")
        print(f"SAR data shape: {SAR.shape}, dtype: {SAR.dtype}, min: {SAR.min():.4f}, max: {SAR.max():.4f}")
        if cloud_mask is not None:
            print(f"Cloud mask shape: {cloud_mask.shape}, dtype: {cloud_mask.dtype}, unique values: {torch.unique(cloud_mask)}")
        
        if _iter >= 5: # Limit output for testing
            break
        _iter += 1
    print("Dataloader test finished.")