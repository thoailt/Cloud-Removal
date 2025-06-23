# codes/utils/misc.py
import torch
import numpy as np
import os
import random # Add random import

def seed_torch(seed=7): # Giữ nguyên seed mặc định nếu bạn muốn
    """Set random seeds for reproducibility across CPU and GPU."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    torch.backends.cudnn.benchmark = False # Disable CuDNN benchmarking for reproducibility
    torch.backends.cudnn.enabled = False # Disable CuDNN for full determinism
    torch.backends.cudnn.deterministic = True

def print_options(opts):
    """Print parsed options to console and save to file."""
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opts).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

    # save to the disk (adjust save directory based on opts type)
    save_dir = None
    if hasattr(opts, 'save_model_dir'):
        save_dir = opts.save_model_dir
    elif hasattr(opts, 'results_dir'):
        save_dir = opts.results_dir

    if save_dir: # Only proceed if a valid save_dir is found
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

def ensure_dir(path):
    """Ensure that a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)