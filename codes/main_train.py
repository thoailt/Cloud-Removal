# codes/main_train.py
import os
import torch
import wandb

from utils.options import get_train_options
from utils.misc import seed_torch, print_options 
from dataloader import AlignedDataset, get_train_val_test_filelists 
from models.model_CR_net import ModelCRNet 
from training.trainer import Trainer

def main():
    # 1. Parse training options
    opts = get_train_options() 
    print_options(opts) 

    # 2. Configure GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    if not torch.cuda.is_available() or not opts.gpu_ids: 
        print("CUDA not available or GPU ID not specified. Training on CPU.")
        opts.gpu_ids = "" # Set to empty to indicate CPU usage consistently
    else:
        print(f"Using GPU(s): {opts.gpu_ids}")

    # 3. Set random seeds for reproducibility
    seed_torch() 

    # 4. Initialize Weights & Biases (Wandb)
    wandb.init(
        project=opts.project_name,
        name=opts.run_name,      
        config=vars(opts)        
    )

    # 5. Create DataLoader
    train_filelist, _, _ = get_train_val_test_filelists(opts.data_list_filepath)
    
    train_data = AlignedDataset(opts, train_filelist)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=opts.batch_sz,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True 
    )
    print(f"Loaded {len(train_data)} training samples.")

    # 6. Create Model
    model = ModelCRNet(opts)
    if torch.cuda.is_available() and opts.gpu_ids:
        print(f"Model moved to {model.device}.")
    else:
        print("Model remains on CPU.")

    # 7. Create Trainer and start training
    trainer = Trainer(model, train_dataloader, opts)
    trainer.train()

    # 8. Finish Wandb run
    wandb.finish()

if __name__ == '__main__':
    main()