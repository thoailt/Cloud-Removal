# codes/main_test.py
import os
import torch
import argparse
import numpy as np
from PIL import Image # For saving images if needed

from utils.options import get_test_options
from utils.misc import seed_torch, print_options, ensure_dir 
from dataloader import AlignedDataset, get_train_val_test_filelists 
from models.model_CR_net import ModelCRNet 
from utils.metrics import MSE, PSNR, SSIM, MAE
from utils.visualization import convert_to_rgb, create_image_grid_with_titles 

def main():
    # 1. Parse testing options
    opts = get_test_options() # Get options defined in utils/options.py
    print_options(opts) # Print parsed options (from utils/misc.py)

    # 2. Configure GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    device = torch.device(f'cuda:{opts.gpu_ids.split(",")[0]}' if torch.cuda.is_available() and opts.gpu_ids else 'cpu')
    print(f"Using device: {device}")

    # 3. Set random seeds (optional for test, but good practice for consistency)
    seed_torch()

    # 4. Create DataLoader for test set
    # get_train_val_test_filelists giờ đây trả về pandas.DataFrame
    _, _, test_filelist_df = get_train_val_test_filelists(opts.data_list_filepath)
    
    data = AlignedDataset(opts, test_filelist_df) # DataFrame
    dataloader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=opts.batch_sz,
        shuffle=False, 
        num_workers=os.cpu_count() // 2, 
        pin_memory=True 
    )
    print(f"Loaded {len(test_filelist_df)} test samples.")

    # 5. Create Model and load checkpoint
    model = ModelCRNet(opts).to(device) 
    
    model.load_checkpoint(opts.model_path) 
    
    model.eval()

    # Initialize metrics tracking
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0 

    # Ensure results directory exists
    results_img_dir = os.path.join(opts.results_dir, 'images')
    ensure_dir(opts.results_dir)
    ensure_dir(results_img_dir)

    print("\n--- Starting Testing ---")
    with torch.no_grad(): 
        for i, data_batch in enumerate(dataloader):
            # Set input data to the model
            model.set_input(data_batch)
            
            # Perform forward pass (no optimization)
            model.forward() 
            
            # Get current visuals (CPU tensors) for metric calculation and visualization
            visuals = model.get_current_visuals()
            
            pred_cf = visuals['pred_Cloudfree_data']
            gt_cf = visuals['cloudfree_data']
            cloudy_data_tensor = visuals['cloudy_data']
            sar_data_tensor = visuals['SAR_data']
            cloud_mask_tensor = visuals['cloud_mask']
            
            file_name = data_batch['file_name'][0]
            
            current_mse = MSE(pred_cf, gt_cf)
            current_mae = MAE(pred_cf, gt_cf)
            current_psnr = PSNR(pred_cf, gt_cf)
            current_ssim = SSIM(pred_cf, gt_cf).item()
                        
            total_mse += current_mse * pred_cf.shape[0]
            total_mae += current_mae * pred_cf.shape[0]
            total_psnr += current_psnr * pred_cf.shape[0]
            total_ssim += current_ssim * pred_cf.shape[0]
            num_samples += pred_cf.shape[0]

            print(f"Processed sample {i+1}/{len(dataloader)} ({file_name}): "
                  f"PSNR={current_psnr:.4f}, SSIM={current_ssim:.4f}, MSE={current_mse:.4f}, MAE={current_mae:.4f}")

            # Optionally save visual results for a few samples or all
            if i % 10 == 0: # Save every 10th sample, adjust frequency as needed
                # Convert tensors to RGB numpy arrays (uint8)
                cloud_data_rgb = convert_to_rgb(cloudy_data_tensor)
                sar_data_rgb = convert_to_rgb(sar_data_tensor)
                predicted_rgb = convert_to_rgb(pred_cf)
                ground_truth_rgb = convert_to_rgb(gt_cf)
                cloud_mask_np = cloud_mask_tensor.numpy()

                # Create and save combined image
                combined_image = create_image_grid_with_titles(
                    cloud_data_rgb, sar_data_rgb, predicted_rgb, ground_truth_rgb, cloud_mask_np
                )
                
                # Use a cleaner filename for saving
                output_image_name = f'{os.path.splitext(file_name)[0]}_results.png'
                combined_image.save(os.path.join(results_img_dir, output_image_name))
                # print(f"Saved visualization for {file_name}")

    # Print average metrics at the end
    if num_samples > 0:
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        avg_mse = total_mse / num_samples
        avg_mae = total_mae / num_samples
    else:
        avg_psnr = avg_ssim = avg_mse = avg_mae = 0.0

    print("\n--- Overall Test Results ---")
    print(f"Total samples processed: {num_samples}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

    # Optionally, save overall results to a summary file
    with open(os.path.join(opts.results_dir, 'test_summary.txt'), 'w') as f:
        f.write(f"Total samples processed: {num_samples}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average MSE: {avg_mse:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")

if __name__ == "__main__":
    main()

    # --pixel_loss_type mse 