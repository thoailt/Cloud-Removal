# codes/training/trainer.py
import os
import time
import torch
import wandb
from utils.metrics import MSE, PSNR, SSIM, MAE
from utils.visualization import convert_to_rgb, create_image_grid_with_titles
from utils.misc import ensure_dir

class Trainer:
    def __init__(self, model, dataloader, opts):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): Model object, inherited from BaseModel, with methods
                       `set_input`, `optimize_parameters`, `get_current_visuals`,
                       `get_current_losses`, `update_learning_rates`, `save_checkpoint`.
            dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            opts (argparse.Namespace): Configuration options.
        """
        self.model = model
        self.dataloader = dataloader
        self.opts = opts
        self.total_steps = 0
        self.log_loss_sum = 0
        
        ensure_dir(self.opts.save_model_dir)

    def _log_metrics_to_wandb(self, epoch, current_loss_value, visuals_dict):
        """
        Log metrics to console and Wandb.

        Args:
            epoch (int): Current epoch.
            current_loss_value (float): Average loss value over log_freq interval.
            visuals_dict (dict): Dictionary containing image tensors (on CPU) from model.get_current_visuals().
        """
        
        pred_cf = visuals_dict['pred_Cloudfree_data']
        gt_cf = visuals_dict['cloudfree_data']
        
        mae_value = MAE(pred_cf, gt_cf)
        mse_value = MSE(pred_cf, gt_cf)
        psnr_value = PSNR(pred_cf, gt_cf)
        ssim_value = SSIM(pred_cf, gt_cf) 

        if self.opts.pixel_loss_type == 'mse':
             print(f'Epoch {epoch:03d}, Steps {self.total_steps:06d}, '
                  f'Loss: {current_loss_value:.4f}, '
                  f'PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, MSE: {mse_value:.4f}')
        elif self.opts.pixel_loss_type == 'mae':
            print(f'Epoch {epoch:03d}, Steps {self.total_steps:06d}, '
                  f'Loss: {current_loss_value:.4f}, '
                  f'PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, MAE: {mae_value:.4f}')

        wandb.log({
            "epoch": epoch,
            "step": self.total_steps,
            "loss": current_loss_value,
            "PSNR": psnr_value,
            "SSIM": ssim_value,
            "MSE": mse_value,
            "MAE": mae_value,
            
            "learning_rate": self.model.optimizers[0].param_groups[0]['lr']
        })

    def _log_images_to_wandb(self, visuals_dict):
        """
        Convert image tensors to RGB numpy arrays and log them to Wandb.

        Args:
            visuals_dict (dict): Dictionary containing image tensors (on CPU) from model.get_current_visuals().
        """
        # Convert tensors to numpy RGB (uint8)
        cloud_data_rgb = convert_to_rgb(visuals_dict['cloudy_data'])
        sar_data_rgb = convert_to_rgb(visuals_dict['SAR_data'])
        predicted_rgb = convert_to_rgb(visuals_dict['pred_Cloudfree_data'])
        ground_truth_rgb = convert_to_rgb(visuals_dict['cloudfree_data'])
        
        cloud_mask_np = visuals_dict['cloud_mask'].numpy() 

        combined_image_with_titles = create_image_grid_with_titles(
            cloud_data_rgb, sar_data_rgb, predicted_rgb, ground_truth_rgb, cloud_mask_np
        )

        wandb.log({
            "Combined_Image": wandb.Image(combined_image_with_titles, caption=f"Results at step {self.total_steps}")
        })

    def train(self):
        """
        Start the model training process.
        """
        print(f'#training images: {len(self.dataloader.dataset)} (Batch size: {self.opts.batch_sz})')

        for epoch in range(self.opts.max_epochs):
            self.model.train() 
            print(f'\n--- Epoch {epoch+1}/{self.opts.max_epochs} ---')

            epoch_start_time = time.time()
            for i, data in enumerate(self.dataloader):
                self.total_steps += 1
                
                self.model.set_input(data)
                
                batch_loss = self.model.optimize_parameters()
                self.log_loss_sum += batch_loss
                
                if self.total_steps % self.opts.log_freq == 0:
                    avg_loss = self.log_loss_sum / self.opts.log_freq
                    visuals = self.model.get_current_visuals() 
                    self._log_metrics_to_wandb(epoch + 1, avg_loss, visuals)
                    self.log_loss_sum = 0 
                    
                if self.total_steps % self.opts.image_log_freq == 0:
                    visuals = self.model.get_current_visuals()
                    self._log_images_to_wandb(visuals)
            
            epoch_time = time.time() - epoch_start_time
            print(f'End of epoch {epoch+1}. Time Taken: {epoch_time:.2f} sec.')

            if (epoch + 1) % self.opts.save_freq == 0:
                self.model.save_checkpoint(epoch + 1)
                wandb.log({"checkpoint": f"Checkpoint saved at epoch {epoch + 1}"})

            if (epoch + 1) >= self.opts.lr_start_epoch_decay:
                self.model.update_learning_rates()