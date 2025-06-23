# codes/models/model_CR_net.py
import torch
import torch.nn as nn

from models.base_model import BaseModel
from .net_CR_RDN import RDN_residual_CR
from utils.metrics import SSIM, MSE, PSNR 

class ModelCRNet(BaseModel): 
    def __init__(self, opts):
        super().__init__(opts) 

        self.opts = opts
        
        # 1. Create network
        self.net_G = RDN_residual_CR(self.opts.crop_size).to(self.device)
        
        # Parallel training 
        if len(self.opts.gpu_ids.split(',')) > 1 and torch.cuda.is_available():
            print(f"Using {len(self.opts.gpu_ids.split(','))} GPUs for DataParallel training.")
            self.net_G = nn.DataParallel(self.net_G)

        self.print_networks() 
        
        # 2. Initialize optimizers and schedulers
        if self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opts.lr)
            
        self.optimizers.append(self.optimizer_G)
        
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G, step_size=self.opts.lr_step_size, gamma=self.opts.lr_gamma # Sử dụng tên opts mới
        )
        
        self.schedulers.append(self.scheduler_G)
        
        # 3. Define loss functions
        if self.opts.pixel_loss_type == 'mse':
            self.criterion_pixel = nn.MSELoss()
            print("Using MSE Loss for pixel-wise comparison.")
        elif self.opts.pixel_loss_type == 'mae':
            self.criterion_pixel = nn.L1Loss()
            print("Using MAE (L1) Loss for pixel-wise comparison.")
        else:
            raise ValueError(f"Unsupported pixel_loss_type: {self.opts.pixel_loss_type}. Choose 'mse' or 'mae'.")
            
        self.criterion_l1 = nn.L1Loss()
                        
        # Internal storage for current batch data and predictions
        self._cloudy_data = None
        self._sar_data = None
        self._cloudfree_data = None
        self._cloud_mask = None
        self._pred_cloudfree_data = None

    def set_input(self, input_data):
        """
        Gán dữ liệu đầu vào cho model. Input_data là dictionary từ Dataloader.
        """
        super().set_input(input_data)
        
        self._cloudy_data = self.input_data['cloudy_data']
        self._sar_data = self.input_data['SAR_data']
        self._cloudfree_data = self.input_data['cloudfree_data']
        self._cloud_mask = self.input_data.get('cloud_mask', None)
        
        if self._cloud_mask is not None:
             self._cloud_mask = self._cloud_mask.float() # Convert mask to float for calculations

    def forward(self):
        """
        Performs the forward pass of the generator network and stores the prediction.
        """
        
        self._pred_cloudfree_data = self.net_G(self._cloudy_data, self._sar_data)
        
        return self._pred_cloudfree_data 
    
    def _calculate_losses(self):
        """
        Calculates all relevant losses and updates self.losses.
        """
        # Loss cho Generator
        self._pred_cloudfree_data = self.forward() # Lấy output từ forward
        
        # Đảm bảo cloud_mask có shape phù hợp
        processed_cloud_mask = self._cloud_mask.unsqueeze(1).expand_as(self._pred_cloudfree_data)
        
        # Tạo mask binary: 1 cho mây/bóng, 0 cho vùng trong suốt
        binary_affected_mask = (processed_cloud_mask != 0).float() 
        
        # 1. Masked L1 Loss (chỉ tính trên vùng có mây/bóng)
        self.losses['masked_L1'] = self.criterion_l1(self._pred_cloudfree_data * binary_affected_mask, 
                                                     self._cloudfree_data * binary_affected_mask)

        # 2. Pixel-wise Loss (MSE hoặc MAE)
        # Hàm criterion_pixel đã được chọn ở __init__
        self.losses['pixel_loss'] = self.criterion_pixel(self._pred_cloudfree_data, self._cloudfree_data)
        
        # 3. SSIM Loss (trên toàn bộ ảnh, 1 - SSIM để biến thành loss)
        self.losses['ssim_loss'] = 1 - SSIM(self._pred_cloudfree_data, self._cloudfree_data)

        # Tổng hợp các loss
        lambda_masked = self.opts.lambda_masked
        # Chọn weight cho pixel_loss dựa trên loại loss đã chọn
        lambda_pixel = self.opts.lambda_mse if self.opts.pixel_loss_type == 'mse' else self.opts.lambda_mae 
        lambda_ssim = self.opts.lambda_ssim
        
        self.loss_G = (lambda_masked * self.losses['masked_L1'] + 
                       lambda_pixel * self.losses['pixel_loss'] + 
                       lambda_ssim * self.losses['ssim_loss'])

        self.losses['total_G'] = self.loss_G.item() # Lưu tổng loss cho logging
        
        return self.loss_G
    
    def optimize_parameters(self):
        """
        Performs one optimization step for the generator.
        """
        # Calculate losses
        loss_G = self._calculate_losses()

        # Zero gradients, backward pass, and update weights
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()
        
        return self.losses['total_G']

    def get_current_visuals(self):
        """
        Returns a dictionary of current image tensors (on CPU, detached) for visualization.
        """
        self.visuals = {
            'cloudy_data': self._cloudy_data.detach().cpu(),
            'SAR_data': self._sar_data.detach().cpu(),
            'pred_Cloudfree_data': self._pred_cloudfree_data.detach().cpu(),
            'cloudfree_data': self._cloudfree_data.detach().cpu(),
            'cloud_mask': self._cloud_mask.detach().cpu()
        }
        return self.visuals

    def get_current_losses(self):
        """
        Returns a dictionary of current scalar loss values.
        """

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.losses.items()}
