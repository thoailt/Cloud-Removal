# codes/models/base_model.py
import torch
import torch.nn as nn
import os
# from torch.optim import lr_scheduler 

class BaseModel(nn.Module): 
    """
    Base class for all models. Provides common functionalities like
    managing device, saving/loading checkpoints, and learning rate updates.
    """
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.optimizers = [] 
        self.schedulers = []
        # (CPU/GPU)
        self.device = torch.device(f'cuda:{opts.gpu_ids.split(",")[0]}' if torch.cuda.is_available() and opts.gpu_ids else 'cpu')
        
        self.input_data = {}
        self.output_data = {}
        self.losses = {}
        self.visuals = {}

    def set_input(self, input_data):
        """
        Assign input data to the model and pass it to the device.
        The subclass will call this function.
        Args:
            input_data (dict): Dictionary containing the input tensors.
        """
        self.input_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}

    def forward(self):
        """
        Implements forward pass. Must be implemented by subclass.
        """
        raise NotImplementedError("The 'forward' method must be implemented by a subclass.")

    def optimize_parameters(self):
        """
        Performs an optimization step (forward, backward, step).
        Must be implemented by a subclass.  
        """
        raise NotImplementedError("The 'optimize_parameters' method must be implemented by a subclass.")

    def update_learning_rates(self):
        """
        Update learning rate for all registered schedulers.
        """
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()
        
        if self.optimizers:
            print(f'Current learning rate: {self.optimizers[0].param_groups[0]["lr"]:.6f}')


    def save_checkpoint(self, epoch):
        """
        Save model checkpoints and optimizers state.
        """
        save_dir = self.opts.save_model_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(), 
            'optimizers_state_dicts': [opt.state_dict() for opt in self.optimizers],
            
        }

        save_filename = f'CR_net_epoch_{epoch:03d}.pth'
        save_path = os.path.join(save_dir, save_filename)
        torch.save(state, save_path)
        print(f'Model checkpoint saved to {save_path}')

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoints and optimizers' states.
        Adjusted to be compatible with old checkpoint format ('network' key)
        and new format ('model_state_dict' key), and handle 'net_G.' prefix.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        loaded_state_dict = None
        if 'model_state_dict' in checkpoint:
            loaded_state_dict = checkpoint['model_state_dict']
            print("Loaded model state using 'model_state_dict' key (new format).")
        elif 'network' in checkpoint:
            loaded_state_dict = checkpoint['network']
            print("Loaded model state using 'network' key (old format).")
        else:
            raise KeyError("Checkpoint does not contain 'model_state_dict' or 'network' key. "
                        "Please ensure your checkpoint is saved in a compatible format.")

        current_model_state_dict = self.state_dict()

        new_state_dict = {}

        keys_to_remove_module = []
        for k in loaded_state_dict.keys():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = loaded_state_dict[k]
            else:
                new_state_dict[k] = loaded_state_dict[k]
        loaded_state_dict = new_state_dict
        
        mapped_state_dict = {}
        for k, v in loaded_state_dict.items():
            if not k.startswith('net_G.') and f'net_G.{k}' in current_model_state_dict:
                mapped_state_dict[f'net_G.{k}'] = v
                
            elif k in current_model_state_dict:
                mapped_state_dict[k] = v
            else:
                print(f"Warning: Skipping unexpected key: {k} (might be due to architecture changes or unmatched prefix)")
                continue 

        self.load_state_dict(mapped_state_dict, strict=False) 

        if 'optimizers_state_dicts' in checkpoint:
            for i, opt_state in enumerate(checkpoint['optimizers_state_dicts']):
                if i < len(self.optimizers):
                    self.optimizers[i].load_state_dict(opt_state)
            print("Loaded optimizer states.")
        else:
            print("Warning: No optimizer state found in checkpoint. Skipping optimizer loading (normal for old test checkpoints).")
            
        if 'schedulers_state_dicts' in checkpoint:
            for i, sch_state in enumerate(checkpoint['schedulers_state_dicts']):
                if i < len(self.schedulers):
                    self.schedulers[i].load_state_dict(sch_state)
            print("Loaded scheduler states.")
        else:
            print("Warning: No scheduler state found in checkpoint. Skipping scheduler loading.")

        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Checkpoint reports epoch {checkpoint.get('epoch', 'N/A')}, continuing from epoch {start_epoch}")
        return start_epoch

    def print_networks(self):
        """Print information about the number of parameters of the networks in the model."""
        print('---------- Networks initialized -------------')
        total_params = 0
        for name, module in self.named_children():
            if isinstance(module, nn.Module):
                num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total_params += num_params
                print(f'[Network] {name}: {num_params / 1e6:.3f} M parameters')
        print(f'[Total Network] Total number of parameters : {total_params / 1e6:.3f} M')
        print('-----------------------------------------------')

    def get_current_visuals(self):
        """
        Returns a dictionary of images (torch.Tensor on CPU) for visualization.
        Should be implemented by subclass.
        """
        return self.visuals # Return empty dict or specific visuals from subclass

    def get_current_losses(self):
        """
        Returns a dictionary of current scalar loss values.
        Should be implemented by subclass.
        """
        return self.losses # Return empty dict or specific losses from subclass