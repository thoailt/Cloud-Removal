# codes/utils/options.py
import argparse

def get_train_options():
    """
    Parses and returns the command-line arguments for training.
    """
    parser = argparse.ArgumentParser(description='Training options for Cloud Removal')

    # Data options
    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
    parser.add_argument('--load_size', type=int, default=256, help='Image load size before cropping')
    parser.add_argument('--crop_size', type=int, default=128, help='Image crop size for training')
    parser.add_argument('--input_data_folder', type=str, default='../data', help='Path to the dataset root folder (e.g., ../../data from codes/)')
    parser.add_argument('--is_use_cloudmask', type=bool, default=True, help='Whether to use cloud mask as input')
    parser.add_argument('--cloud_threshold', type=float, default=0.2, help='Threshold for cloud mask generation (only useful when is_use_cloudmask=True)')
    parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv', help='Path to the CSV file containing data splits (e.g., ../../data/data.csv from codes/)')

    # Optimization options
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type (e.g., Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_step_size', type=int, default=5, help='Number of epochs to decay learning rate')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate decay rate (multiplicative factor)')
    parser.add_argument('--lr_start_epoch_decay', type=int, default=5, help='Epoch to start learning rate decay')
    parser.add_argument('--max_epochs', type=int, default=30, help='Maximum number of training epochs')
    
    # Loss weights (added for ModelCRNet's combined_loss)
    parser.add_argument('--pixel_loss_type', type=str, default='mae', choices=['mse', 'mae'], help='Type of pixel-wise loss: mse or mae')
    parser.add_argument('--lambda_masked', type=float, default=2.0, help='Weight for masked L1 loss')
    parser.add_argument('--lambda_mse', type=float, default=0.5, help='Weight for MSE loss (only used if pixel_loss_type is mse)')
    parser.add_argument('--lambda_mae', type=float, default=0.5, help='Weight for MAE loss (only used if pixel_loss_type is mae)')
    parser.add_argument('--lambda_ssim', type=float, default=0.5, help='Weight for (1 - SSIM) loss')


    # Logging and saving options
    parser.add_argument('--save_freq', type=int, default=1, help='Frequency (in epochs) to save model checkpoints')
    parser.add_argument('--log_freq', type=int, default=10, help='Frequency (in steps) to log metrics to console and Wandb')
    parser.add_argument('--image_log_freq', type=int, default=50, help='Frequency (in steps) to log visual results to Wandb')
    parser.add_argument('--save_model_dir', type=str, default='./ckpt', help='Directory used to store trained networks')
    parser.add_argument('--project_name', type=str, default='Cloud_Removal', help='Wandb project name')
    parser.add_argument('--run_name', type=str, default='CRNet_Training', help='Wandb run name')

    # GPU options
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated list of GPU IDs to use (e.g., "0", "0,1")')

    return parser.parse_args()

def get_test_options():
    """
    Parses and returns the command-line arguments for testing.
    """
    parser = argparse.ArgumentParser(description='Testing options for Cloud Removal')

    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for testing')
    parser.add_argument('--load_size', type=int, default=256, help='Image load size before cropping')
    parser.add_argument('--crop_size', type=int, default=128, help='Image crop size for testing')
    parser.add_argument('--input_data_folder', type=str, default='../data', help='Path to the dataset root folder (e.g., ../../data from codes/)')
    parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv', help='Path to the CSV file containing data splits (e.g., ../../data/data.csv from codes/)')
    parser.add_argument('--model_path', type=str, default="../ckpt/CR_net.pth", help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save test results (e.g., ../../results from codes/)')
    parser.add_argument('--is_use_cloudmask', type=bool, default=True, help='Whether to use cloud mask in dataloader during test')
    parser.add_argument('--cloud_threshold', type=float, default=0.2, help='Threshold for cloud mask generation in dataloader (only useful when is_use_cloudmask=True)')

    # Add optimizer-related options for consistency with ModelCRNet's __init__
    # Even if not used in testing, ModelCRNet's __init__ expects them.
    # Provide dummy values or values that align with the trained model.
    parser.add_argument('--optimizer', type=str, default='Adam', help='Dummy optimizer type for model init in test (not actually used)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Dummy learning rate for model init in test (not actually used)')
    parser.add_argument('--lr_step_size', type=int, default=5, help='Dummy lr decay step size for model init in test (not actually used)')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Dummy lr decay gamma for model init in test (not actually used)')
    
    # Loss weights (ModelCRNet's __init__ might need these for _calculate_losses if called during init, though not ideal)
    # Better to initialize ModelCRNet without needing loss functions at test time,
    # but for now, add dummy values if it crashes without them.
    parser.add_argument('--pixel_loss_type', type=str, default='mse', choices=['mse', 'mae'], help='Dummy pixel-wise loss type for model init in test')
    parser.add_argument('--lambda_masked', type=float, default=2.0, help='Dummy weight for masked L1 loss in test (not actually used)')
    parser.add_argument('--lambda_mse', type=float, default=0.5, help='Dummy weight for MSE loss in test (not actually used)')
    parser.add_argument('--lambda_mae', type=float, default=0.5, help='Dummy weight for MAE loss in test (not actually used)')
    parser.add_argument('--lambda_ssim', type=float, default=0.5, help='Dummy weight for (1 - SSIM) loss in test (not actually used)')

    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated list of GPU IDs to use (e.g., "0", "0,1")')

    return parser.parse_args()