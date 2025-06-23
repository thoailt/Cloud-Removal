# Cloud-Aware SAR Fusion for Enhanced Optical Sensing in Space Missions

This repository contains the official implementation for the paper **"Cloud-Aware SAR Fusion for Enhanced Optical Sensing in Space Missions"**.

## Overview

This project introduces a **Cloud-Aware Reconstruction Framework** for satellite image cloud removal. It fuses SAR and optical data, utilizing an attention mechanism and an adaptive loss strategy to prioritize and enhance reconstruction accuracy specifically in cloud-occluded regions, yielding high-fidelity, cloud-free optical images.

## Prerequisites & Installation

This code was developed and tested with **Python 3.10** and **CUDA 12.4**.

1.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv cloud_removal_env
    .\cloud_removal_env\Scripts\activate # On Windows
    source cloud_removal_env/bin/activate # On Linux/macOS
    # Or, if using conda:
    # conda create -n cloud_removal_env python=3.10
    # conda activate cloud_removal_env
    ```
2.  **Install PyTorch with CUDA support:**
    Ensure you install the PyTorch version compatible with your CUDA toolkit. For CUDA 12.4, use the following command (adjust if your CUDA version differs):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

3.  **Install Other Dependencies:**
    After activating your virtual environment, navigate to the project's root directory (e.g., `Cloud-Removal`) and install the remaining libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    *(**Note**: Your `requirements.txt` file has been curated to include only direct dependencies, and `pip` will automatically handle their sub-dependencies.)*

## Dataset

This project utilizes the [SEN12MS-CR DATASET](https://patricktum.github.io/cloud_removal/sen12mscr/), which serves as a benchmark for multimodal cloud removal in remote sensing.

## Get Started

### 1.  Train the Network
```bash
cd codes/
python main_train.py
```
To view more training options:
```bash
cd codes/
python main_train.py --help
```

### 2. Test the Network
```bash
cd codes/
python main_test.py --model_path ../ckpt/CR_net.pth
```
Test results and metric summaries will be saved in the `./results/` directory.

## Contact
Email: thoailt@hcmue.edu.vn