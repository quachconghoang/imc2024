from pathlib import Path
import os
import torch
import kornia as K
import numpy as np
import h5py

root_path=Path('/')
if os.getenv('LOCAL_DATASETS'):
    root_path = Path(os.getenv('LOCAL_DATASETS'))

INPUT_PATH =  root_path / 'kaggle' / 'input'
WORKING_PATH = root_path / 'kaggle' / 'working'
DATA_PATH = INPUT_PATH / 'image-matching-challenge-2024'

# device = K.utils.get_cuda_device_if_available(0)
DEVICE: torch.device = K.utils.get_cuda_device_if_available(0)
print(DEVICE)

LOCAL_DEBUG = True
SUBMISSION = False

class CONFIG:
    base_path: Path = INPUT_PATH / "image-matching-challenge-2024"
    feature_dir: Path = WORKING_PATH / "feature_outputs"

    device: torch.device = K.utils.get_cuda_device_if_available(0)

    pair_matching_args = {
        "model_name": str(INPUT_PATH) + "/dinov2/pytorch/base/1",
        "similarity_threshold": 0.6,
        "tolerance": 500,
        "min_matches": 100,
        "exhaustive_if_less": 50,
        "p": 2.0,
    }

    keypoint_detection_args = {
        "num_features": 4096,
        "resize_to": 1024,
    }

    keypoint_distances_args = {
        "min_matches": 15,
        "verbose": False,
    }

    colmap_mapper_options = {
        "min_model_size": 3,
        # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
        "max_num_models": 2,
    }