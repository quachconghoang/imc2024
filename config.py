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

LOCAL_DEBUG = True
SUBMISSION = False

IMC_PATH = INPUT_PATH / 'image-matching-challenge-2024'
CUSTOM_PATH = INPUT_PATH / 'imc24-custom'
CUSTOM_TRAIN = INPUT_PATH / 'imc24-custom' / 'train'

# if LOCAL_DEBUG:
    # IMC_PATH = WORKING_PATH / 'imc24-custom'
    # CUSTOM_PATH = WORKING_PATH / 'imc24-custom' / 'train'

# device = K.utils.get_cuda_device_if_available(0)
DEVICE: torch.device = K.utils.get_cuda_device_if_available(0)
print(DEVICE)



class CONFIG:
    base_path: Path = IMC_PATH
    feature_dir: Path = WORKING_PATH / "feature_outputs"
    # feature_dir: Path = Path.cwd() / ".feature_outputs"
    device: torch.device = K.utils.get_cuda_device_if_available(0)
    embed_model = str(INPUT_PATH) + "/dinov2/pytorch/base/1"

    pair_matching_args = {
        "model_name": str(INPUT_PATH) + "/dinov2/pytorch/base/1",
        "similarity_threshold": 0.5,
        "tolerance": 500,
        "min_matches": 100,
        "exhaustive_if_less": 20,
        "p": 2.0,
    }

    # COLMAP Reconstruction
    CAMERA_MODEL = "simple-radial"
    # CAMERA_MODEL = "simple-pinhole"
    # Rotation correction
    ROTATION_CORRECTION = True
    # scene_check
    scene_aliasing = False
    # Keypoints Extraction Parameters

    keypoint_detection_args = {
        "num_features": 8192,
        "resize_to": 1024,
    }

    keypoint_distances_args = {
        "min_matches": 15,
        "verbose": False,
    }

    colmap_mapper_options = {
        "min_model_size": 3, # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
        "max_num_models": 2,
    }

    # params_aliked_lightglue = {
    #     "num_features": 4096,
    #     "detection_threshold": 0.001,
    #     "min_matches": 15,
    #     "resize_to": 1024,
    # }
    # params_superpoint_lightglue = {
    #     "num_features": 4096,
    #     "detection_threshold": 0.005,
    #     "min_matches": 15,
    #     "resize_to": 1024,
    # }