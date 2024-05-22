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
DATA_PATH_FAKE = WORKING_PATH / 'imc24'

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

    # COLMAP Reconstruction
    CAMERA_MODEL = "simple-radial"
    # CAMERA_MODEL = "simple-pinhole"

    # Rotation correction
    ROTATION_CORRECTION = True

    # scene_check
    scene_aliasing = False

    # Keypoints Extraction
    use_aliked_lightglue = True
    use_doghardnet_lightglue = False
    use_superpoint_lightglue = False
    use_disk_lightglue = False
    use_sift_lightglue = False
    use_loftr = False
    use_dkm = False
    use_superglue = False
    use_matchformer = False

    # Keypoints Extraction Parameters
    params_aliked_lightglue = {
        "num_features": 4096,
        "detection_threshold": 0.001,
        "min_matches": 15,
        "resize_to": 1024,
    }

    params_doghardnet_lightglue = {
        "num_features": 4096,
        "detection_threshold": 0.001,
        "min_matches": 15,
        "resize_to": 1024,
    }

    params_superpoint_lightglue = {
        "num_features": 4096,
        "detection_threshold": 0.005,
        "min_matches": 15,
        "resize_to": 1024,
    }

    params_disk_lightglue = {
        "num_features": 8192,
        "detection_threshold": 0.001,
        "min_matches": 15,
        "resize_to": 1024,
    }

    params_sift_lightglue = {
        "num_features": 4096,
        "detection_threshold": 0.001,
        "min_matches": 15,
        "resize_to": 1024,
    }

    params_loftr = {
        "resize_small_edge_to": 750,
        "min_matches": 15,
    }

    params_dkm = {
        "num_features": 2048,
        "detection_threshold": 0.4,
        "min_matches": 15,
        "resize_to": (540, 720),
    }

    # superpoint + superglue  ...  https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/416873
    params_sg1 = {
        "sg_config":
            {
                "superpoint": {
                    "nms_radius": 4,
                    "keypoint_threshold": 0.005,
                    "max_keypoints": -1,
                },
                "superglue": {
                    "weights": "outdoor",
                    "sinkhorn_iterations": 20,
                    "match_threshold": 0.2,
                },
            },
        "resize_to": 1088,
        "min_matches": 15,
    }
    params_sg2 = {
        "sg_config":
            {
                "superpoint": {
                    "nms_radius": 4,
                    "keypoint_threshold": 0.005,
                    "max_keypoints": -1,
                },
                "superglue": {
                    "weights": "outdoor",
                    "sinkhorn_iterations": 20,
                    "match_threshold": 0.2,
                },
            },
        "resize_to": 1280,
        "min_matches": 15,
    }
    params_sg3 = {
        "sg_config":
            {
                "superpoint": {
                    "nms_radius": 4,
                    "keypoint_threshold": 0.005,
                    "max_keypoints": -1,
                },
                "superglue": {
                    "weights": "outdoor",
                    "sinkhorn_iterations": 20,
                    "match_threshold": 0.2,
                },
            },
        "resize_to": 1376,
        "min_matches": 15,
    }
    params_sgs = [params_sg1, params_sg2, params_sg3]

    params_matchformer = {
        "detection_threshold": 0.15,
        "resize_to": (560, 750),
        "num_features": 2000,
        "min_matches": 15,
    }