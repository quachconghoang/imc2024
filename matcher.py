from config import *
from utils import load_torch_image, embed_images, get_pairs_exhaustive
import tqdm

import cv2 as cv

from lightglue import match_pair
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

def detect_keypoints(
        paths: list[Path],
        feature_dir: Path,
        num_features: int = 4096,
        resize_to: int = 1024,
        device: torch.device = torch.device("cpu"),
) -> None:
    """Detects the keypoints in a list of images with ALIKED

    Stores them in feature_dir/keypoints.h5 and feature_dir/descriptors.h5
    to be used later with LightGlue
    """
    dtype = torch.float32 # ALIKED has issues with float16

    extractor = ALIKED(
        max_num_keypoints=num_features,
        detection_threshold=0.01,
        resize=resize_to
    ).eval().to(device, dtype)

    feature_dir.mkdir(parents=True, exist_ok=True)
    print("computing detect_keypoints", len(paths))
    with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints, \
            h5py.File(feature_dir / "descriptors.h5", mode="w") as f_descriptors:

        for path in tqdm(paths, desc="detect_keypoints"):
            #         for path in paths:

            key = path.name

            with torch.inference_mode():
                image = load_torch_image(path, device=device).to(dtype)
                features = extractor.extract(image)

                f_keypoints[key] = features["keypoints"].squeeze().detach().cpu().numpy()
                f_descriptors[key] = features["descriptors"].squeeze().detach().cpu().numpy()


def detect_roi(img_names):
    ...
