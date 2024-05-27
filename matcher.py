from config import *
from utils import load_torch_image, embed_images, get_pairs_exhaustive
from tqdm import tqdm

import cv2 as cv
import kornia as K
import kornia.feature as KF

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
    dtype = torch.float32  # ALIKED has issues with float16

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


def keypoint_distances(
        paths: list[Path],
        index_pairs: list[tuple[int, int]],
        feature_dir: Path,
        min_matches: int = 15,
        verbose: bool = True,
        device: torch.device = torch.device("cpu"),
) -> None:
    """Computes distances between keypoints of images.
    Stores output at feature_dir/matches.h5
    """
    matcher_params = {
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True if 'cuda' in str(device) else False,
    }
    matcher = KF.LightGlueMatcher("aliked", matcher_params).eval().to(device)
    print("computing keypoint_distances", len(paths), "len pair", len(index_pairs))
    with h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints, \
            h5py.File(feature_dir / "descriptors.h5", mode="r") as f_descriptors, \
            h5py.File(feature_dir / "matches.h5", mode="w") as f_matches:

        for idx1, idx2 in tqdm(index_pairs, desc="keypoint_distances"):
            #             for idx1, idx2 in index_pairs:
            key1, key2 = paths[idx1].name, paths[idx2].name

            keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(device)
            keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(device)
            descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(device)
            descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(device)

            with torch.inference_mode():
                distances, indices = matcher(
                    descriptors1,
                    descriptors2,
                    KF.laf_from_center_scale_ori(keypoints1[None]),
                    KF.laf_from_center_scale_ori(keypoints2[None]),
                )

            # We have matches to consider
            n_matches = len(indices)
            if n_matches:
                if verbose:
                    print(f"{key1}-{key2}: {n_matches} matches")
                # Store the matches in the group of one image
                if n_matches >= min_matches:
                    group = f_matches.require_group(key1)
                    group.create_dataset(key2, data=indices.detach().cpu().numpy().reshape(-1, 2))