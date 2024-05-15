# General utilities
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
from pathlib import Path
from time import time, sleep
from fastprogress import progress_bar
import gc
import numpy as np
import h5py
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy
from typing import Any
import itertools
import pandas as pd

# CV/MLe
import cv2
import torch
from torch import Tensor as T
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import torch
from lightglue import match_pair
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

# 3D reconstruction
import pycolmap

# Provided by organizers
from utils.database import *
from utils.h5_to_db import *

def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])

def load_torch_image(file_name: Path | str, device=torch.device("cpu")):
    """Loads an image and adds batch dimension"""
    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

device = K.utils.get_cuda_device_if_available(0)
print(device)

DEBUG = len([p for p in Path("/kaggle/input/image-matching-challenge-2024/test/").iterdir() if p.is_dir()]) == 2
print("DEBUG:", DEBUG)


def embed_images(
        paths: list[Path],
        model_name: str,
        device: torch.device = torch.device("cpu"),
) -> T:
    """Computes image embeddings.
    Returns a tensor of shape [len(filenames), output_dim]
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    embeddings = []
    for i, path in tqdm(enumerate(paths), desc="Global descriptors"):
        image = load_torch_image(path)
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)  # last_hidden_state and pooled
            # Max pooling over all the hidden states but the first (starting token)
            # To obtain a tensor of shape [1, output_dim]
            # We normalize so that distances are computed in a better fashion later
            embedding = F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=-1, p=2)
        embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings, dim=0)


def get_pairs_exhaustive(lst: list[Any]) -> list[tuple[int, int]]:
    """Obtains all possible index pairs of a list"""
    return list(itertools.combinations(range(len(lst)), 2))

def get_image_pairs(
        paths: list[Path],
        model_name: str,
        similarity_threshold: float = 0.6,
        tolerance: int = 1000,
        min_matches: int = 20,
        exhaustive_if_less: int = 20,
        p: float = 2.0,
        device: torch.device = torch.device("cpu"),
) -> list[tuple[int, int]]:
    """Obtains pairs of similar images"""
    if len(paths) <= exhaustive_if_less:
        return get_pairs_exhaustive(paths)

    matches = []

    # Embed images and compute distances for filtering
    embeddings = embed_images(paths, model_name)
    distances = torch.cdist(embeddings, embeddings, p=p)
    # Remove pairs above similarity threshold (if enough)
    mask = distances <= similarity_threshold
    image_indices = np.arange(len(paths))
    for current_image_index in range(len(paths)):
        mask_row = mask[current_image_index]
        indices_to_match = image_indices[mask_row]
        # We don't have enough matches below the threshold, we pick most similar ones
        if len(indices_to_match) < min_matches:
            indices_to_match = np.argsort(distances[current_image_index])[:min_matches]
        for other_image_index in indices_to_match:
            # Skip an image matching itself
            if other_image_index == current_image_index:
                continue

            # We need to check if we are below a certain distance tolerance
            # since for images that don't have enough matches, we picked
            # the most similar ones (which could all still be very different
            # to the image we are analyzing)
            if distances[current_image_index, other_image_index] < tolerance:
                # Add the pair in a sorted manner to avoid redundancy
                matches.append(tuple(sorted((current_image_index, other_image_index.item()))))

    return sorted(list(set(matches)))

# https://www.kaggle.com/code/hoangqc/imc-baseline

if DEBUG:
    images_list = list(Path("/kaggle/input/image-matching-challenge-2024/test/church/images/").glob("*.png"))[:10]
    index_pairs = get_image_pairs(images_list, "/kaggle/input/dinov2/pytorch/base/1")
    print(index_pairs)