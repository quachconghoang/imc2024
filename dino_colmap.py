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
from database import *
from h5_to_db import *

def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])

def load_torch_image(file_name: Path | str, device=torch.device("cpu")):
    """Loads an image and adds batch dimension"""
    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

device = K.utils.get_cuda_device_if_available(0)
print(device)


root_path=""
if os.getenv('LOCAL_DATASETS'):
    root_path = os.getenv('LOCAL_DATASETS')
working_path = root_path+"/kaggle/working/imc-data/"
imgDB = Path(working_path+"/church/images")
img_test = Path(working_path+"/church_test/images")

DEBUG = len([p for p in Path(root_path+"/kaggle/input/image-matching-challenge-2024/test/").iterdir() if p.is_dir()]) == 2
print("DEBUG:", DEBUG)
# DEBUG = len([p for p in Path(working_path).iterdir() if p.is_dir()]) == 2
# print("DEBUG:", DEBUG)

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
        similarity_threshold: float = 1,
        tolerance: int = 1000,
        min_matches: int = 20,
        exhaustive_if_less: int = 50,
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

images_list = list(imgDB.glob("*.png"))
embeddings = embed_images(images_list, "./kaggle/input/dinov2/pytorch/base/1", device=device)

# from config import Config
# index_pairs = get_image_pairs(images_list, **Config.pair_matching_args)

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms


def get_global_desc(fnames, model,
                    device=torch.device('cpu')):
    model = model.eval()
    model = model.to(device)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    global_descs_convnext = []
    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        img = Image.open(img_fname_full).convert('RGB')
        timg = transform(img).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            desc = model.forward_features(timg.to(device)).mean(dim=(-1, 2))  #
            # print (desc.shape)
            desc = desc.view(1, -1)
            desc_norm = F.normalize(desc, dim=1, p=2)
        # print (desc_norm)
        global_descs_convnext.append(desc_norm.detach().cpu())
    global_descs_all = torch.cat(global_descs_convnext, dim=0)
    return global_descs_all.to(torch.float32)


def convert_1d_to_2d(idx, num_images):
    idx1 = idx // num_images
    idx2 = idx % num_images
    return (idx1, idx2)


def get_pairs_from_distancematrix(mat):
    pairs = [convert_1d_to_2d(idx, mat.shape[0]) for idx in np.argsort(mat.flatten())]
    pairs = [pair for pair in pairs if pair[0] < pair[1]]
    return pairs


def get_img_pairs_exhaustive(img_fnames, model, device):
    # index_pairs = []
    # for i in range(len(img_fnames)):
    #    for j in range(i+1, len(img_fnames)):
    #        index_pairs.append((i,j))
    # return index_pairs
    descs = get_global_desc(img_fnames, model, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    matching_list = get_pairs_from_distancematrix(dm)
    return matching_list


def get_image_pairs_shortlist(fnames,
                              sim_th=0.5,  # should be strict
                              min_pairs=20,
                              exhaustive_if_less=20,
                              device=torch.device('cpu')):
    num_imgs = len(fnames)

    model = timm.create_model('tf_efficientnet_b7',
                              checkpoint_path='./kaggle/input/tf_efficientnet_b7_ra-6c08e654.pth')
    model.eval()
    descs = get_global_desc(fnames, model, device=device)

    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames, model, device)

    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs - 1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total += 1
    matching_list = sorted(list(set(matching_list)))
    return matching_list

img_test = Path(working_path+"/church_test/images")
images_list = list(img_test.glob("*.png"))

index_pairs = get_image_pairs_shortlist(images_list,
                                  sim_th = 0.6, # should be strict
                                  min_pairs = 50, # we select at least min_pairs PER IMAGE with biggest similarity
                                  exhaustive_if_less = 50,
                                  device=device)

model = timm.create_model('tf_efficientnet_b7',
                          checkpoint_path='./kaggle/input/tf_efficientnet_b7_ra-6c08e654.pth')
model.eval()
descs = get_global_desc(images_list, model, device=device)
dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()

# show files in pair
for p in index_pairs:
    print(images_list[p[0]], images_list[p[1]])