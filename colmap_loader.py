import shutil, os
from pathlib import Path
import pandas as pd
import numpy as np

root_path=""
if os.getenv('LOCAL_DATASETS'):
    root_path = os.getenv('LOCAL_DATASETS')

home_folder = root_path+"/kaggle/working/"
new_test_fold = home_folder + "test/"

old_test_fold = root_path+'/kaggle/input/image-matching-challenge-2024/test/'

if os.path.exists(new_test_fold):
    shutil.rmtree(new_test_fold)
os.makedirs(new_test_fold)

list_scene_test = [p for p in Path(old_test_fold).iterdir() if p.is_dir()]

user_df = pd.read_csv(root_path+'/kaggle/input/image-matching-challenge-2024/sample_submission.csv')
orin_image_path = user_df['image_path'].to_list()
print("len sbmission: ", len(user_df))

for scene in list_scene_test:
    scene = str(scene)
    scene_train = os.path.join(root_path+"/kaggle/input/image-matching-challenge-2024/train/", os.path.basename(scene))
    save_new_test = os.path.join(new_test_fold, os.path.basename(scene), "images")
    if os.path.exists(scene_train):
        os.makedirs(save_new_test, exist_ok = True)
        images_training = os.listdir(scene_train+"/images")
        images_test = os.listdir(scene+"/images")
        num_add = min(50, len(images_test))
        add_imgs = np.random.choice(images_training, num_add)
        # add_imgs = np.random.choice(images_training, 2)

        for images in add_imgs:
            src = os.path.join(scene_train, "images", images)
            dst = os.path.join(save_new_test, images)
            image_path = dst.replace(home_folder, "")
            if image_path in user_df['image_path'].to_list():
                continue
            try:
                shutil.copy(src, dst)
                new_row = pd.Series({'image_path': image_path, 'dataset': os.path.basename(scene), 'scene': os.path.basename(scene), "rotation_matrix":"", "translation_vector":""})
                user_df.loc[len(user_df)] = new_row
            except:
                print("cannot create new")

print("len after", len(user_df))

for dirpath, dirnames, filenames in os.walk(new_test_fold):
    # Skip the root directory
    if dirpath == new_test_fold:
        continue

    num_files = len(filenames)
    print(f"Folder: {os.path.relpath(dirpath, new_test_fold)} - Number of files: {num_files}")
user_df.to_csv(root_path+"/kaggle/working/sample_submission.csv", index=False, index_label=False)

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
from utils.database import *
from utils.h5_to_db import *
from pathlib import Path
import os

test_path = "/kaggle/input/image-matching-challenge-2024/test/"
root_path = ""
DEBUG = True

if os.getenv('LOCAL_DATASETS'):
    root_path = os.getenv('LOCAL_DATASETS')
test_dir = Path(root_path+test_path)

