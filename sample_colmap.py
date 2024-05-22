import shutil, os
from pathlib import Path
import pandas as pd
import numpy as np
import glob

from config import *
from utils import *

src = DATA_PATH
if LOCAL_DEBUG:
    src = DATA_PATH_FAKE

# Get data from csv.
data_dict = {}
with open(f'{src}/sample_submission.csv', 'r') as f:
    for i, l in enumerate(f):
        # Skip header.
        if l and i > 0:
            image, dataset, scene, _, _ = l.strip().split(',')
            if dataset not in data_dict:
                data_dict[dataset] = {}
            if scene not in data_dict[dataset]:
                data_dict[dataset][scene] = []
            data_dict[dataset][scene].append(image)

catFile_list = glob.glob(f'{src}/*/categories.csv')
categories_dict ={}
for catFile in catFile_list:
    with (open(catFile, 'r') as f):
        for i, l in enumerate(f):
            if l and i > 0:
                scene, categories = l.strip().split(',')
                if scene not in data_dict:
                    categories_dict[scene] = []
                categories_dict[scene] = categories.strip().split(';')

UNKOWN_CAT = False
if len(catFile_list) > 2: UNKOWN_CAT = True
if len(categories_dict.keys()) > 7: UNKOWN_CAT = True

# with (open(f'{src}/test/categories.csv', 'r') as f):
#     for i, l in enumerate(f):
#         if l and i > 0:
#             scene, categories = l.strip().split(',')
#             if scene not in data_dict:
#                 categories_dict[scene] = []
#             categories_dict[scene] = categories.strip().split(';')


for dataset in data_dict:
    for scene in data_dict[dataset]:
        print(f'{dataset} / {scene} -> {len(data_dict[dataset][scene])} images')

datasets = []
for dataset in data_dict:
    datasets.append(dataset)


out_results = {}
timings = {
    "rotation_detection" : [],
    "shortlisting":[],
   "feature_detection": [],
   "feature_matching":[],
   "RANSAC": [],
   "Reconstruction": []
}

white_list =['transp_obj_glass_cup', 'transp_obj_glass_cylinder',
 'multi-temporal-temple-baalshamin', 'lizard', 'pond', 'church','dioscuri']
pinhole_list = ['church', 'lizard', 'pond']

def check_config(scene_cat):
    # DEFAULT CONFIG
    CONFIG.CAMERA_MODEL = "simple-pinhole"

    is_aliasing = False
    if 'symmetries-and-repeats' in scene_cat:
        is_aliasing = True

    if 'historical_preservation' in scene_cat:
        ...

    if 'transparent' in scene_cat:
        CONFIG.CAMERA_MODEL = "simple-radial"


for dataset in datasets:
        print(dataset)
        if dataset not in out_results:
            out_results[dataset] = {}
        for scene in data_dict[dataset]:
            scene_categories = categories_dict[scene]
            print(scene, categories_dict[scene])
            # Fail gently if the notebook has not been submitted and the test data is not populated.
            # You may want to run this on the training data in that case?
            img_dir = f'{src}/test/{dataset}/images'
            if not os.path.exists(img_dir):
                continue