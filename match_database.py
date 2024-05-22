import os
from pathlib import Path
import pandas as pd
import numpy as np

# HoangQC Import
from config import *
from utils import *

def image_path_gen(row):
    row['image_path'] = 'train/' + row['dataset'] + '/images/' + row['image_name']
    return row

# def getImageTrainingPaths():
# train_df = pd.read_csv(DATA_PATH / 'train/train_labels.csv')
# train_df = train_df.apply(image_path_gen, axis=1).drop_duplicates(subset=['image_path'])
# G = train_df.groupby(['dataset', 'scene'])['image_path']
# image_paths = []
# for g in G:
#     n = g[1]
#     for image_path in g[1]:
#         image_paths.append(image_path)

def createCrossValid(N_SAMPLES = 50, PERCENT=0.33):
    # os.makedirs(outPath)
    train_df = pd.read_csv(DATA_PATH / 'train/train_labels.csv')
    train_df = train_df.apply(image_path_gen, axis=1).drop_duplicates(subset=['image_path'])

    # categories = pd.read_csv(data_path / 'train/categories.csv')
    G = train_df.groupby(['dataset', 'scene'])['image_path']
    image_paths = []
    for g in G:
        n = N_SAMPLES
        if len(g[1]) > 500:
            n = int(PERCENT*len(g[1]))
        if len(g[1]) < N_SAMPLES:
            n = len(g[1])

        n = n if n < len(g[1]) else len(g[1]) # If less than 50 -> Full DB
        g = g[0], g[1].sample(n, random_state=42).reset_index(drop=True)
        for image_path in g[1]:
            image_paths.append(image_path)

    gt_df = train_df[train_df.image_path.isin(image_paths)].reset_index(drop=True)
    pred_df = gt_df[['image_path', 'dataset', 'scene', 'rotation_matrix', 'translation_vector']]
    # pred_df.to_csv('pred_df.csv', index=False)

    return pred_df, image_paths

    # image_path,dataset,scene,rotation_matrix,translation_vector


def generateLocalValidSet(_test_name = 'test'):
    test_df, img_paths = createCrossValid()

    # os.makedirs(WORKING_PATH,exist_ok=True)
    # copy all file in img_paths to ./test/ with subfolders keeping
    for img_path in img_paths:
        print(img_path)
        os.makedirs(WORKING_PATH / os.path.dirname(img_path), exist_ok=True)
        os.system('cp ' + str(DATA_PATH / img_path) + ' ' + str(WORKING_PATH / img_path))

    # change folder ./kaggle/output/train to ./kagge/output/test
    os.rename(str(WORKING_PATH / 'train'), str(WORKING_PATH / _test_name))

    test_df['image_path'] = test_df['image_path'].apply(lambda x: x.replace('train', _test_name))
    test_df.to_csv(WORKING_PATH / _test_name / 'test_gt.csv', index=False)

generateLocalValidSet(_test_name='test_0')
generateLocalValidSet(_test_name='test_1')