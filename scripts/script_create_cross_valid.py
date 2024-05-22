import os
from pathlib import Path
import pandas as pd
import numpy as np
from glob import glob

root_path=Path('/')
if os.getenv('LOCAL_DATASETS'):
    root_path = Path(os.getenv('LOCAL_DATASETS'))

INPUT_PATH =  root_path / 'kaggle' / 'input'
WORKING_PATH = root_path / 'kaggle' / 'working'
MIN_SAMPLES = 50
PERCENT=0.45

def image_test_path_gen(row):
    row['image_path'] = 'test/' + row['dataset'] + '/images/' + row['image_name']
    return row

train_df = pd.read_csv(WORKING_PATH / 'test/test_labels.csv')
train_df = train_df.apply(image_test_path_gen, axis=1).drop_duplicates(subset=['image_path'])

# categories = pd.read_csv(data_path / 'train/categories.csv')
G = train_df.groupby(['dataset', 'scene'])['image_path']
image_paths = []
for g in G:
    n = MIN_SAMPLES
    if len(g[1]) > 120:
        n = int(PERCENT * len(g[1]))
    if len(g[1]) < MIN_SAMPLES:
        n = len(g[1])

    n = n if n < len(g[1]) else len(g[1])  # If less than 50 -> Full DB
    g = g[0], g[1].sample(n, random_state=42).reset_index(drop=True)
    for image_path in g[1]:
        image_paths.append(image_path)

gt_df = train_df[train_df.image_path.isin(image_paths)].reset_index(drop=True)
pred_df = gt_df[['image_path', 'dataset', 'scene', 'rotation_matrix', 'translation_vector']]
pred_df.to_csv(WORKING_PATH/'sample_submission.csv', index=False)