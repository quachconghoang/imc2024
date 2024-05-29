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

# CUSTOM_PATH = INPUT_PATH / 'imc24-custom'
CUSTOM_PATH = WORKING_PATH / 'imc24-custom'

MIN_SAMPLES = 221
PERCENT=0.33
white_list =[
    # 'church',
    'dioscuri',
    # 'lizard',
    # 'multi-temporal-temple-baalshamin',
    # 'pond',
    # 'transp_obj_glass_cup',
    # 'transp_obj_glass_cylinder'
             ]

def image_test_path_gen(row):
    row['image_path'] = 'test/' + row['dataset'] + '/images/' + row['image_name']
    return row

train_df = pd.read_csv(CUSTOM_PATH / 'test/test_labels.csv')
train_df = train_df.apply(image_test_path_gen, axis=1).drop_duplicates(subset=['image_path'])

# categories = pd.read_csv(data_path / 'train/categories.csv')
G = train_df.groupby(['dataset', 'scene'])['image_path']
image_paths = []
for g in G:
    dataset_name = g[0][0]
    if not dataset_name in white_list:
        print('Skip: ', dataset_name)
        continue
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
pred_df.to_csv(WORKING_PATH/'test_gt.csv', index=False)

### dupilcate pred_df to create sample_submission
sub_df = pred_df.copy()
### generate indentity rotation matrix and zero translation vector
dump_rotation = np.eye(3,3).flatten()
dump_translation = np.zeros(3)
sub_df['rotation_matrix'] = [f"{';'.join([str(x) for x in dump_rotation])}"] * len(sub_df)
sub_df['translation_vector'] = [f"{';'.join([str(x) for x in dump_translation])}"] * len(sub_df)

sub_df.to_csv(WORKING_PATH/'sample_submission.csv', index=False)
