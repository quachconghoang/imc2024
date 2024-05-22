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
DATA_PATH = INPUT_PATH / 'image-matching-challenge-2024'
# DATA_PATH_FAKE = WORKING_PATH / 'imc24'

def image_path_gen(row):
    row['image_path'] = 'train/' + row['dataset'] + '/images/' + row['image_name']
    return row


train_df = pd.read_csv(DATA_PATH / 'train/train_labels.csv')
test_df = train_df.apply(image_path_gen, axis=1).drop_duplicates(subset=['image_path'])

# check image path and remove if not available in test_df
img_paths = test_df.image_path.to_list()
for img_path in img_paths:
    if not os.path.exists(DATA_PATH / img_path):
        test_df = test_df[test_df.image_path != img_path]
        print('Not Found - Removed: ', DATA_PATH / img_path)


img_paths = test_df.image_path.to_list()
for img_path in img_paths:
    # print(img_path)
    os.makedirs(WORKING_PATH / 'test-full' / os.path.dirname(img_path), exist_ok=True)
    os.system('cp ' + str(DATA_PATH / img_path) + ' ' + str(WORKING_PATH / 'test-full' / img_path))

### remove image_path field in test_df
test_df_to_save = test_df.drop(columns=['image_path'])
os.makedirs(WORKING_PATH / 'test-full', exist_ok=True)
test_df_to_save.to_csv(WORKING_PATH / 'test-full' / 'test_labels.csv', index=False)

