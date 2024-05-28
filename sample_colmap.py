import shutil, os, time
from pathlib import Path
import pandas as pd
import numpy as np
import glob

from database import *
from h5_to_db import *
from IPython.display import clear_output
from copy import deepcopy

import pycolmap
import torch
from lightglue import match_pair
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

from config import *
from utils import *
from matcher import keypoint_distances, detect_keypoints

def import_into_colmap(
    path: Path,
    feature_dir: Path,
    database_path: str = "colmap.db",
) -> None:
    """Adds keypoints into colmap"""
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    path1 = path
    path2 = Path(str(path1).replace("input/image-matching-challenge-2024", "working"))
    orin_image_name = os.listdir(path1)
    fname_to_id = add_keypoints(db, feature_dir, path1, path2, orin_image_name, "", "simple-pinhole", single_camera)
    add_matches(db, feature_dir, fname_to_id,)
    db.commit()

### check submission
sample_path = IMC_PATH/'sample_submission.csv'
if LOCAL_DEBUG:
    sample_path = WORKING_PATH/'sample_submission.csv'


user_df = pd.read_csv(sample_path)
orin_image_path = user_df['image_path'].to_list()
print("len sbmission: ", len(user_df))

results = {}
data_dict = parse_sample_submission(sample_path)
datasets = list(data_dict.keys())

for dataset in datasets:
    if dataset not in results:
        results[dataset] = {}

    for scene in data_dict[dataset]:
        images_dir = data_dict[dataset][scene][0].parent
        results[dataset][scene] = {}
        image_paths = data_dict[dataset][scene]
        print (f"{scene}: Got {len(image_paths)} images")

        try:
            feature_dir = CONFIG.feature_dir / f"{dataset}_{scene}"
            feature_dir.mkdir(parents=True, exist_ok=True)
            database_path = feature_dir / "colmap.db"
            if database_path.exists():
                database_path.unlink()

            # 1. Get the pairs of images that are somewhat similar
            index_pairs = get_image_pairs(
                image_paths,
                **CONFIG.pair_matching_args,
                device=CONFIG.device,
            )
            # gc.collect()

            # 2. Detect keypoints of all images
            detect_keypoints(
                image_paths,
                feature_dir,
                **CONFIG.keypoint_detection_args,
                device=CONFIG.device,
            )
            # gc.collect()

            # 3. Match  keypoints of pairs of similar images
            keypoint_distances(
                image_paths,
                index_pairs,
                feature_dir,
                **CONFIG.keypoint_distances_args,
                device=CONFIG.device,
            )
            # gc.collect()

            # sleep(1)

            # 4.1. Import keypoint distances of matches into colmap for RANSAC
            import_into_colmap(images_dir, feature_dir, database_path, )
            output_path = feature_dir / "colmap_rec_aliked"
            output_path.mkdir(parents=True, exist_ok=True)

            # 4.2. Compute RANSAC (detect match outliers)
            pycolmap.match_exhaustive(database_path)
            mapper_options = pycolmap.IncrementalPipelineOptions(**CONFIG.colmap_mapper_options)

            # 5.1 Incrementally start reconstructing the scene (sparse reconstruction)
            maps = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=images_dir,
                output_path=output_path,
                options=mapper_options,
            )

#             print(maps)
#             clear_output(wait=False)

            # 5.2. Look for the best reconstruction: The incremental mapping offered by
            # pycolmap attempts to reconstruct multiple models, we must pick the best one
            images_registered  = 0
            best_idx = None

            print ("Looking for the best reconstruction")

            if isinstance(maps, dict):
                for idx1, rec in maps.items():
                    print(idx1, rec.summary())
                    try:
                        if len(rec.images) > images_registered:
                            images_registered = len(rec.images)
                            best_idx = idx1
                    except Exception:
                        continue

            # Parse the reconstruction object to get the rotation matrix and translation vector
            # obtained for each image in the reconstruction
            if best_idx is not None:
                for k, im in maps[best_idx].images.items():
                    key = CONFIG.base_path / "test" / scene / "images" / im.name
                    results[dataset][scene][key] = {}
                    results[dataset][scene][key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
                    results[dataset][scene][key]["t"] = deepcopy(np.array(im.cam_from_world.translation))

            print(f"Registered: {dataset} / {scene} -> {len(results[dataset][scene])} images")
            print(f"Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images")
            create_submission(results, data_dict, CONFIG.base_path, orin_image_path)
            #copy submission to  WORKING_PATH
            shutil.copy('tmp_submission.csv', WORKING_PATH / 'submission.csv')
            # gc.collect()

        except Exception as e:
            print(e)



if LOCAL_DEBUG:
    # shutil.copy(IMC_PATH/'sample_submission.csv', WORKING_PATH/'test_gt.csv')
    trans = np.eye(4)
    trans[:3, -1] = [0, 0, 0]
    gt_csv = WORKING_PATH / 'test_gt.csv'
    user_csv = WORKING_PATH / 'submission.csv'
    gt_df = pd.read_csv(gt_csv).sort_values(by='image_path')
    user_df = pd.read_csv(user_csv)
    # user_df['image_path'] = user_df['image_path'].apply(lambda x: os.path.basename(x))
    # user_df = user_df.head(41)
    user_df = user_df.sort_values(by='image_path')
    start = time.time()
    res = score(gt_df, user_df)
    end = time.time()
    print(f"\nGlobal mAA: {res * 100}%")
    print("Total running time: %s" % (end - start))
