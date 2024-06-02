import shutil, os, time
from pathlib import Path
import pandas as pd
import numpy as np
import glob
import gc

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
from match_database import CustomDB

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
    fname_to_id = add_keypoints(db, feature_dir, path1, path2, orin_image_name, "", "simple-radial", single_camera)
    add_matches(db, feature_dir, fname_to_id,)
    db.commit()

def get_embedded_dict(_input_dict:dict):
    test_embeddings_dict = {}
    for dataset in _input_dict:
        print(dataset)
        for scene in _input_dict[dataset]:
            print(scene)
            img_dir = os.path.join(CONFIG.base_path, '/'.join(str(_input_dict[dataset][scene][0]).split('/')[:-1]))
            print(img_dir)
            try:
                img_fnames = [os.path.join(CONFIG.base_path, x) for x in _input_dict[dataset][scene]]
                print(f"Got {len(img_fnames)} images")
                scene_embeddings = embed_images(paths=img_fnames, model_name=CONFIG.embed_model, device=CONFIG.device)
                test_embeddings_dict.update({dataset: {scene: scene_embeddings}})

            except Exception as e:
                print(e)
                pass

    return test_embeddings_dict

def parse_sample_submission2(data_path:Path) -> dict[dict[str, list[Path]]]:
    data_dict = {}
    with open(data_path, "r") as f:
        for i, l in enumerate(f):
            # Skip header
            if i == 0:
                print("header:", l)
            if l and i > 0:
                image_path, dataset, scene, _, _ = l.strip().split(',')
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                if image_path in orin_image_path:
                    base_path = IMC_PATH
                else:
                    base_path = WORKING_PATH
                data_dict[dataset][scene].append(Path(base_path / image_path))

    for dataset in data_dict:
        for scene in data_dict[dataset]:
            print(f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images")

    return data_dict

### check working folder
# home_folter = WORKING_PATH
new_test_fold = WORKING_PATH /'test'
old_test_fold = IMC_PATH / 'test'

if LOCAL_DEBUG:
    old_test_fold = CUSTOM_PATH / 'test'
if os.path.exists(CONFIG.feature_dir):
    shutil.rmtree(CONFIG.feature_dir)
if os.path.exists(new_test_fold):
    shutil.rmtree(new_test_fold)
os.makedirs(new_test_fold)


# list_scene_test = [p for p in Path(old_test_fold).iterdir() if p.is_dir()]

### check submission
sample_path = IMC_PATH/'sample_submission.csv'
if LOCAL_DEBUG:
    # copy to new place
    shutil.copy(CUSTOM_PATH/'sample_submission.csv', WORKING_PATH/'sample_submission.csv')
    shutil.copy(CUSTOM_PATH / 'test_gt.csv', WORKING_PATH / 'test_gt.csv')
    sample_path = WORKING_PATH/'sample_submission.csv'
else:
    shutil.copy(IMC_PATH / 'sample_submission.csv', WORKING_PATH / 'sample_submission.csv')


test_dict = parse_sample_submission(sample_path)
list_scene_test = []
list_scene_test_custom = []
for dataset in test_dict:
    list_scene_test.append( dataset)

db_custom = CustomDB()
db_custom.createFromPath(db_custom_path=CUSTOM_TRAIN)
test_embeddings_dict = get_embedded_dict(test_dict)

for dataset in test_dict:
    for scene in test_dict[dataset]:
        test_embedding = test_embeddings_dict[dataset][scene]
        match_dataset_custom = db_custom.checkDesc(test_embedding)
        # print(dataset, ' : ', match_dataset_custom)
        list_scene_test_custom.append(match_dataset_custom)

# Modify database
user_df = pd.read_csv(sample_path)
orin_image_path = user_df['image_path'].to_list()

custom_scenes = ['church', 'dioscuri',
                'lizard-day', 'lizard-night', 'lizard-winter',
                'pond-day', 'pond-night', 'temple',
                'transp_obj_glass_cup', 'transp_obj_glass_cylinder'
                 ]

for id,scene in enumerate(list_scene_test):
    print(scene, '---> CUSTOM: ', list_scene_test_custom[id])
    scene_train = list_scene_test_custom[id]
    scene_train_path = CUSTOM_TRAIN / scene_train
    save_new_test = WORKING_PATH / 'test' / scene / 'images'
    if os.path.exists(scene_train_path):
        os.makedirs(save_new_test, exist_ok=True)
        images_training = os.listdir(scene_train_path / "images")
        ### check submission
        images_test = user_df[user_df['scene'] == scene]['image_path'].to_list()
        images_test = [os.path.basename(x) for x in images_test]
        print(images_test)
        #Default
        add_imgs = []
        if scene_train == 'church':
            num_add = 80
            add_imgs = np.random.choice(images_training, num_add)
            add_imgs =list(set(add_imgs))

        if scene_train == 'temple':
            add_imgs = images_training

        if scene_train == 'dioscuri':
            num_add = 100
            add_imgs = np.random.choice(images_training, num_add)
            add_imgs =list(set(add_imgs))
            # add_imgs = images_training

        if scene_train == ['lizard-day', 'lizard-winter', 'pond-day']:
            num_add = 50
            add_imgs = np.random.choice(images_training, num_add)
            add_imgs = list(set(add_imgs))

        if scene_train == ['lizard-night', 'pond-night']:
            num_add = 100
            add_imgs = np.random.choice(images_training, num_add)
            add_imgs = list(set(add_imgs))

        if scene_train == 'UNKNOWN':
            add_imgs = []

        for images in add_imgs:
            src = os.path.join(str(scene_train_path), "images", images)
            dst = os.path.join(str(save_new_test), images)
            image_path = dst.replace(str(WORKING_PATH)+'/', '')
            # print(image_path)
            if image_path in orin_image_path:
                # print('duplicated!')
                continue
            try:
                shutil.copy(src, dst)
                new_row = pd.Series({   'image_path': image_path,
                                        'dataset': os.path.basename(scene),
                                        'scene': os.path.basename(scene),
                                        "rotation_matrix": "",
                                        "translation_vector": ""}   )
                user_df.loc[len(user_df)] = new_row
            except:
                print("cannot create new")

print("len after", len(user_df))

for dirpath, dirnames, filenames in os.walk(new_test_fold):
    # Skip the root directory
    if dirpath == new_test_fold:
        continue

###
user_df.to_csv(WORKING_PATH / "sample_submission2.csv", index=False, index_label=False)

data_dict = parse_sample_submission2(WORKING_PATH / "sample_submission2.csv")
datasets = list(data_dict.keys())
results = {}
for dataset in datasets:
    if dataset not in results:
        results[dataset] = {}

    for scene in data_dict[dataset]:
        images_dir = data_dict[dataset][scene][0].parent
        results[dataset][scene] = {}
        image_paths = data_dict[dataset][scene]
        print (f"---> NEW : {scene}: Got {len(image_paths)} images")

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
            gc.collect()

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
            gc.collect()

            # 4.1. Import keypoint distances of matches into colmap for RANSAC
            import_into_colmap(
                images_dir,
                feature_dir,
                database_path,
            )

            output_path = feature_dir / "colmap_rec_aliked"
            output_path.mkdir(parents=True, exist_ok=True)

            # 4.2. Compute RANSAC (detect match outliers)
            # By doing it exhaustively we guarantee we will find the best possible Configuration
            pycolmap.match_exhaustive(database_path)

            mapper_options = pycolmap.IncrementalPipelineOptions(**CONFIG.colmap_mapper_options)

            # 5.1 Incrementally start reconstructing the scene (sparse reconstruction)
            # The process starts from a random pair of images and is incrementally extended by
            # registering new images and triangulating new points.
            maps = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=images_dir,
                output_path=output_path,
                options=mapper_options,
            )

            # print(maps)
            clear_output(wait=False)

            # 5.2. Look for the best reconstruction: The incremental mapping offered by
            # pycolmap attempts to reconstruct multiple models, we must pick the best one
            images_registered = 0
            best_idx = None

            print("Looking for the best reconstruction")

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