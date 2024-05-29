from pathlib import Path
import os, shutil, time
import matplotlib.pyplot as plt

from config import *
from utils import *
from copy import deepcopy
from matcher import keypoint_distances, detect_keypoints

from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from pycolmap import Reconstruction
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_exhaustive
)



import torch
from lightglue import match_pair
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

from match_database import CustomDB

sample_path = IMC_PATH/'sample_submission.csv'
if LOCAL_DEBUG:
    sample_path = WORKING_PATH/'sample_submission.csv'

db_custom = CustomDB()
db_custom.createFromPath(db_custom_path=CUSTOM_TRAIN)

test_dict = {}
test_dict = parse_sample_submission(sample_path)
out_results = {}

user_df = pd.read_csv(sample_path)
orin_image_path = user_df['image_path'].to_list()

datasets = []
for dataset in test_dict:
    datasets.append(dataset)

working_path = WORKING_PATH / 'hloc-cache'
scene_name = 'pond-night'
images_dir = working_path / scene_name / 'images'
# queries_dir = working_path / scene_name / 'queries'
outputs = working_path / scene_name / 'hloc' / 'disk'
sfm_pairs = outputs / 'pairs-eigenplaces.txt'
sfm_dir = outputs / 'sfm'
matches = outputs / 'matches.h5'
loc_pairs = outputs / 'pairs-loc.txt'
log_registration = outputs / 'log.txt'

retrieval_conf = extract_features.confs['eigenplaces']
feature_conf = extract_features.confs['disk']
matcher_conf = match_features.confs['disk+lightglue']

feature_path = outputs / 'feats-disk.h5'
match_path = outputs / 'feats-disk_matches-disk-lightglue_pairs-eigenplaces.h5'
model = Reconstruction(sfm_dir)
# model = reconstruction.main(sfm_dir, images_dir, sfm_pairs, feature_path, match_path)

test_embeddings_dict = {}
dataset = 'pond'
scene = 'pond'

### get name in querry list
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

query = [x.name for x in test_dict[dataset][scene]]
img_dir =test_dict[dataset][scene][0].parent

extract_features.main(feature_conf, image_dir=img_dir, image_list=query, feature_path=feature_path, overwrite=False)

references = os.listdir(images_dir)
references_registered = [model.images[i].name for i in model.reg_image_ids()]
pairs_from_exhaustive.main(loc_pairs, image_list=query, ref_list=references)
match_features.main(matcher_conf, loc_pairs, features=feature_path, matches=matches, overwrite=False)

conf = {
    'estimation': {'ransac': {'max_error': 12}},
    'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
}

ref_ids = [model.find_image_with_name(n).image_id for n in references_registered]

results = {}
results[dataset] = {}
results[dataset][scene] = {}

poses = []
for img in query:
    camera = pycolmap.infer_camera_from_image(img_dir / img)
    localizer = QueryLocalizer(model, conf)
    ret, log = pose_from_cluster(localizer, img, camera, ref_ids, feature_path, matches)
    print(img,ret["cam_from_world"])
    key = CONFIG.base_path / "test" / scene / "images" / img
    results[dataset][scene][key] = {}
    results[dataset][scene][key]["R"] = deepcopy(ret["cam_from_world"].rotation.matrix())
    results[dataset][scene][key]["t"] = deepcopy(np.array(ret["cam_from_world"].translation))
    # poses.append(ret["cam_from_world"])

create_submission(results, test_dict, CONFIG.base_path, orin_image_path)
shutil.copy('tmp_submission.csv', WORKING_PATH / 'submission.csv')

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

# test_embeddings_dict = {}
# for dataset in test_dict:
#     print(dataset)
#     if dataset not in out_results:
#         out_results[dataset] = {}
#     for scene in test_dict[dataset]:
#         print(scene)
#         # img_dir = os.path.join(CONFIG.base_path, '/'.join(str(test_dict[dataset][scene][0]).split('/')[:-1]))
#         # print(img_dir)
#         try:
#             out_results[dataset][scene] = {}
#             img_fnames = [os.path.join(CONFIG.base_path, x) for x in test_dict[dataset][scene]]
#             print(f"Got {len(img_fnames)} images")
#             scene_embeddings = embed_images(paths=img_fnames, model_name=CONFIG.embed_model, device=CONFIG.device)
#             test_embeddings_dict.update({dataset: {scene: scene_embeddings}})
#
#         except Exception as e:
#             print(e)
#             pass
#
# for dataset in test_dict:
#     for scene in test_dict[dataset]:
#         test_embedding = test_embeddings_dict[dataset][scene]
#         match_dataset_custom = db_custom.checkDesc(test_embedding)
#         print(dataset, ' : ', match_dataset_custom)


