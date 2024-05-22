from pathlib import Path
import os
import matplotlib.pyplot as plt
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_exhaustive
)

root_path=''
if os.getenv('LOCAL_DATASETS'):
    root_path = os.getenv('LOCAL_DATASETS')

working_path = root_path+'/kaggle/working/train/'

scenes = ['church', 'dioscuri', 'temple',
          'lizard-day', 'lizard-night', 'lizard-winter',
          'pond-day', 'pond-night']

for scene_name in scenes:
    images = Path(working_path + scene_name + '/images')
    outputs = Path(working_path + scene_name + '/hloc/disk')
    sfm_pairs = outputs / 'pairs-eigenplaces.txt'
    sfm_dir = outputs / 'sfm'
    matches = outputs / 'matches.h5'
    loc_pairs = outputs / 'pairs-loc.txt'
    log_registration = outputs / 'log.txt'

    retrieval_conf = extract_features.confs['eigenplaces']
    feature_conf = extract_features.confs['disk']
    matcher_conf = match_features.confs['disk+lightglue']

    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=50)

    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], outputs)

    model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

    references_registered = [model.images[i].name for i in model.reg_image_ids()]
    _unreg = [i for i in os.listdir(images) if i not in references_registered]

    #Write below text to log.txt
    with open(log_registration, 'w') as f:
        f.write(f'SCENE_NAME: {scene_name}\n')
        f.write(f'num_reg_images: {len(references_registered)}\n')
        f.write(f'num_cameras: {len(model.images)}\n')
        f.write(f'num_points3D: {len(model.points3D)}\n')
        f.write(f'num_observations: {model.compute_num_observations()}\n')
        f.write(f'mean_track_length: {model.compute_mean_track_length()}\n')
        f.write(f'mean_observations_per_image: {model.compute_mean_observations_per_reg_image()}\n')
        f.write(f'mean_reprojection_error: {model.compute_mean_reprojection_error()}\n')
        f.write(f'num_input_images: {len(os.listdir(images))}\n')
        f.write(f'unregistered: {_unreg}\n')

    # print('-----', scene_name, '-----')
    # print('Registered: \n', references_registered)
    # print('UnReg: \n', _unreg)



# DISK
# num_reg_images = 107
# num_cameras = 107
# num_points3D = 38501
# num_observations = 293705
# mean_track_length = 7.6285
# mean_observations_per_image = 2744.91
# mean_reprojection_error = 0.958783
# num_input_images = 111

# images_test = Path(working_path+"/church_test/images")
# references = os.listdir(images)
# query = '00037.png'
# queries = ['00035.png','00037.png','00039.png','00050.png','00036.png']

# references_registered = [model.images[i].name for i in model.reg_image_ids()]
# extract_features.main(feature_conf, images_test, image_list=queries, feature_path=feature_path, overwrite=True)
# pairs_from_exhaustive.main(loc_pairs, image_list=queries, ref_list=references)
# match_features.main(matcher_conf, loc_pairs, features=feature_path, matches=matches, overwrite=True);
#
# qID=3
# camera = pycolmap.infer_camera_from_image(images_test / queries[qID])
# ref_ids = [model.find_image_with_name(n).image_id for n in references_registered]
# conf = {
#     'estimation': {'ransac': {'max_error': 12}},
#     'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
# }
# localizer = QueryLocalizer(model, conf)
# ret, log = pose_from_cluster(localizer, queries[qID], camera, ref_ids, feature_path, matches)
#
# print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
# visualization.visualize_loc_from_log(images, queries[qID], log, model)
# plt.show()

