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

root_path=""
if os.getenv('LOCAL_DATASETS'):
    root_path = os.getenv('LOCAL_DATASETS')

working_path = root_path+"/kaggle/working/imc-data/"

images = Path(working_path+"/church/images")
outputs = Path(working_path+"/church/hloc")
sfm_pairs = outputs / "pairs-netvlad.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"
matches = outputs / "matches.h5"
loc_pairs = outputs / "pairs-loc.txt"


retrieval_conf = extract_features.confs["netvlad"]
feature_conf = extract_features.confs["superpoint_aachen"]
matcher_conf = match_features.confs["superglue"]

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], outputs)

model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

images_test = Path(working_path+"/church_test/images")
references = os.listdir(images)
# query = '00037.png'
queries = ['00035.png','00037.png','00039.png','00050.png']

references_registered = [model.images[i].name for i in model.reg_image_ids()]
extract_features.main(feature_conf, images_test, image_list=queries, feature_path=feature_path, overwrite=True)
pairs_from_exhaustive.main(loc_pairs, image_list=queries, ref_list=references)
match_features.main(matcher_conf, loc_pairs, features=feature_path, matches=matches, overwrite=True);

qID=1
camera = pycolmap.infer_camera_from_image(images_test / queries[qID])
ref_ids = [model.find_image_with_name(n).image_id for n in references_registered]
conf = {
    'estimation': {'ransac': {'max_error': 12}},
    'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
}
localizer = QueryLocalizer(model, conf)
ret, log = pose_from_cluster(localizer, queries[qID], camera, ref_ids, feature_path, matches)

print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
visualization.visualize_loc_from_log(images, queries[qID], log, model)
plt.show()