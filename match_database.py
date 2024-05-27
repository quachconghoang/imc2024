import os
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy

# HoangQC Import
from config import *
from utils import parse_sample_submission
from utils import embed_images
from sklearn.cluster import KMeans

class CustomDB:
    def __init__(self):
        self.datasets = ['church', 'dioscuri',
                        'lizard-day', 'lizard-night', 'lizard-winter',
                        'pond-day', 'pond-night', 'temple',
                        'transp_obj_glass_cup', 'transp_obj_glass_cylinder']
        self.img_names = {}
        self.img_paths = {}
        self.db_embeddings = {}

    def createFromPath(self, db_custom_path: Path):
        ### check tmp file is available
        if os.path.exists(CUSTOM_PATH / 'db_img_names.npy') & os.path.exists(CUSTOM_PATH / 'embeddings_dict.npy'):
            self.img_names = np.load(db_custom_path / 'db_img_names.npy', allow_pickle=True).item()
            for dataset in self.datasets:
                img_paths = [os.path.join(CUSTOM_PATH, dataset, 'images', x) for x in self.img_names[dataset]]
                self.img_paths[dataset] = img_paths
            self.db_embeddings = np.load(db_custom_path / 'embeddings_dict.npy', allow_pickle=True).item()


        for dataset in self.datasets:
            img_names = os.listdir(os.path.join(db_custom_path, dataset, 'images'))
            img_names.sort()
            self.img_names[dataset] = img_names

        for dataset in self.datasets:
            img_paths = [os.path.join(CUSTOM_PATH, dataset, 'images', x) for x in self.img_names[dataset]]
            self.img_paths[dataset] = img_paths

        for dataset in self.datasets:
            path_list = self.img_paths[dataset]
            embeddings = embed_images(paths=path_list, model_name=CONFIG.embed_model, device=CONFIG.device)
            self.db_embeddings[dataset] = embeddings

        np.save(CUSTOM_PATH / 'db_img_names.npy', self.img_names)
        np.save(CUSTOM_PATH / 'embeddings_dict.npy', self.db_embeddings)


    def checkDesc(self, input_embeddings):
        dis_mean = {}
        for dataset in self.datasets:
            embedding = self.db_embeddings[dataset]
            distances = torch.cdist(input_embeddings, embedding, p=2).cpu().numpy()
            # get top min 10 of each rows
            distances_argsort = np.argsort(distances, axis=1)
            distances_top10 = distances_argsort[:, :10]
            distances_top10_values = np.take_along_axis(distances, distances_top10, axis=1)
            distances_top10_mean = np.mean(distances_top10_values, axis=1)
            # Using K-mean cluster top10_mean to low_bound and high bound
            kmeans = KMeans(n_clusters=2, random_state=0).fit(distances_top10_mean.reshape(-1, 1))
            low_bound = kmeans.cluster_centers_.min()
            high_bound = kmeans.cluster_centers_.max()
            dis_mean.update({dataset: [low_bound, high_bound]})

        # get min key in dis_mean
        match_dataset = min(dis_mean, key=lambda k: dis_mean[k][0])
        return match_dataset




# # 4. Read sample submission and querry.
# db_custom = CustomDB()
# db_custom.createFromPath(db_custom_path=CUSTOM_PATH)
#
# test_dict = {}
# test_dict = parse_sample_submission(IMC_PATH / 'sample_submission.csv')
# out_results = {}
#
# datasets = []
# for dataset in test_dict:
#     datasets.append(dataset)
#
# test_embeddings_dict = {}
# for dataset in test_dict:
#     print(dataset)
#     if dataset not in out_results:
#         out_results[dataset] = {}
#     for scene in test_dict[dataset]:
#         print(scene)
#         img_dir = os.path.join(CONFIG.base_path, '/'.join(test_dict[dataset][scene][0].split('/')[:-1]))
#         print(img_dir)
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
