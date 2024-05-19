# General utilities
import matplotlib.pyplot as plt
from utils import load_torch_image, embed_images
from config import *


# working_path = root_path+"/kaggle/working/imc-data/"
imgDB = (DATA_PATH / 'train' / 'transp_obj_glass_cylinder' / 'images')
images_list = list(imgDB.glob("*.png"))
images_list.sort()
images_fname = [str(f).split('/')[-1] for f in images_list]

DINO_MODEL = str(INPUT_PATH / "dinov2/pytorch/base/1")
embeddings = embed_images(images_list, DINO_MODEL, device=DEVICE)
distances = torch.cdist(embeddings, embeddings, p=2).cpu().numpy()

plt.imshow(distances)
plt.show()

# convert full path from images_list to file name list only



for dist in distances:
    # get index of top 5  min of dist
    ...
dist = distances[0]
idx = np.argsort(dist)[1:6]




