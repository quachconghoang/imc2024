import shutil

import numpy as np
import pandas as pd
import os

imc_root = './kaggle/input/image-matching-challenge-2024/train/'
scenes = ['church', 'dioscuri', 'lizard', 'multi-temporal-temple-baalshamin',
          'pond', 'transp_obj_glass_cup', 'transp_obj_glass_cylinder']

scnID=6
path_sfm = imc_root + scenes[scnID] + '/sfm'
path_img = imc_root + scenes[scnID] + '/images'

with open(path_sfm + '/images.txt', 'r') as file:
    fullData = file.read()
with open(path_sfm + '/images.txt', 'r') as file:
    Lines = file.readlines()[4:]

img_dat = Lines[0::2]

replacements=[]
for buf in img_dat:
    _tmp = buf.split()
    old_name = _tmp[9]
    new_name_1 = old_name.lower().replace('.jpg','.png')
    new_name_2 = new_name_1.replace('-','_').replace('__','_').replace('(','').replace(')','')
    # print(old_name , ' -> ' , new_name_2)
    replacements.append([old_name,new_name_2])


for item in replacements:
    fullData = fullData.replace(item[0], item[1])
    print('Replacing ... ', item[0], '->', item[1])


new_path = imc_root + scenes[scnID]+ '/sfm_mapped'
os.makedirs(new_path)
with open(new_path+'/images.txt', 'w') as f:
    f.write(fullData)
# copy points3D.txt and cameras.txt if not exits:
if not os.path.exists(new_path+'/points3D.txt'):
    shutil.copy2(path_sfm + '/points3D.txt', new_path)
if not os.path.exists(new_path + '/cameras.txt'):
    shutil.copy2(path_sfm + '/cameras.txt', new_path)