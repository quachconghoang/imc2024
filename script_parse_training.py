import numpy as np
import pandas as pd
import os

imc_root = './kaggle/input/image-matching-challenge-2024/train/'
scenes = ['church', 'dioscuri', 'lizard', 'multi-temporal-temple-baalshamin','pond']
# transparent is fixed - just hardcode!

scnID = 4

path_sfm = imc_root + scenes[scnID] + '/sfm'
path_img = imc_root + scenes[scnID] + '/images'


def loadSFM(path_sfm:str, path_img:str):
    cameras_dat = np.loadtxt(path_sfm + '/cameras.txt', dtype=str)
    # convert cameras_dat as dict
    cameras_dict = {}
    for _item in cameras_dat:
        _key = _item[0]
        _type = _item[1]
        _size = np.asarray(_item[2:4], dtype=np.int32)
        _params = np.asarray(_item[4:], dtype=np.float64)
        cameras_dict.update({_key: {'SIZE': _size, 'TYPE': _type, 'PARAMS': _params}})

    images_dat = open(path_sfm + '/images.txt', 'r')
    Lines = images_dat.readlines()[4:]
    img_dat = Lines[0::2]
    print('NUMBER OF REGISTERED IMAGE:', len(img_dat))
    rs = []
    for buf in img_dat:
        _tmp = buf.split()
        _cam = {'SIZE': np.array((-1, -1), dtype=np.int32), 'TYPE': 'UNKNOWN',
                'PARAMS': np.array((-1, -1, -1), dtype=np.float32)}
        if cameras_dict.get(_tmp[0]):
            _cam = cameras_dict[_tmp[0]]
        else:
            print('UNKNOWN CAMERA for Image:', _tmp[9])

        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        rs.append({'IMAGE_ID': _tmp[0], 'CAMERA_ID': _tmp[8], 'NAME': _tmp[9],
                   'POSE': np.array(_tmp[1:8], dtype=np.float64), **_cam})

        # POINTS2D[] as (X, Y, POINT3D_ID)
        # pts_dat = Lines[1::2]
        # tmp = np.array(pts_dat[0].split())
        # tmp3D = np.array_split(tmp, indices_or_sections=len(tmp)/3)
        # _pt = np.asarray(tmp3D,dtype=np.float64)
        # point2D =  _pt[:,0:2]
        # pointID = _pt[:,2].astype(np.int32)
    return rs

rs = loadSFM(path_sfm, path_img)

# Save _rs to CSV by Pandas with IMAGE_ID CAMERA_ID NAME POSE (QwQxQyQzTxTyTz)
df = pd.DataFrame(rs)
df.to_csv(imc_root + scenes[scnID] + '/sfmDB.csv', index=False)
# test reading the saved file by pandas with HEADER
new_df = pd.read_csv(imc_root + scenes[scnID] + '/sfmDB.csv')
# print(new_df.head(7))

img_reg = new_df.NAME
img_files = os.listdir(path_img)
img_files.sort()

if (len(img_files) != len(img_reg)):
    print('ISSUE with db')
    if len(img_files) > len(img_reg):
        print('May has Extra images')

#find similar filename in two lists: ignore extension - return Index
def find_similar_files(list1, list2, hardcore=False): #input 1 - actual files, input 2 db files
    match = []
    unmatch_in = []
    match_db_status = np.zeros(len(list2))
    for i in range(len(list1)):
        _founded = False
        expected_name = list1[i].split('.')[0]
        for j in range(len(list2)):
            name_in_db = list2[j].split('.')[0]
            if hardcore:
                name_in_db = name_in_db.replace('(','').replace(')','').replace('__','_')
            if expected_name.lower().replace('-','_') == name_in_db.lower().replace('-','_'):
                _founded = True
            if _founded:
                match.append((i, j))
                match_db_status[j]=1
                break

        if _founded != True:
            unmatch_in.append(i)
    return match, unmatch_in,match_db_status

dumb = []
mtc, non_mtc, db_status = find_similar_files(img_files,img_reg,hardcore=True)
for p0,p1 in mtc:
    print('Found: ', img_files[p0], ' for registered image (', img_reg[p1], ') [', new_df.IMAGE_ID[p1] ,']')
    dumb.append({ 'NAME': img_files[p0] ,'IMAGE_ID':new_df.IMAGE_ID[p1], 'OLD_NAME' : img_reg[p1]})
for p0 in non_mtc:
    print('Not found ', img_files[p0], 'in registed db')
    dumb.append({'NAME': img_files[p0], 'IMAGE_ID': int(-1), 'OLD_NAME': 'UNKNOWN'})

print('---------Total not found = ', len(non_mtc))

for id,st in enumerate(db_status):
    if st==0: print('UNMAP: ',img_reg[id])

#save dumb to csv by pandas
df_dumb = pd.DataFrame(dumb)
df_dumb.to_csv(imc_root + scenes[scnID] + '/map_to_sfmDB.csv', index=False)