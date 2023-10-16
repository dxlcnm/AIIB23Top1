
"""
Created on 2023/8/3

@author: Xiaoliu Ding
"""

import SimpleITK as sitk
import scipy.ndimage as ndimage
import numpy as np
import skimage.measure as measure
import time
import os
from multiprocessing import Pool
import pandas as pd

def get_ND_bounding_box(volume, margin=None):
    """
    get the bounding box of nonzero region in an ND volume
    """
    input_shape = volume.shape
    if (margin is None):
        margin = [0] * len(input_shape)
    assert (len(input_shape) == len(margin))
    indxes = np.nonzero(volume)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max() + 1)

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i])
    return idx_min, idx_max

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
        
def gen_breakage_attetion_map(sitk_mask_path, sitk_lung_path, save_dir, f):
    mask_path = os.path.join(sitk_mask_path, f, "pred.nii.gz")
    lung_path = os.path.join(sitk_lung_path, f, "lung.nii.gz")
    sitk_mask = sitk.ReadImage(mask_path)
    mask_np = sitk.GetArrayFromImage(sitk_mask)
    if os.path.exists(lung_path):
        lung_mask = sitk.ReadImage(lung_path)
        lung_mask_np = sitk.GetArrayFromImage(lung_mask)
        minbox, maxbbox = get_ND_bounding_box(lung_mask_np)
        time1 = time.time()
        print("im in")
        lung_mask_bbox_np = np.zeros_like(lung_mask_np)
        lung_mask_bbox_np[minbox[0]: maxbbox[0], minbox[1]: maxbbox[1], minbox[2]: maxbbox[2]] = 1
        mask_np = mask_np * lung_mask_bbox_np
    print("label....")
    cd, num = measure.label(mask_np, return_num=True, connectivity=2)
    print("label done, num: ", num)

    volume = {}
    print("sort...")
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    print("sort done...")
        
    volume_sort = dict(sorted(volume.items(), key=lambda x: x[1], reverse=True))
   
    attention_min = np.ones_like(mask_np, dtype=np.float32) * 10000000
    attetnion_second = np.ones_like(mask_np, dtype=np.float32) * 100000
    i=0
    for key, value in volume_sort.items():
        if value> 50:
            i+=1
            print(key)
            data_component = (cd == (key + 1)).astype(np.uint8)
            dis_map = ndimage.distance_transform_edt((1 - data_component).astype(np.uint8))
            
            bool_mat = np.zeros_like(mask_np, dtype=np.bool8)
            bool_mat[dis_map < attention_min] = True
            attetnion_second[bool_mat] = attention_min[bool_mat]
            attention_min[bool_mat] = dis_map[bool_mat]
            
            
            logiand = np.logical_and(bool_mat == False, dis_map < attetnion_second)
            attetnion_second[logiand] = dis_map[logiand]
 
    attetnion_second = sigmoid(5 - attetnion_second)
    attetnion_second[attetnion_second < 0.001] = 0
    attetnion_second_out = sitk.GetImageFromArray(attetnion_second)
    attetnion_second_out.CopyInformation(sitk_mask)
    sitk.WriteImage(attetnion_second_out, os.path.join(save_dir, f, "breakage_map.nii.gz"), useCompression=True)
    print("save to: ", os.path.join(save_dir, f, "breakage_map.nii.gz"))
    time2 = time.time()
    print("cost time: ", time2 - time1)
        


if __name__ == "__main__":
    mask_dir = "/data/data2/datasets/ctLung/Airway/nifty_new/"
    lung_dir = "/data/data2/datasets/ctLung/Airway/nifty_new/"
    save_dir =  "/data/data2/datasets/ctLung/Airway/nifty_new/"
    filelist = pd.read_csv("/data/data1/dxl/code/groupRad/segVol/config/airway_new/atm/trainname.csv")["filename"].tolist()
    p = Pool(processes=2)
    for case_name in filelist:
        p.apply_async(gen_breakage_attetion_map, (mask_dir, lung_dir, save_dir, case_name))
    p.close()
    p.join()
        