# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:58:02 2021

@author: Hao Zheng
"""

import numpy as np
import os
import nibabel
from scipy import ndimage
from skimage.morphology import skeletonize_3d
import SimpleITK as sitk
from Data import load_train_file
from multiprocessing.pool import Pool
import tqdm
# from train import create_directories
import pandas as pd

def save_BP_weight_tw(mask_dir, name):
    # file_list = load_train_file('./data/base_dict.json', folder='0', mode=['train', 'val'])
    # file_list.sort()
    # for i in range(len(file_list)):
        #load the label, predition and gradient
        # name = file_list[i]
    # print(f"{name} start")
    
    # distance_bp_path = os.path.join('./data/LIBBP/distance_bp', name+'.nii.gz')
    # distance_bp_np = sitk.GetArrayFromImage(sitk.ReadImage(distance_bp_path, outputPixelType=sitk.sitkUInt8))
    # if np.sum(distance_bp_np) == 0:

    label_itk = sitk.ReadImage(os.path.join(mask_dir, name, 'airway.nii.gz'), outputPixelType=sitk.sitkUInt8)
    label = sitk.GetArrayFromImage(label_itk)
    
    # pred = nibabel.load(os.path.join('./data/LIBBP/preds', name+'.nii.gz'))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, name, 'pred.nii.gz'), sitk.sitkUInt8))
    
    # pred = pred.get_data()[0]
    # sitk.WriteImage(sitk.GetImageFromArray(pred), "/data/data1/dxl/code/ATM2022/result/a.nii.gz", useCompression=True)
    print(f"name: {name}, pred shape: {pred.shape}")
    # grad = np.load(os.path.join('./data/LIBBP/grads', name+'.npy'))
    # sitk.WriteImage(sitk.GetImageFromArray(grad), "/data/data1/dxl/code/ATM2022/result/a.nii.gz", useCompression=True)
    # grad = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, name, "grads.nii.gz")))
    
    fn = ((label.astype(np.float16) - pred)>0).astype(np.uint8)
    # skeleton = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('./data/skeleton', name+'.nii.gz'), outputPixelType=sitk.sitkUInt8))
    skeleton = skeletonize_3d(label)
    # grad_fn_skel = (1-grad)*fn*skeleton
    grad_fn_skel = fn*skeleton
    edt, inds = ndimage.distance_transform_edt(1-skeleton, return_indices=True)
    # sitk.WriteImage(sitk.GetImageFromArray(edt), os.path.join(mask_dir, name, "edt.nii.gz"))
    grad_wgt0 = grad_fn_skel[inds[0,...], inds[1,...], inds[2,...]] * label
    # sitk.WriteImage(sitk.GetImageFromArray(grad_fn_skel[inds[0,...], inds[1,...], inds[2,...]]), os.path.join(mask_dir, name, "grad_fn_skel_index.nii.gz"))
    # sitk.WriteImage(sitk.GetImageFromArray(grad_wgt0), os.path.join(mask_dir, name, "grad_wgt0.nii.gz"))
    
    loc = (grad_wgt0>0).astype(np.uint8)
    f = loc * edt
    f = f * (1. - skeleton)
    maxf = np.amax(f)
    D = -((1./(maxf)) * f) + 1
    D = D * loc
    
    grad_wgt = (grad_wgt0**2)*(D**2)
    grad_wgt = grad_wgt.astype(np.float16)
    
    # np.save(os.path.join('./data/LIBBP/distance_bp', name+'.npy'), grad_wgt)
    # create_directories(os.path.join(mask_dir, 'distance_bp', name))
    dis = sitk.GetImageFromArray(grad_wgt.astype(np.float32))
    dis.CopyInformation(label_itk)
    sitk.WriteImage(dis, os.path.join(mask_dir, name, 'distance_bp.nii.gz'), useCompression=True)
    print(os.path.join(mask_dir, name, 'distance_bp.nii.gz'))
    # print(name)
    
def save_BP_weight_tw_process():
    file_list = sorted(pd.read_csv("/data/data1/dxl/code/groupRad/segVol/config/airway_new/atm/name.csv")['filename'].tolist())
    mask_dir = "/data/data2/datasets/ctLung/Airway/nifty_new/"
    file_list.sort()
    p = Pool(processes=8)
    for case_name in file_list:
        # case_name = file_list[i]
        # print(case_name)
        # save_BP_weight_tw(mask_dir, case_name)
        p.apply_async(save_BP_weight_tw, (mask_dir, case_name))
    p.close()
    p.join()

def save_BP_weight(data_path, save_path):
    file_list = os.listdir(data_path)
    file_list.sort()
    for i in range(len(file_list) // 4):
        # load the label, predition and gradient
        label = nibabel.load(os.path.join(data_path + file_list[4 * i + 2]))
        pred = nibabel.load(os.path.join(data_path + file_list[4 * i + 3]))
        grad = nibabel.load(os.path.join(data_path + file_list[4 * i]))
        label = label.get_data()
        pred = pred.get_data()
        grad = grad.get_data()

        fn = ((label.astype(np.float16) - pred) > 0).astype(np.uint8)
        skeleton = skeletonize_3d(label)
        grad_fn_skel = (1 - grad) * fn * skeleton
        # grad_fn_skel = fn*skeleton
        edt, inds = ndimage.distance_transform_edt(1 - skeleton, return_indices=True)
        grad_wgt0 = grad_fn_skel[inds[0, ...], inds[1, ...], inds[2, ...]] * label

        loc = (grad_wgt0 > 0).astype(np.uint8)
        f = loc * edt
        f = f * (1. - skeleton)
        maxf = np.amax(f)
        D = -((1. / (maxf)) * f) + 1
        D = D * loc

        grad_wgt = (grad_wgt0 ** 2) * (D ** 2)
        grad_wgt = grad_wgt.astype(np.float16)
        save_name = save_path + file_list[4 * i + 1].split('_')[0] + "_dis2.npy"
        np.save(save_name, grad_wgt)


if __name__ == '__main__':
    # save_BP_weight_tw()
    save_BP_weight_tw_process()