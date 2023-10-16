# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 19:23:00 2021

@author: Hao Zheng
"""
import numpy as np
import os
from scipy import ndimage
import SimpleITK as sitk
from multiprocessing import Pool
import pandas as pd

from train import create_directories
def neighbor_descriptor(label, filters):
    den = filters.sum()
    conv_label = ndimage.convolve(label.astype(np.float32), filters, mode='mirror')/den
    conv_label[conv_label==0] = 1
    conv_label = -np.log10(conv_label)
    return conv_label

def save_local_imbalance_based_weight(label_path, save_path):
    file_list = os.listdir(label_path)
    file_list.sort()
    for i in range(len(file_list)//5):
        label = np.load(os.path.join(label_path, file_list[5*i])) #load the binary labels
        filter0 = np.ones([7,7,7], dtype=np.float32)
        weight = neighbor_descriptor(label, filter0)       
        weight = weight*label
        #Here is constant weight. During training, varied weighted training is adopted.
        #weight = weight**np.random.random(2,3) * label + (1-label) in dataloader.
        weight = weight**2.5 
        weight = weight.astype(np.float16)
        save_name = save_path + file_list[5*i].split('_')[0] + "_weight.npy"
        np.save(save_name, weight)   

def save_lib_weight(label_path, save_path, f):

    print(f)
    label_Img = sitk.ReadImage(os.path.join(label_path, f, "airway.nii.gz"))
    label = sitk.GetArrayFromImage(label_Img)
    name = f
    
    filter0 = np.ones([7, 7, 7], dtype=np.float32)
    weight = neighbor_descriptor(label, filter0)
    weight = weight * label
    # Here is constant weight. During training, varied weighted training is adopted.
    # weight = weight**np.random.random(2,3) * label + (1-label) in dataloader.
    # weight = weight ** 2.5
    weight = weight.astype(np.float16)
    create_directories(os.path.join(save_path, name))
    save_itk = sitk.GetImageFromArray(weight.astype(np.float32))
    save_itk.CopyInformation(label_Img)
    sitk.WriteImage(save_itk, os.path.join(save_path, name, 'weight_merge.nii.gz'), useCompression=True)
        
def save_lib_weight_process():
    file_list = sorted(pd.read_csv("/data/data2/datasets/AIIB2023/AIIB23_Train_T1_use_all_trainset/AIIB23_Train_T1/AIIB2023_valid.csv")['filename'].tolist())
    mask_dir = "/data/data2/datasets/AIIB2023/AIIB23_Train_T1_use_all_trainset/AIIB23_Train_T1/nifti/"
    file_list.sort()
    p = Pool(processes=8)
    for case_name in file_list:
        p.apply_async(save_lib_weight, (mask_dir, mask_dir, case_name))
    p.close()
    p.join()


if __name__ == '__main__':
    save_lib_weight_process()















