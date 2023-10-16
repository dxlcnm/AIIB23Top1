import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import SimpleITK as sitk
import scipy.ndimage as ndimage
from Data import load_train_file
import nibabel
from utilies.augment_utilies import get_patch_size
import pandas as pd
np.random.seed(777)  # numpy

import torchvision.transforms as transforms
from utilies.augment import get_transform

from utilies.parse_config import parse_config
config = parse_config("./config.cfg")

def central_crop(sample, label, dist, crop_size):
    origin_size = sample.shape
    crop_size = np.array(crop_size)
    start = (origin_size - crop_size) // 2
    sample = sample[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
             start[2]:(start[2] + crop_size[2])]
    label = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1], start[2]:start[2] + crop_size[2]]
    dist = dist[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1], start[2]:start[2] + crop_size[2]]
    return sample, label, dist

def random_rotate(img, label, dist, angle, threshold):
    rotate_angle = np.random.randint(angle) * np.sign(np.random.random() - 0.5)
    rotate_axes = [(0, 1), (1, 2), (0, 2)]
    k = np.random.randint(0, 3)
    img = ndimage.interpolation.rotate(img, angle=rotate_angle, axes=rotate_axes[k], reshape=False)
    label = label.astype(np.float32)
    label = ndimage.interpolation.rotate(label, angle=rotate_angle, axes=rotate_axes[k], reshape=False)
    threshold = threshold  # threshold=0.7 in stage1 and 0.9 in stage2
    label[label >= threshold] = 1
    label[label < threshold] = 0
    label = label.astype(np.uint8)

    dist = dist.astype(np.float32)
    dist = ndimage.interpolation.rotate(dist, angle=rotate_angle, axes=rotate_axes[k], reshape=False)
    dist[dist > 1] = 1
    dist[dist < 0] = 0
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)
    return img, label, dist

def skeleton_sample(img, label, weight, skeleton, loc, crop_size):
    origin_size = img.shape[-3:]
    # crop_size = np.array([cube_size, cube_size, cube_size])
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[:,start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skeleton_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop, skeleton_crop

def small_airway_sample(img, label, weight, skeleton, loc, crop_size):
    origin_size = img.shape[-3:]
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[:,start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skeleton_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop, skeleton_crop

def random_sample(img, label, weight, skeleton, crop_size):
    origin_size = img.shape[-3:]
    start = [np.random.randint(0, origin_size[0] - crop_size[0]), np.random.randint(0, origin_size[1] - crop_size[1]),
             np.random.randint(0, origin_size[2] - crop_size[2])]
    img_crop = img[:,start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skeleton_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    return img_crop, label_crop, weight_crop, skeleton_crop

def center_sample(img, label, weight, skeleton, crop_size):
    origin_size = img.shape[-3:]
    start = [max(0, origin_size[0]//2 - crop_size[0] // 2), max(0, origin_size[1]//2 - crop_size[1] // 2), max(0, origin_size[2]//2 - crop_size[2] // 2)]

    img_crop = img[:,start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skeleton_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    return img_crop, label_crop, weight_crop, skeleton_crop

class AirwayHMData(Dataset):
    def __init__(self, file_path, data_root, batch_size, cube_size, ifbp=False):
        self.root = data_root
        self.file_list = pd.read_csv(file_path)['filename'].tolist()
        self.batch_size = batch_size
        self.cube_size = cube_size
        self.ifbp = ifbp
        
    def __len__(self):
        return len(self.file_list)

    def process_img(self, img_crops):
        img2_crops = []
        for i in range(len(img_crops)):
            crop = img_crops[i]
            crop2 = crop.copy()
            crop2[crop2 > 500] = 500
            crop2[crop2 < -1000] = -1000
            crop2 = (crop2 + 1000) / 1500
            
            crop[crop > 1024] = 1024
            crop[crop < -1024] = -1024
            crop = (crop + 1024) / 2048
            img2_crops.append(crop2) # or crop???
            img_crops[i] = crop
        return img_crops, img2_crops
    
    def process_img_mean_std(self, img_crops):
        img2_crops = []
        for i in range(len(img_crops)):
            crop = img_crops[i]
            crop2 = crop.copy()
            crop2[crop2 > 1024] = 1024
            crop2[crop2 < -1024] = -1024
            crop2 = (crop2 + 1024) / 2048

            crop[crop > -225] = -225
            crop[crop < -1024] = -1024
            crop = (crop + 929.7) / 119.8
            img2_crops.append(crop2) # or crop???
            img_crops[i] = crop
        return img_crops, img2_crops
    

    def crop(self, img, label, weight, pred, skeleton):
        img_crops, label_crops, weight_crops, skeleton_crops = [], [], [], []
        label_temp = label.copy()
        label_temp[label_temp > 0] = 1
        dis = ndimage.distance_transform_edt(label_temp)
        loc_small = np.where((dis * skeleton) < 2)
        loc_skeleton = np.where(skeleton * (1 - pred))
       
        cube_size = [self.cube_size] * 3
        cube_size = self.cube_size

        if (pred * skeleton).sum() == skeleton.sum():
            for i in range(self.batch_size):
                p = np.random.random()
                if p > 0.5 and len(loc_small) > 0:
                    # print("== small")
                    img_crop, label_crop, weight_crop, skeleton_crop = small_airway_sample(img, label, weight, skeleton, loc_small, cube_size)
                else:
                    # print("== random")
                    img_crop, label_crop, weight_crop, skeleton_crop = random_sample(img, label, weight, skeleton, cube_size)
                img_crops.append(img_crop)
                label_crops.append(label_crop)
                weight_crops.append(weight_crop)
                skeleton_crops.append(skeleton_crop)
        else:
            for i in range(self.batch_size):
                p = np.random.random()
                # skeleton hard sample mining
                if p > 0.75 and len(loc_skeleton) > 0: # 0.5
                    # print("!=skeleton")
                    img_crop, label_crop, weight_crop, skeleton_crop = skeleton_sample(img, label, weight, skeleton, loc_skeleton, cube_size)
                # sample on small airway
                elif p > 0.5 and len(loc_small) > 0: # 0.25
                    # print("!=small")
                    img_crop, label_crop, weight_crop, skeleton_crop = small_airway_sample(img, label, weight, skeleton, loc_small, cube_size)
                # random sampling
                else:
                    # print("!=random")
                    img_crop, label_crop, weight_crop, skeleton_crop = random_sample(img, label, weight, skeleton, cube_size)
                img_crops.append(img_crop)
                label_crops.append(label_crop)
                weight_crops.append(weight_crop)
                skeleton_crops.append(skeleton_crop)

        return img_crops, label_crops, weight_crops, skeleton_crops

    def __getitem__(self, item):
        name = self.file_list[item]
        img = sitk.ReadImage(os.path.join(self.root, name,'image.nii.gz'))
        img = sitk.GetArrayFromImage(img)


        label = sitk.ReadImage(os.path.join(self.root, name,'airway.nii.gz'))
        label = sitk.GetArrayFromImage(label).astype(np.float32)
        if not self.ifbp:
   
            weight = sitk.ReadImage(os.path.join(self.root, name, 'weight.nii.gz'))
            weight = sitk.GetArrayFromImage(weight)

            pred = sitk.ReadImage(os.path.join(self.root, name, 'pred.nii.gz'))
            pred = sitk.GetArrayFromImage(pred)

        else:

            weight = sitk.ReadImage(os.path.join(self.root, name, 'weight.nii.gz'))
            weight = sitk.GetArrayFromImage(weight)

            bp_weight = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, name, 'distance_bp.nii.gz')))

            breakage_weight = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, name, 'breakage_map.nii.gz')))
            weight = weight + 0.5 * bp_weight + breakage_weight
            # weight = weight + 0.5 * bp_weight 
 
            pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, name,'pred.nii.gz')))

        skeleton = sitk.ReadImage(os.path.join(self.root, name, 'skeleton.nii.gz'))
        skeleton = sitk.GetArrayFromImage(skeleton)
        weight = weight ** (np.random.random() + 2) * label + (1 - label)
        weight = weight.astype(np.float32)
        
        img = np.expand_dims(img, axis=0)

        img_crops, label_crops, weight_crops, skeleton_crops = self.crop(img, label, weight, pred, skeleton)

        img_crops, img2_crops = self.process_img(img_crops)
        print(f"name: {name}, img_crops shape: {np.array(img_crops).shape}")
        img1_crops = torch.from_numpy(np.array(img_crops)[:,0,:,:,:])
        print(f"name: {name}, img1 crops shape: {img1_crops.shape}")

        
        label_crops = torch.from_numpy(np.array(label_crops))
        weight_crops = torch.from_numpy(np.array(weight_crops))
        skeleton_crops = torch.from_numpy(np.array(skeleton_crops))
        if len(weight_crops.shape) < 3:
            print(f"name: {name}, weight_crops.shape: {weight_crops.shape} incrorrect.")

        return img1_crops, label_crops, weight_crops, skeleton_crops, name

class OnlineHMData(Dataset):
    def __init__(self, data_root, batch_size, rate=0.33):
        self.data_root = data_root
        self.batch_size = batch_size
        self.name_list = os.listdir(os.path.join(data_root, 'image'))
        self.name_list.sort(key=lambda x:float(x.split('_')[0]))
        self.name_list = self.name_list[-int(rate * len(self.name_list)):]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        image = np.load(os.path.join(self.data_root, 'image', name))
        label = np.load(os.path.join(self.data_root, 'label', name))
        weight = np.load(os.path.join(self.data_root, 'weight', name))
        skeleton = np.load(os.path.join(self.data_root, 'skeleton', name))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        weight = torch.from_numpy(weight)
        skeleton = torch.from_numpy(skeleton)

        return image, label, weight, skeleton


if __name__ == '__main__':
    p1 = nibabel.load('./data/pred2/ATM_001_0000.nii.gz')
    p1 = p1.get_data()
    p2 = nibabel.load('./data/LIBBP/preds/ATM_001_0000.nii.gz')
    p2 = p2.get_data()[0]
    print(p1.shape, p2.shape, 2 * (p1 * p2).sum() / (p1 + p2).sum())





















