import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import SimpleITK as sitk

import pandas as pd

config = parse_config("./config.cfg")

def random_flip(data_list):
    flipid = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    for i in range(len(data_list)):
        data_list[i] = np.ascontiguousarray(data_list[i][::flipid[0], ::flipid[1], ::flipid[2]])
    return data_list

def random_rotate(data_list):
    def rotate_left(data):
        data = data.transpose((0, 2, 1))
        data = np.ascontiguousarray(data[:, ::-1])
        return data
    def rotate_right(data):
        data = np.ascontiguousarray(data[:, ::-1])
        data = data.transpose((0, 2, 1))
        data = np.ascontiguousarray(data[:, ::-1])
        return data

    for i in range(len(data_list)):
        if random.random() > 0.5:
            data_list[i] = rotate_left(data_list[i])
        else:
            data_list[i] = rotate_right(data_list[i])
    return data_list

def random_color(data, rate=0.2):
    r1 = (random.random() - 0.5) * 2 * rate
    r2 = (random.random() - 0.5) * 2 * rate
    data = data * (1 + r2) + r1
    return data

def load_train_file(file_path, folder='0', mode=['train']):
    with open(file_path, 'r') as file:
        data = json.load(file)
    file_list = []
    if int(folder) >= 0:
        for m in mode:
            file_list += data[folder][m]
    else:
        file_list = data[mode[0]]
    # file_list = [f[:f.find(".nii.gz")] for f in file_list]
    return file_list

########################################################################################################################
class CropSegData(Dataset):
    def __init__(self, file_path, data_root, batch_size, ifgrad=False):
        super(CropSegData, self).__init__()
        self.file_list = sorted(pd.read_csv(file_path)['filename'].tolist())
   
        self.root = data_root
        self.batch_size = batch_size
        self.ifgrad = ifgrad

    def __len__(self):
        return len(self.file_list)

    def crop(self, data_list, crop_size=[128, 128, 128]):
        _data = data_list[0]
        shape = _data.shape

        if (shape < np.array(crop_size)).any():
            for data in data_list:
                data = np.pad(data, ((0, max(0, self.crop_size[0] - shape[1])),
                                                                              (0, max(0, self.crop_size[1] - shape[2])),
                                                                      (0, max(0, self.crop_size[2] - shape[3]))), 'constant')

        random_range = [[crop_size[i]//2, shape[i]-crop_size[i]//2] for i in range(3)]
        random_center = []
        for i in range(self.batch_size):
            z = random.randint(random_range[0][0], random_range[0][1])
            y = random.randint(random_range[1][0], random_range[1][1])
            x = random.randint(random_range[2][0], random_range[2][1])
            random_center.append([z, y ,x])
        out_list = []
        for data in data_list:
            out = []
            for c in random_center:
                z, y, x = c[0], c[1], c[2]
                out.append(data[z-crop_size[0]//2 : z+crop_size[0]//2,
                                y-crop_size[1]//2 : y+crop_size[1]//2,
                                x-crop_size[2]//2 : x+crop_size[2]//2])
            out_list.append(out)
        return out_list

    def process_imgmsk(self, data, mask):
        data = data.astype(float)
        data2 = data.copy()
        data2[data2 > 500] = 500
        data2[data2 < -1000] = -1000
        data2 = (data2 + 1000) / 1500
        
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048
        mask = (mask > 0).astype(int).astype(float)
        return data, data2, mask
    
    def process_imgmsk_mean_std(self, data, mask):
        data = data.astype(float)
        data2 = data.copy()
        data2[data2 > 500] = 500
        data2[data2 < -1000] = -1000
        data2 = (data2 + 1000) / 1500

        data[data > -225] = -225
        data[data < -1024] = -1024
        data = (data + 929.7) / 119.8
        mask = (mask > 0).astype(int).astype(float)
        return data, data2, mask

    def augment(self, data_list):
        if random.random() > 0.5:
            result = random_flip(data_list)
            data_list = result
        if random.random() > 0.5:
            result = random_rotate(data_list)
            data_list = result
        # if random.random() > 0.5:
        #     data = random_color(data_list[0])
        #     data2 = random_color(data_list[1])
        #     data_list[0] = data
        #     data_list[1] = data2
        return data_list

    def __getitem__(self, item):
        name = self.file_list[item]
        img = sitk.ReadImage(os.path.join(self.root, name, 'image.nii.gz'))
        img = sitk.GetArrayFromImage(img)
        label = sitk.ReadImage(os.path.join(self.root,  name, 'airway_three_cla.nii.gz'))
        label = sitk.GetArrayFromImage(label)
        skeleton = sitk.ReadImage(os.path.join(self.root,  name, 'skeleton.nii.gz'))
        skeleton = sitk.GetArrayFromImage(skeleton)
        # weight = np.load(os.path.join(self.root, name, 'weight.nii.gz'))
        # weight = sitk.ReadImage(os.path.join(self.root, name, "weight.nii.gz"))
        # weight = sitk.GetArrayFromImage(weight)
        if self.ifgrad:
            dis = np.load(os.path.join(self.root, 'LIBBP', 'distance', name + '.npy'))
            grads = np.load(os.path.join(self.root, 'LIBBP', 'grads', name + '.npy'))

        img, img2, label = self.process_imgmsk(img, label)
        # img, img2, label = self.process_imgmsk_mean_std(img, label)
        # img2 = img

        # weight = weight ** (np.random.random() + 2) * label + (1 - label)

        # img = np.expand_dims(img, axis=0)
        # label = np.expand_dims(label, axis=0)
        # transform_list = [get_transform(name, config['dataset']) \
        #             for name in config['dataset']['train_transform']]
        # transform = transforms.Compose(transform_list)
        # sample = dict()
        # sample['image'] = img
        # sample['label'] = label
        # sample = transform(sample)
        # img = np.squeeze(sample['image'], axis=0)
        # label = np.squeeze(sample['label'], axis=0)

        data_list = self.crop([img, label, skeleton])

        # # for i in range(len(data_list[0])):
        # #     _aug = self.augment([data_list[j][i] for j in range(len(data_list))])
        # #     for j in range(len(data_list)):
        # #         data_list[j][i] = _aug[j]

        img, label, ske = data_list[0], data_list[1], data_list[2]
        img = torch.from_numpy(np.array(img))
        # img2 = torch.from_numpy(np.array(img2))
        label = torch.from_numpy(np.array(label))
        ske = torch.from_numpy(np.array(ske))
        # weight = torch.from_numpy(np.array(weight))
        # print(img.shape, img2.shape, label.shape, weight.shape)

        return img, label, ske, name

########################################################################################################################
# class SegValData(Dataset):
#     def __init__(self, file_path, data_root):
#         self.file_list = load_train_file(file_path, folder='0', mode=['val'])
#         self.root = data_root

#     def __len__(self):
#         return len(self.file_list)

#     def process_imgmsk(self, data, mask):
#         data = data.astype(float)
#         data2 = data.copy()
#         data2[data2 > 500] = 500
#         data2[data2 < -1000] = -1000
#         data2 = (data2 + 1000) / 1500
#         data[data > 1024] = 1024
#         data[data < -1024] = -1024
#         data = (data + 1024) / 2048
#         mask = (mask > 0).astype(int).astype(float)
#         return data, data2, mask

#     def __getitem__(self, item):
#         name = self.file_list[item]
#         img = sitk.ReadImage(os.path.join(self.root,  name + 'image.nii.gz'))
#         img = sitk.GetArrayFromImage(img)
#         label = sitk.ReadImage(os.path.join(self.root,  name + 'airway.nii.gz'))
#         label = sitk.GetArrayFromImage(label)
#         img, img2, label = self.process_imgmsk(img, label)

#         img = np.array([img, img2])
#         label = label[np.newaxis, ...]
#         return torch.from_numpy(img.astype(np.float32)), \
#                torch.from_numpy(label.astype(np.float32)), name

########################################################################################################################
class SegValCropData(Dataset):
    def __init__(self, file_path, data_root, batch_size, cube_size=128, step=64):
        self.file_list = sorted(pd.read_csv(file_path)['filename'].tolist())
        self.root = data_root
        self.cube_size = cube_size
        self.batch_size = batch_size
        self.step = step
        self.file_dic, self.pos_list = self.crop_pos()
        self.last_name = ''
        self.img = None

    def __len__(self):
        return len(self.file_list)

    def crop_pos(self):
        file_dic = {}
        cube_size, step = self.cube_size, self.step
        for f in self.file_list:
            tmp = []
            img = sitk.ReadImage(os.path.join(self.root,  f, 'image.nii.gz'))
            x = sitk.GetArrayFromImage(img)               
            if x.shape[0] < self.cube_size[0] or x.shape[1] < self.cube_size[1] or x.shape[2] < self.cube_size[2]:
                pad = []
                for i in range(3):
                    si = x.shape[i]
                    if si < self.cube_size[i]:
                        pad.append((0, self.cube_size[i]-si))
                    else:
                        pad.append((0, 0))
                x = np.pad(x, pad, 'minimum')
                # print("x pad shape: ", x.shape)
            x = x[np.newaxis, ...]
            xnum = (x.shape[1] - cube_size[0]) // step[0] + 1 if (x.shape[1] - cube_size[0]) % step[0] == 0 else \
                (x.shape[1] - cube_size[0]) // step[0] + 2
            ynum = (x.shape[2] - cube_size[1]) // step[1] + 1 if (x.shape[2] - cube_size[1]) % step[1] == 0 else \
                (x.shape[2] - cube_size[1]) // step[1] + 2
            znum = (x.shape[3] - cube_size[2]) // step[2] + 1 if (x.shape[3] - cube_size[2]) % step[2] == 0 else \
                (x.shape[3] - cube_size[2]) // step[2] + 2
            for xx in range(xnum):
                xl = step[0] * xx
                xr = step[0] * xx + cube_size[0]
                if xr > x.shape[1]:
                    xr = x.shape[1]
                    xl = x.shape[1] - cube_size[0]
                for yy in range(ynum):
                    yl = step[1] * yy
                    yr = step[1] * yy + cube_size[1]
                    if yr > x.shape[2]:
                        yr = x.shape[2]
                        yl = x.shape[2] - cube_size[1]
                    for zz in range(znum):
                        zl = step[2] * zz
                        zr = step[2] * zz + cube_size[2]
                        if zr > x.shape[3]:
                            zr = x.shape[3]
                            zl = x.shape[3] - cube_size[2]
                        tmp.append([xl, xr, yl, yr, zl, zr])
            while (len(tmp) % self.batch_size) != 0:
                tmp.append(tmp[0])
            file_dic[f] = tmp

        file_list, pos_list = [], []
        for f in self.file_list:
            file_list += [f for i in range(len(file_dic[f]))]
            pos_list += file_dic[f]
        self.file_list = file_list
        return file_dic, pos_list

    def crop(self, data, pos):
        xl, xr, yl, yr, zl, zr = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
        data_crop = data[:, xl:xr, yl:yr, zl:zr]
        return data_crop

    def process_imgmsk(self, data):
        data = data.astype(float)
        data2 = data.copy()
        data2[data2 > 500] = 500
        data2[data2 < -1000] = -1000
        data2 = (data2 + 1000) / 1500
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048

        return data, data2

    def process_imgmsk_mean_std(self, data):
        data = data.astype(float)
        data2 = data.copy()
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048

        data2[data2 > -225] = -225
        data2[data2 < -1024] = -1024
        data2 = (data2 + 929.7) / 119.8

        return data, data2
    

    def __getitem__(self, item):
        name = self.file_list[item]
        if name != self.last_name:
            img = sitk.ReadImage(os.path.join(self.root,  name, 'image.nii.gz'))
            img = sitk.GetArrayFromImage(img)
            if img.shape[0] < self.cube_size[0] or img.shape[1] < self.cube_size[1] or img.shape[2] < self.cube_size[2]:
                pad = []
                for i in range(3):
                    si = img.shape[i]
                    if si < self.cube_size[i]:
                        pad.append((0, self.cube_size[i]-si))
                    else:
                        pad.append((0, 0))
                img = np.pad(img, pad, 'minimum')
            img, img2 = self.process_imgmsk(img)

            img = np.expand_dims(img, axis=0)
            self.img = img
            self.last_name = name
        else:
            img = self.img
        img_crop = self.crop(img, self.pos_list[item])
        pos = np.array(self.pos_list[item])

        return torch.from_numpy(img_crop.astype(np.float32)), name, torch.from_numpy(pos)

def make_json():
    import pandas as pd
    train_name_list = pd.read_csv("/data/data2/datasets/AIIB2023/AIIB23_Train_T1/AIIB2023_train.csv")["filename"].tolist()
    valid_name_list = pd.read_csv("/data/data2/datasets/AIIB2023/AIIB23_Train_T1/AIIB2023_valid.csv")['filename'].tolist()
    test_name_list = pd.read_csv("/data/data2/datasets/AIIB2023/AIIB23_Valid_T1/AIIB2023_test.csv")["filename"].tolist()
 
    data = {}
    
    data['0'] = {}
    data['0']['train'] = train_name_list
    data['0']['val'] = valid_name_list
    data['test'] = test_name_list
    with open('./data/base_dict_aiib2023.json', 'w', newline='\n') as file:
        json.dump(data, file,indent=4)
        
    

if __name__ == '__main__':
    make_json()





