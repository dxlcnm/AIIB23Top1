import numpy as np
import torch

import os
from torch.utils.data import DataLoader
from WingsNet_Ori import WingsNet
from Data import CropSegData, SegValCropData
from skimage.morphology import skeletonize_3d
import SimpleITK as sitk
import time
from postprocessing import large_connected_domain

def create_directories(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = ((iflat) * tflat).sum()

    return 1 - ((2. * intersection + smooth) / ((iflat).sum() + (tflat).sum() + smooth))

def Tversky_loss(pred, target):
    smooth = 1.0
    alpha = 0.05
    beta = 1 - alpha
    intersection = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()
    return 1 - (intersection + smooth) / (intersection + alpha * FP + beta * FN + smooth)

def general_union_loss(pred, target, dist):
    weight = dist * target + (1 - target)
    # when weight = 1, this loss becomes Root Tversky loss
    smooth = 1.0
    alpha = 0.1  # alpha=0.1 in stage1 and 0.2 in stage2
    beta = 1 - alpha
    sigma1 = 0.0001
    sigma2 = 0.0001
    weight_i = target * sigma1 + (1 - target) * sigma2
    intersection = (weight * ((pred + weight_i) ** 0.7) * target).sum()
    intersection2 = (weight * (alpha * pred + beta * target)).sum()
    return 1 - (intersection + smooth) / (intersection2 + smooth)

def train():
    max_epoches = 100
    batch_size = 10
    os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
    train_path = "/data/data2/datasets/AIIB2023/AIIB23_Train_T1/AIIB2023_train.csv"
    val_path = "/data/data2/datasets/AIIB2023/AIIB23_Train_T1/AIIB2023_valid.csv"
    data_root = '/data/data2/datasets/AIIB2023/AIIB23_Train_T1/nifti/'
    save_dir = './saved_model_aiib/'
    save_name = "aiib_stage1"

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    in_channel = 1
    n_classes = 1
    
    model = WingsNet(in_channel=1, n_classes=n_classes)

    print(f"inchannel: {in_channel}, n_classes: {n_classes}")
    train_dataset = CropSegData(file_path=train_path, data_root=data_root, batch_size=batch_size)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=4,
                                   pin_memory=True, drop_last=True)
  

    max_step = len(train_dataset) * max_epoches
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # resume
    # weights_dict = torch.load("")
    # model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    ep_start = 0
    for ep in range(ep_start, max_epoches):
        time1 = time.time()
        for iter, pack in enumerate(train_data_loader):
            data = pack[0].float().cuda()
          
            label = pack[1].float().cuda()
 
            skeleton = pack[2].float().cuda()
            name = pack[3]

            data = data.transpose(0, 1)
     
            label = label.transpose(0, 1)
            skeleton = skeleton.transpose(0, 1)

            pred_en, pred_de = model(data)

            pred_en = torch.sigmoid(pred_en, 1)
            pred_de = torch.sigmoid(pred_de, 1)
            
            dice_loss_en = dice_loss(pred_en, label)
            dice_loss_de = dice_loss(pred_de, label)
            loss = dice_loss_en + dice_loss_de
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset), '/', max_step,
                      'loss:', loss.item(), 'dice loss encode:', dice_loss_en.item(),
                      'dice loss decode:', dice_loss_de.item())
            torch.cuda.empty_cache()
            # break

        print('validation')

        valid_dataset = SegValCropData(val_path, data_root, batch_size=batch_size, cube_size = [128,128,128], step = [64,64,64])
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True, drop_last=True)
        save_folder = os.path.join(save_dir, save_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        validation(model, valid_dataloader, ep, data_root, save_dir, save_name)
        
        print('saving model')
        save_folder = os.path.join(save_dir, save_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        torch.save(model.module.state_dict(),
                   os.path.join(save_folder, 'wingsnet_' + str(ep) + '.pth'))

def validation(model, valid_dataloader, epoch, data_root, save_dir, save_name):
    time1 = time.time()
    model.train()
    # sliding window
    sens, pres, branches, dices = [], [], [], []
    last_name = ''
    flag = False
    with torch.no_grad():
        for i, (x, name, pos) in enumerate(valid_dataloader):
            name = name[0]
            if name != last_name:
                if last_name != '':
                    pred = pred / pred_num
                    if pred.shape[1] ==1:
                        pred[pred >= 0.5] = 1
                        pred[pred < 0.5] = 0
                    else:
                        pred = np.argmax(pred, axis=1)
                    pred = np.squeeze(pred)
                    sen, pre, branch, dice = evaluation_case(pred, label, last_name, data_root)
                    sens.append(sen)
                    pres.append(pre)
                    branches.append(branch)
                    dices.append(dice)

                label = sitk.ReadImage(os.path.join(data_root, name, 'airway.nii.gz'))
                label = sitk.GetArrayFromImage(label)
                pred = np.zeros(label.shape)
                pred = pred[np.newaxis, np.newaxis, ...]
                pred_num = np.zeros(pred.shape)
                last_name = name

            x = x.cuda()
            p0, p = model(x)
     
            p = torch.sigmoid(p)
            p = p.cpu().detach().numpy()
            pos = pos.numpy()
            for i in range(len(pos)):
                xl, xr, yl, yr, zl, zr = pos[i,0], pos[i,1], pos[i,2], pos[i,3], pos[i,4], pos[i,5]
                pred[0, :, xl:xr, yl:yr, zl:zr] += p[i]
                pred_num[0, :, xl:xr, yl:yr, zl:zr] += 1

        pred = pred / pred_num
        if pred.shape[1] ==1:
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
        else:
            pred = np.argmax(pred, axis=1)
        pred = np.squeeze(pred)
        sen, pre, branch, dice = evaluation_case(pred, label, last_name, data_root)
        sens.append(sen)
        pres.append(pre)
        branches.append(branch)
        dices.append(dice)

        sen_mean = np.mean(sens)
        sen_std = np.std(sens)
        pre_mean = np.mean(pres)
        pre_std = np.std(pres)
        branch_mean = np.mean(branches)
        branch_std = np.std(branches)
        dice_mean = np.mean(dices)
        dice_std = np.std(dices)
        print("len mean: %0.4f (%0.4f), branch: %0.4f (%0.4f), pre: %0.4f (%0.4f), dice: %0.4f (%0.4f)" % (
               sen_mean, sen_std, branch_mean, branch_std, pre_mean, pre_std, dice_mean, dice_std))
        line = "len mean: %0.4f (%0.4f), branch: %0.4f (%0.4f), pre: %0.4f (%0.4f), dice: %0.4f (%0.4f)" % ( \
               sen_mean, sen_std, branch_mean, branch_std, pre_mean, pre_std, dice_mean, dice_std)
        save_log_path = os.path.join(save_dir, save_name + '.txt')
        with open(save_log_path, 'a') as file:
            file.writelines(['epoch:' + str(epoch)+'\n', line+'\n', '\n'])

def evaluation_case(pred, label, name, data_dir):
    
    if len(pred.shape) > 3:
        pred = pred[0]
    if len(label.shape) > 3:
        label = label[0]
        """
    # cd, num = measure.label(pred, return_num=True, connectivity=1)
    # volume = np.zeros([num])
    # for k in range(num):
    #     volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    # volume_sort = np.argsort(volume)
    # print(volume_sort)
    # large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    """
    large_cd = pred

    # skeleton = skeletonize_3d(label)
    skeleton_path = os.path.join(data_dir, name, 'skeleton.nii.gz')
    if not os.path.isfile(skeleton_path):
        skeleton = skeletonize_3d(label)
    else:
        skeleton = sitk.ReadImage(skeleton_path)
        skeleton = sitk.GetArrayFromImage(skeleton)

    skeleton = (skeleton > 0)
    skeleton = skeleton.astype('uint8')

    # print(pred.shape, label.shape, skeleton.shape)
    sen = (large_cd * skeleton).sum() / skeleton.sum()
    pre = (large_cd * label).sum() / large_cd.sum()
    """
    num_branch = parsing.max()
    detected_num = 0
    # for j in range(num_branch):
    #     branch_label = ((parsing == (j + 1)).astype(int)) * skeleton
    #     if (large_cd * branch_label).sum() / branch_label.sum() >= 0.8:
    #         detected_num += 1
    branch_label = parsing * skeleton
    branch_pred = branch_label * large_cd
    label_value_dic = dict(zip(*np.unique(branch_label, return_counts=True)))
    pred_value, pred_count = np.unique(branch_pred, return_counts=True)
    for j in range(len(pred_value)):
        if pred_value[j] == 0: continue
        if pred_count[j] / label_value_dic[pred_value[j]] >= 0.8:
            detected_num += 1

    branch = detected_num / num_branch
    """
    branch = 0
    dice = 2 * (pred * label).sum() / ((pred + label).sum() + 1)

    print(name, "Length: %0.4f" % (sen), "Precision: %0.4f" % (pre), "Branch: %0.4f" % (branch),
          "Dice: %0.4f" % (dice))
    return sen, pre, branch, dice

def test():
    from WingsNet_Ori import WingsNet
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    file_path = "/data/data2/filenames.csv"
    data_root = "/data/data2/datasets/nifti/"
        # resume
    model_name = 'wingsnet_99'
    # save_dir = os.path.join('./result', folder_name, model_name)
    save_dir = "/data/data1/"
    create_directories(save_dir)
    weights_dict = torch.load(model_name)

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    model = WingsNet(in_channel=1, n_classes=1)

    valid_dataset = SegValCropData(file_path, data_root, batch_size=8, cube_size=[128,128,128], step=[64,64,64])
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False, num_workers=4,
                                  pin_memory=True, drop_last=True)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    last_name = ''
    flag = False
    with torch.no_grad():
        for i, (x, name, pos) in enumerate(valid_dataloader):

            name = name[0]

            if name != last_name:
                if last_name != '':
                    print(i, last_name)
                    pred = pred / pred_num
                    pred = pred[0, 0]
                    pred[pred >= 0.5] = 1
                    pred[pred < 0.5] = 0
                    pred = large_connected_domain(pred)
                    pred_img = sitk.GetImageFromArray(pred.astype(np.byte))
                    pred_img.SetOrigin(img.GetOrigin())
                    pred_img.SetDirection(img.GetDirection())
                    pred_img.SetSpacing(img.GetSpacing())
                    create_directories(os.path.join(save_dir, last_name))
                    sitk.WriteImage(pred_img, os.path.join(save_dir, last_name, 'airway.nii.gz'))

                img = sitk.ReadImage(os.path.join(data_root, name, 'image.nii.gz'))
                arr = sitk.GetArrayFromImage(img)
                sizex, sizey, sizez = arr.shape
                pred = np.zeros(arr.shape)
                pred = pred[np.newaxis, np.newaxis, ...]
                pred_num = np.zeros(pred.shape)
                last_name = name

            x = x.cuda()
            # print(f'x shape: {x.shape}')
            # print(pos)
            p0, p = model(x)
            p = torch.sigmoid(p)
            p = p.cpu().detach().numpy()
            pos = pos.numpy()
            
            for i in range(len(pos)):
                # 
                xl, xr, yl, yr, zl, zr = pos[i, 0], pos[i, 1], pos[i, 2], pos[i, 3], pos[i, 4], pos[i, 5]
                if xr > sizex:
                    dietax = xr - sizex
                    xr = sizex
                else:
                    dietax = 0
                if yr > sizey:
                    dietay = yr - sizey
                    yr = sizey
                else:
                    dietay = 0
                if zr > sizez:
                    dietaz = zr - sizez
                    zr = sizez
                else:
                    dietaz = 0

                pred[0, :, xl:xr, yl:yr, zl:zr] += p[i][:, 0:128-dietax, 0:128-dietay, 0:128-dietaz]
                pred_num[0, :, xl:xr, yl:yr, zl:zr] += 1

        pred = pred / pred_num
        pred = pred[0, 0]
        
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = large_connected_domain(pred)
        
        pred_img = sitk.GetImageFromArray(pred.astype(np.byte))
        pred_img.SetOrigin(img.GetOrigin())
        pred_img.SetDirection(img.GetDirection())
        pred_img.SetSpacing(img.GetSpacing())

        create_directories(os.path.join(save_dir, last_name))
        sitk.WriteImage(pred_img, os.path.join(save_dir, last_name, 'airway.nii.gz'))

if __name__ == '__main__':
    
    # train()
    test()

    




























