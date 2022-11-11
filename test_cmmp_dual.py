import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import pdb
from time import time
from tqdm import tqdm
from utils.evaluator import Evaluator
from utils.logger import save_logger
from networks.dinknet_cmmp import DinkNet34CMMP
from networks.dinknet_cmmp_gconv import dlinknet_cmmp_gconv, dlinknet_cmmp_gconv_new_param, dlinknet_cmmp_gconv_new_param_gf, dlinknet_cmmp_gconv_old_param_gf
from framework import MyFrame, FusionFrame
from loss import dice_bce_loss
from dataloader.rgb_dataset import ImageFolder
from dataloader.fusion_dataset import TLCGISDataset

@torch.no_grad()
def test_dual(net1, net2, dataloader, save_result=False):
    evaluator = Evaluator(2)
    net1.eval()
    net2.eval()
    evaluator.reset()
    data_iter = iter(dataloader)
    tbar = tqdm(data_iter)
    
    count = 1
    for img, lpu, mask, id in tbar:
        img = img.cuda()
        lpu = lpu.cuda()
        pred1 = net1.forward(img, lpu).cpu().data.numpy()
        pred2 = net2.forward(img, lpu).cpu().data.numpy()
        pred1[pred1>0.5] = 1
        pred1[pred1<=0.5] = 0     
        pred2[pred2>0.5] = 1
        pred2[pred2<=0.5] = 0     
        mask = mask.data.numpy()

        mask = mask.squeeze().astype(np.uint8)
        pred1 = pred1.squeeze().astype(np.uint8)
        pred2 = pred2.squeeze().astype(np.uint8)
        mask = np.where((mask==0)|(mask==1), mask^1, mask)
        pred1 = np.where((pred1==0)|(pred1==1), pred1^1, pred1)
        pred2 = np.where((pred2==0)|(pred2==1), pred2^1, pred2)
        
        # evaluator.add_batch_sklearn(mask, pred)

        run_once = False

        if save_result:
            for i in range (0, mask.shape[0]):
                img_dir = output_dir + '/images/'
                os.makedirs(img_dir, exist_ok=True)
                img_2write = (img[i].cpu().data.numpy().transpose(1, 2, 0)+1.6)/3.2 *255
                lpu_2write = (lpu[i].cpu().data.numpy().transpose(1, 2, 0)+1.6)/3.2 *255
                mask_2write = np.expand_dims(mask[i]*255, 2)
                mask_2write = np.dstack((mask_2write, mask_2write, mask_2write))

                pred_2write1 = np.expand_dims(pred1[i]*255, 2)
                pred_2write1 = np.dstack((pred_2write1, pred_2write1, pred_2write1))

                pred_2write2 = np.expand_dims(pred2[i]*255, 2)
                pred_2write2 = np.dstack((pred_2write2, pred_2write2, pred_2write2))

                padding = 255*np.ones((mask[i].shape[0], 10, 1))
                padding = np.dstack((padding, padding, padding))
                
               

                temp = np.concatenate((img_2write, padding, lpu_2write, padding, mask_2write, padding, pred_2write1, padding, pred_2write2), axis = 1)

                if not run_once:
                    total = temp
                    run_once = True
                else:
                    padding = 255*np.ones((10, total.shape[1], 1))
                    padding = np.dstack((padding, padding, padding))
                    total = np.concatenate((total, padding, temp), axis = 0)

        cv2.imwrite(img_dir + str(count) + '.png',total)
        count = count + 1

        
    # class_index = 0
    # class_index1 = 1
    # Acc = evaluator.Pixel_Accuracy()
    # Acc_class = evaluator.Pixel_Accuracy_Class()
    # mIoU = evaluator.Mean_Intersection_over_Union()
    # IoU = evaluator.Intersection_over_Union(class_index)
    # IoU1 = evaluator.Intersection_over_Union(class_index1)
    # Precision = evaluator.Pixel_Precision()
    # Recall = evaluator.Pixel_Recall()
    # F1 = evaluator.Pixel_F1()
    # print("Val results:")
    # print("Acc:{:.2f}, Acc_class:{:.2f}, mIoU:{:.2f}, IoU:{:.2f}(class{})/{:.2f}(class{}), Precision:{:.2f}, Recall:{:.2f}, F1:{:.2f}"
    #       .format(Acc*100, Acc_class*100, mIoU*100, IoU*100, class_index, IoU1*100, class_index1, Precision*100, Recall*100, F1*100))
    IoU = 0
    return IoU*100

if __name__ == '__main__':

    output_dir = 'results/compare_test_8_lpu'
    save_logger(output_dir, filename="log_test.txt")

    SHAPE = (512,512)
    ROOT = 'data/TLCGIS/'

    model_dir1 = 'results/dink34_fusion_exp1'
    model_dir2 = 'results/dink34_fusion_exp16_test_gconv_new_order'
    WEIGHT_NAME1 = 'train_best_86.70.pth'
    WEIGHT_NAME2 = 'val_best_88.04.pth'

    multi_gpu = False
    net1 = DinkNet34CMMP()
    solver1 = FusionFrame(net1, dice_bce_loss, 2e-4, multi_gpu, data_para=True)
    net2 = dlinknet_cmmp_gconv_new_param_gf()
    solver2 = FusionFrame(net2, dice_bce_loss, 2e-4, multi_gpu)
    batchsize = 8

    with open(os.path.join(ROOT, 'test.txt')) as file:
        imagelist = file.readlines()
    validlist = list(map(lambda x: x[:-1], imagelist))
    val_dataset = TLCGISDataset(validlist, ROOT, val=True)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0)

    solver1.load(os.path.join(model_dir1, WEIGHT_NAME1))
    net1 = solver1.net
    solver2.load(os.path.join(model_dir2, WEIGHT_NAME2))
    net2 = solver2.net
    test_dual(net1, net2, val_data_loader, True)