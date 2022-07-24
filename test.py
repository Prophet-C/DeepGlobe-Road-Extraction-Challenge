import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time
from tqdm import tqdm
from evaluator import Evaluator
from logger import save_logger
from networks.dinknet import DinkNet34
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder

@torch.no_grad()
def test(net, dataloader):
    evaluator = Evaluator(2)
    net.eval()
    evaluator.reset()
    data_iter = iter(dataloader)
    tbar = tqdm(data_iter)
    
    index = 0
    for img, mask, id in tbar:
        pred = net.forward(img).cpu().data.numpy()
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0     
        mask = mask.data.numpy()

        mask = mask.squeeze().astype(np.uint8)
        pred = pred.squeeze().astype(np.uint8)
        mask = np.where((mask==0)|(mask==1), mask^1, mask)
        pred = np.where((pred==0)|(pred==1), pred^1, pred)
        
        evaluator.add_batch_sklearn(mask, pred)
        img_dir = output_dir + '/images/'
        os.makedirs(img_dir, exist_ok=True)
        temp = np.concatenate((mask*255, 255*np.ones((mask.shape[0], 10)),pred*255), axis = 1)
        cv2.imwrite(img_dir +id[0]+'_compare.png',temp)
        index = index + 1
        

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    IoU = evaluator.Intersection_over_Union(class_index=1)
    Precision = evaluator.Pixel_Precision()
    Recall = evaluator.Pixel_Recall()
    F1 = evaluator.Pixel_F1()
    print("Val results:")
    print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
          .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1))


output_dir = 'results/dink34_rgb_only_exp0'
save_logger(output_dir, filename="log_val.txt")

SHAPE = (512,512)
ROOT = 'dataset/TLCGIS/'

WEIGHT_NAME = 'best'
BATCHSIZE_PER_CARD = 8

solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
batchsize = 1

with open(os.path.join(ROOT, 'valid.txt')) as file:
    imagelist = file.readlines()
validlist = list(map(lambda x: x[:-1], imagelist))
val_dataset = ImageFolder(validlist, ROOT, val=True)
val_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=0)

solver.load(os.path.join(output_dir, 'train_best.pth'))
net = solver.net
test(net, val_data_loader)