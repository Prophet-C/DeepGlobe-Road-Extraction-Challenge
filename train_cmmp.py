import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time
from tqdm import tqdm
import random
from utils.evaluator import Evaluator
from utils.logger import save_logger
from networks.dinknet import DinkNet34
from networks.dinknet_cmmp import DinkNet34CMMP
from networks.dinknet_cmmp_gconv import dlinknet_cmmp_gconv
from framework import MyFrame, FusionFrame
from loss import dice_bce_loss, bce_loss
from dataloader.rgb_dataset import ImageFolder
from dataloader.fusion_dataset import TLCGISDataset, multi_loader, simple_loader
from test_cmmp import test


if __name__ == '__main__':
    
    multi_gpu = False
    data_loader = multi_loader
    output_dir = 'results/dink34_fusion_exp7_test'
    loss_func = dice_bce_loss
    save_logger(output_dir, force_merge=True)

    SHAPE = (512,512)
    ROOT = 'data/TLCGIS/'

    WEIGHT_NAME = 'best'
    BATCHSIZE_PER_CARD = 16

    net = dlinknet_cmmp_gconv()
    solver = FusionFrame(net, loss_func, 2e-4, multi_gpu=multi_gpu)

    if multi_gpu:
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
    else: 
        batchsize = BATCHSIZE_PER_CARD

    with open(os.path.join(ROOT, 'train.txt')) as file:
        imagelist = file.readlines()
    trainlist = list(map(lambda x: x[:-1], imagelist))
    train_dataset = TLCGISDataset(trainlist, ROOT, loader=data_loader)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=16)

    with open(os.path.join(ROOT, 'valid.txt')) as file:
        imagelist = file.readlines()
    validlist = list(map(lambda x: x[:-1], imagelist))
    val_dataset = TLCGISDataset(validlist, ROOT, val=True, loader=data_loader)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=16)
        
    timer = time()
    no_optim = 0
    total_epoch = 300
    train_epoch_best_loss = 100.
    val_best = 0
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(train_data_loader)
        train_epoch_loss = 0
        tbar = tqdm(data_loader_iter)
        for img, lpu, mask, index in tbar:
            solver.set_input(img, lpu, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)
        print('********')
        print('epoch:',epoch,'    time:',int(time()-timer))
        print('train_loss:',train_epoch_loss)
        print('SHAPE:',SHAPE)
        
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            print("Save latest model:")
            solver.save(os.path.join(output_dir, 'train_best.pth'))
            print("Start evaluation on val dataset:")
            IoU = test(solver.net, val_data_loader, save_result=False)
            if IoU > val_best:
                if val_best != 0:
                    os.remove(os.path.join(output_dir, 'val_best_{:.2f}.pth'.format(val_best)))
                val_best = IoU
                solver.save(os.path.join(output_dir, 'val_best_{:.2f}.pth'.format(val_best)))

        if no_optim > 6:
            print('early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load(os.path.join(output_dir, 'train_best.pth'))
            solver.update_lr(5.0, factor = True)

    print('Finish!')
