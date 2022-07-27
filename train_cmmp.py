import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time
from tqdm import tqdm
from utils.evaluator import Evaluator
from utils.logger import save_logger
from networks.dinknet import DinkNet34
from networks.dinknet_cmmp import DinkNet34CMMP
from framework import MyFrame, FusionFrame
from loss import dice_bce_loss
from dataloader.data import ImageFolder
from dataloader.dataloader import TLCGISDataset
from test_cmmp import test


output_dir = 'results/dink34_fusion_exp1'
save_logger(output_dir, force_merge=True)

SHAPE = (512,512)
ROOT = 'dataset/TLCGIS/'
# imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
# trainlist = map(lambda x: x[:-8], imagelist)

WEIGHT_NAME = 'best'
BATCHSIZE_PER_CARD = 4

solver = FusionFrame(DinkNet34CMMP, dice_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

with open(os.path.join(ROOT, 'train.txt')) as file:
    imagelist = file.readlines()
trainlist = list(map(lambda x: x[:-1], imagelist))
train_dataset = TLCGISDataset(trainlist, ROOT)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)

with open(os.path.join(ROOT, 'train.txt')) as file:
    imagelist = file.readlines()
validlist = list(map(lambda x: x[:-1], imagelist))
val_dataset = TLCGISDataset(validlist, ROOT, val=False)
val_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=0)
    
timer = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.
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
        test(solver.net, val_data_loader, save_result=False)
    if no_optim > 6:
        print('early stop at %d epoch' % epoch)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load(os.path.join(output_dir, 'train_best.pth'))
        solver.update_lr(5.0, factor = True)
    

    import pdb
    pdb.set_trace()
print('Finish!')
