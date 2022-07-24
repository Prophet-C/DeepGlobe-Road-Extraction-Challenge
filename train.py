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

    for img, mask in tbar:
        pred = net.forward(img).cpu().data.numpy().squeeze(1)
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
        evaluator.add_batch(mask, pred)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    IoU = evaluator.Intersection_over_Union()
    Precision = evaluator.Pixel_Precision()
    Recall = evaluator.Pixel_Recall()
    F1 = evaluator.Pixel_F1()
    print("Val results:")
    print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
          .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1))

output_dir = 'results/dink34_rgb_only_exp0'
save_logger(output_dir)

SHAPE = (512,512)
ROOT = 'dataset/TLCGIS/'
# imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
# trainlist = map(lambda x: x[:-8], imagelist)

WEIGHT_NAME = 'best'
BATCHSIZE_PER_CARD = 8

solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

with open(os.path.join(ROOT, 'train.txt')) as file:
    imagelist = file.readlines()
trainlist = list(map(lambda x: x[:-1], imagelist))
train_dataset = ImageFolder(trainlist, ROOT)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)

with open(os.path.join(ROOT, 'valid.txt')) as file:
    imagelist = file.readlines()
validlist = list(map(lambda x: x[:-1], imagelist))
val_dataset = ImageFolder(validlist, ROOT)
val_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)
    
tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(train_data_loader)
    train_epoch_loss = 0
    tbar = tqdm(data_loader_iter)
    for img, mask in tbar:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print('********')
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    print('SHAPE:',SHAPE)
    
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss

        solver.save(os.path.join(output_dir, 'train_best.pth'))
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
