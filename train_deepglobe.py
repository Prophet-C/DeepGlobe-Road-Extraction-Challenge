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
# from networks.dinknet import DinkNet34
from framework import MyFrame
from loss import dice_bce_loss
from dataloader.deepglobe_dataset import DeepGlobe
from networks.dlinknet_modify.dinknet_new import DinkNet34_cfilter, DinkNet34_gnconv
from networks.msp_se_dlinknet.dinknet_modified import DinkNet34_SP

@torch.no_grad()
def test_deepglobe(net, dataloader, save_result=False):
    evaluator = Evaluator(2)
    net.eval()
    evaluator.reset()
    data_iter = iter(dataloader)
    tbar = tqdm(data_iter)

    for img, mask, id in tbar:
        img = img.cuda()
        pred = net.forward(img).cpu().data.numpy()
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = mask.data.numpy()
        mask = mask.squeeze().astype(np.uint8)
        pred = pred.squeeze().astype(np.uint8)
        mask = np.where((mask==0)|(mask==1), mask^1, mask)
        pred = np.where((pred==0)|(pred==1), pred^1, pred)

        evaluator.add_batch_sklearn(mask, pred)

        if save_result: 
            for i in range (0, mask.shape[0]):
                img_dir = output_dir + '/images/'
                os.makedirs(img_dir, exist_ok=True)
                # img_2write = (img[i].cpu().data.numpy().transpose(1, 2, 0)+1.6)/3.2 *255
                # mask_2write = np.expand_dims(mask[i]*255, 2)
                # mask_2write = np.dstack((mask_2write, mask_2write, mask_2write))
                pred_2write = np.expand_dims(pred[i]*255, 2)
                pred_2write = np.dstack((pred_2write, pred_2write, pred_2write))
                # padding = 255*np.ones((mask[i].shape[0], 10, 1))
                # padding = np.dstack((padding, padding, padding))
                # temp = np.concatenate((img_2write, padding, mask_2write, padding, pred_2write), axis = 1)
                cv2.imwrite(img_dir +id[i]+'.png', pred_2write)

    class_index = 0
    class_index1 = 1
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    IoU = evaluator.Intersection_over_Union(class_index)
    IoU1 = evaluator.Intersection_over_Union(class_index1)
    Precision = evaluator.Pixel_Precision()
    Recall = evaluator.Pixel_Recall()
    F1 = evaluator.Pixel_F1()
    print("Val results:")
    print("Acc:{:.2f}, Acc_class:{:.2f}, mIoU:{:.2f}, IoU:{:.2f}(class{})/{:.2f}(class{}), Precision:{:.2f}, Recall:{:.2f}, F1:{:.2f}"
          .format(Acc*100, Acc_class*100, mIoU*100, IoU*100, class_index, IoU1*100, class_index1, Precision*100, Recall*100, F1*100))

    return IoU*100

if __name__ == '__main__':

    test_only = True

    output_dir = 'results/dink34_cfilter_[d3,d2,d1]'
    multi_gpu = False
    save_logger(output_dir)

    SHAPE = (512,512)

    WEIGHT_NAME = 'best'
    BATCHSIZE_PER_CARD = 16

    net = DinkNet34_cfilter()
    solver = MyFrame(net, dice_bce_loss, 2e-4)

    if multi_gpu:
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
    else: 
        batchsize = BATCHSIZE_PER_CARD


    with open('data/deepglobe/img/train_crop.txt') as file:
        imagelist = file.readlines()
    trainlist = list(map(lambda x: x[:-1], imagelist))
    train_dataset = DeepGlobe(trainlist)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)

    with open('data/deepglobe/img/offical_test.txt') as file:
        imagelist = file.readlines()
    validlist = list(map(lambda x: x[:-1], imagelist))
    val_dataset = DeepGlobe(validlist)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)
        
    timer = time()
    no_optim = 0
    total_epoch = 300
    train_epoch_best_loss = 100.
    val_best = 0
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(train_data_loader)
        train_epoch_loss = 0
        tbar = tqdm(data_loader_iter)
        for img, mask, index in tbar:
            solver.set_input(img, mask)
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
            IoU = test_deepglobe(solver.net, val_data_loader)
            if IoU > val_best:
                if val_best != 0:
                    os.remove(os.path.join(output_dir, 'val_best_{:.2f}.pth'.format(val_best)))
                val_best = IoU
                solver.save(os.path.join(output_dir, 'val_best_{:.2f}.pth'.format(val_best)))

        # import pdb
        # pdb.set_trace()

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
