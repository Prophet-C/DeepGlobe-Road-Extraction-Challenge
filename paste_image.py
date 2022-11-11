import argparse
import os
import mmap
import cv2
import time
import numpy as np
from skimage import io
from tqdm import tqdm
tqdm.monitor_interval = 0


def paste():
    folders = ['results/dink34_test_dataloader/images',]

    for folder in folders:
        new_folder = folder+'_paste'
        os.makedirs(new_folder, exist_ok=True)

            
        with open('data/deepglobe/img/offical_test.txt') as file:
                imagelist = file.readlines()
        image_id = list(map(lambda x: x[:-1], imagelist))

        for id in tqdm(image_id):
            id = id.split('/')[-1].replace('.jpg', '').replace('_sat', '')

            img_0_0 = cv2.imread(folder+'/'+id+'_0_0_sat.png', cv2.IMREAD_COLOR)
            img_0_1 = cv2.imread(folder+'/'+id+'_0_1_sat.png', cv2.IMREAD_COLOR)
            img_1_0 = cv2.imread(folder+'/'+id+'_1_0_sat.png', cv2.IMREAD_COLOR)
            img_1_1 = cv2.imread(folder+'/'+id+'_1_1_sat.png', cv2.IMREAD_COLOR)


            img_0 = np.concatenate((img_0_0, img_1_0), axis = 0)
            img_1 = np.concatenate((img_0_1, img_1_1), axis = 0)

            img = np.concatenate((img_0, img_1), axis = 1)

            
            cv2.imwrite(new_folder+'/'+id+'_mask.png', img)

            # import pdb
            # pdb.set_trace()

            
            
                     

if __name__ == "__main__":
    paste()