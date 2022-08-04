import os
import cv2
import numpy as np


# create new masks where road is 1 and background is 0
root = 'data/TLCGIS/'
folder = 'mask'
new_folder = 'new_mask'

with open(os.path.join(root, 'valid.txt')) as file:
    imagelist = file.readlines()

for file in imagelist:
    mask = cv2.imread(os.path.join(root, folder, '{}.png'.format(file[:-1])), cv2.IMREAD_GRAYSCALE)
    mask = np.where((mask==0)|(mask==1), mask^1, mask)
    cv2.imwrite(os.path.join(root, new_folder, '{}.png'.format(file[:-1])), mask*255)