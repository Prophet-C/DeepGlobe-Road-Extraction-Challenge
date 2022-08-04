import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

view_list = 'data/TLCGIS/view.txt'
data_path = 'data/TLCGIS/rgb'

with open(view_list) as file:
    lines = file.readlines()

file_list = []
for line in lines:
    file_list.append(line.strip('\n'))

pic_lines = np.split(np.array(file_list), 20)


for pic_line in pic_lines:
    run_once = True
    for block in pic_line:
        img_pth = os.path.join(data_path, block+'.png')
        img = cv2.imread(img_pth)
        plt.imshow(img)
        plt.show()

        import pdb
        pdb.set_trace()

