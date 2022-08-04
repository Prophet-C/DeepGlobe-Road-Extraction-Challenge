from random import random
import torch 
from torch.utils.data import Dataset

import cv2
import numpy as np
import os


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, lpu, mask,  
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        lpu = cv2.warpPerspective(lpu, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))    
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, lpu, mask

def randomHorizontalFlip(image, lpu, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        lpu = cv2.flip(lpu, 1)
        mask = cv2.flip(mask, 1)

    return image, lpu, mask

def randomVerticleFlip(image, lpu, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        lpu = cv2.flip(lpu, 0)
        mask = cv2.flip(mask, 0)

    return image, lpu, mask

def randomRotate90(image, lpu, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        lpu=np.rot90(lpu)
        mask=np.rot90(mask)

    return image, lpu, mask

def randomRotate(image, lpu, mask, u=0.5):
    rotete_times = np.random.randint(4)
    image=np.rot90(image, rotete_times)
    lpu=np.rot90(lpu, rotete_times)
    mask=np.rot90(mask, rotete_times)

    return image, lpu, mask

def resize(image, lpu, mask, shape):
    image = cv2.resize(image, shape)
    lpu = cv2.resize(lpu, shape)
    mask = cv2.resize(mask, shape)

    return image, lpu, mask

def random_crop(image, lpu, mask, crop_size=[0.7, 0.9]):
    h, w, c = image.shape
    crop_size = crop_size[np.random.randint(len(crop_size))]
    h_crop = int(h * crop_size)
    w_crop = int(w * crop_size)

    starth = h//2-(h_crop//2)
    startw = w//2-(w_crop//2)
    image = image[startw:startw+w_crop,starth:starth+h_crop]
    lpu = lpu[startw:startw+w_crop,starth:starth+h_crop]
    mask = mask[startw:startw+w_crop,starth:starth+h_crop]

    return resize(image, lpu, mask, (h, w))

def multi_loader(id, root, val=False):
    img_root = os.path.join(root, 'rgb')
    lpu_root = os.path.join(root, 'depth_lpu')
    mask_root = os.path.join(root, 'mask')
    img = cv2.imread(os.path.join(img_root+'/{}.png').format(id))
    lpu = cv2.imread(os.path.join(lpu_root+'/{}.png').format(id))
    mask = cv2.imread(os.path.join(mask_root+'/{}.png').format(id), cv2.IMREAD_GRAYSCALE)
    img, lpu, mask = resize(img, lpu, mask, (512, 512))

    if not val:
        img = randomHueSaturationValue(img,
                                    hue_shift_limit=(-30, 30),
                                    sat_shift_limit=(-5, 5),
                                    val_shift_limit=(-15, 15))
        
        img, lpu, mask = randomShiftScaleRotate(img, lpu, mask,
                                        shift_limit=(-0.1, 0.1),
                                        scale_limit=(-0.1, 0.1),
                                        aspect_limit=(-0.1, 0.1),
                                        rotate_limit=(-0, 0))
        img, lpu, mask = randomHorizontalFlip(img, lpu, mask)
        img, lpu, mask = randomVerticleFlip(img, lpu, mask)
        img, lpu, mask = randomRotate90(img, lpu, mask)
    
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    lpu = np.array(lpu, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)#/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    #mask = abs(mask-1) 
    return img, lpu, mask

def simple_loader(id, root, val=False):
    img_root = os.path.join(root, 'rgb')
    lpu_root = os.path.join(root, 'depth_lpu')
    mask_root = os.path.join(root, 'mask')
    img = cv2.imread(os.path.join(img_root+'/{}.png').format(id))
    lpu = cv2.imread(os.path.join(lpu_root+'/{}.png').format(id))
    mask = cv2.imread(os.path.join(mask_root+'/{}.png').format(id), cv2.IMREAD_GRAYSCALE)
    img, lpu, mask = resize(img, lpu, mask, (512, 512))

    if not val:
        img, lpu, mask = randomHorizontalFlip(img, lpu, mask)
        img, lpu, mask = randomVerticleFlip(img, lpu, mask)
        img, lpu, mask = randomRotate(img, lpu, mask)
        img, lpu, mask = random_crop(img, lpu, mask)
    
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    lpu = np.array(lpu, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    return img, lpu, mask

class TLCGISDataset(Dataset):
    def __init__(self, trainlist, root, val=False, loader=multi_loader):
        self.ids = trainlist
        self.loader = loader
        self.root = root
        self.val = val

    def __getitem__(self, index):
        id = self.ids[index]
        if self.val:
            img, lpu, mask = self.loader(id, self.root, val=True)
        else:
            img, lpu, mask = self.loader(id, self.root)
        img = torch.Tensor(img)
        lpu = torch.Tensor(lpu)
        mask = torch.Tensor(mask)
        return img, lpu, mask, id

    def __len__(self):
        return len(self.ids)