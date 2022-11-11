import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
from dataloader.fusion_dataset import TLCGISDataset, multi_loader


def get_dataloader(imagelist_path, batchsize, DATA_ROOT = 'data/TLCGIS/', data_loader=multi_loader):
    with open(imagelist_path) as file:
        imagelist = file.readlines()
    imagelist = list(map(lambda x: x[:-1], imagelist))

    dataset = TLCGISDataset(imagelist, DATA_ROOT, loader=data_loader)
    sampler = DistributedSampler(dataset, num_replicas=2,
                                              rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_size=batchsize,
                            num_workers=16,
                            sampler=sampler,
                            shuffle=False)

def main():

    pass