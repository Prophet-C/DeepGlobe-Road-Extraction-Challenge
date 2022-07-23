

from torch.utils.data import Dataset


class TLCGISDataset(Dataset):
    def __init__(self, list_pth) -> None:
        super(TLCGISDataset, self).__init__()

        self.list_pth = list_pth

    def __getitem__(self, index):
        pass 