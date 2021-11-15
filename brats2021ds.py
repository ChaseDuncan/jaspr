import torch
import os
from torch.utils.data import Dataset

class BraTS2021Dataset(Dataset):
    def __init__(self, datadirectory):
        assert os.path.isdir(datadirectory), f'{datadirectory} is not a directory.'

        self.ids = [os.path.join(datadirectory, d) for d in os.listdir(datadirectory) \
                if os.path.isdir(os.path.join(datadirectory, d)) ]

        assert len(self.ids), f'No data found. Check path name {datadirectory}'

    def __len__(self):
        'Return size of the dataset.'
        return len(self.ids)

    def load_ds(self, path):
        'Load raw nii.gz data. Transform into pt tensors.'
        assert os.path.isdir(path)
        raw_files = [os.path.join(path, f) for f in os.listdir(path) if f[-6:] == 'nii.gz']


    def __getitem__(self, index):
        self.load_ds(self.ids[index])


if __name__=='__main__':
    bds = BraTS2021Dataset('/shared/mrfil-data/cddunca2/brats2021/')

    from torch.utils.data import DataLoader
    dataloader = DataLoader(bds)
    for d in dataloader:
        pass
