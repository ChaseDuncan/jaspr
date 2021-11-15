import torch
import torchio as tio
from torch.utils.data import DataLoader

# Each instance of tio.Subject is passed arbitrary keyword arguments.
# Typically, these arguments will be instances of tio.Image
t1ce='/shared/mrfil-data/cddunca2/brats2021/BraTS2021_00495/BraTS2021_00495_t1ce.nii.gz'
t1='/shared/mrfil-data/cddunca2/brats2021/BraTS2021_00495/BraTS2021_00495_t1.nii.gz'
t2='/shared/mrfil-data/cddunca2/brats2021/BraTS2021_00495/BraTS2021_00495_t2.nii.gz'
flair='/shared/mrfil-data/cddunca2/brats2021/BraTS2021_00495/BraTS2021_00495_flair.nii.gz'
seg='/shared/mrfil-data/cddunca2/brats2021/BraTS2021_00495/BraTS2021_00495_seg.nii.gz'

subject_a = tio.Subject(
    t1ce=tio.ScalarImage(t1ce),
    t1=tio.ScalarImage(t1),
    t2=tio.ScalarImage(t2),
    flair=tio.ScalarImage(flair),
    seg=tio.LabelMap(seg)
)

subjects_list = [subject_a]

# Let's use one preprocessing transform and one augmentation transform
# This transform will be applied only to scalar images:
rescale = tio.RescaleIntensity(out_min_max=(0, 1))

# As RandomAffine is faster then RandomElasticDeformation, we choose to
# apply RandomAffine 80% of the times and RandomElasticDeformation the rest
# Also, there is a 25% chance that none of them will be applied
spatial = tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    },
    p=0.75,
)

# Transforms can be composed as in torchvision.transforms
transforms = [rescale, spatial]
transform = tio.Compose(transforms)

# SubjectsDataset is a subclass of torch.data.utils.Dataset
subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)

# Images are processed in parallel thanks to a PyTorch DataLoader
training_loader = DataLoader(subjects_dataset, batch_size=4, num_workers=4)

# Training epoch
for subjects_batch in training_loader:
    inputs = torch.cat([subjects_batch['t1ce'][tio.DATA],
                       subjects_batch['t1'][tio.DATA],
                       subjects_batch['t2'][tio.DATA],
                       subjects_batch['flair'][tio.DATA]], 1)
    targets = subjects_batch['seg']

