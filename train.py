import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchio as tio
from unet import UNet3D

# Each instance of tio.Subject is passed arbitrary keyword arguments.
# Typically, these arguments will be instances of tio.Image
ct='LiTS/volume_pt1/volume-0.nii'
seg='LiTS/segmentations/segmentation-0.nii'
all_data = list()

print('Generating subjects list.')
for (dirpath, dirnames, filenames) in os.walk('LiTS'):
        all_data += [os.path.join(dirpath, file) for file in filenames ]
cts = sorted([ d for d in all_data if 'volume-' in d],
    key = lambda x : int(x.split('-')[-1].split('.')[0]))
segs = sorted([ d for d in all_data if 'segmentation' in d],
    key = lambda x : int(x.split('-')[-1].split('.')[0]))

subjects_list = [tio.Subject(
    ct=tio.ScalarImage(ct),
    seg=tio.LabelMap(seg)
    ) \
            for ct, seg in zip(cts, segs)
]

print('Done.')
#subjects_list = [tio.Subject(
#    ct=tio.ScalarImage(ct),
#    seg=tio.LabelMap(seg)
#    )
#]

esm = tio.EnsureShapeMultiple(4)
# Let's use one preprocessing transform and one augmentation transform
# This transform will be applied only to scalar images:
rescale = tio.RescaleIntensity(out_min_max=(0, 1))

# Convert multiclass problem to multilabel problem.
onehot = tio.OneHot()

# Transforms can be composed as in torchvision.transforms
transforms = [ tio.ToCanonical(), tio.Resample('ct'), tio.Pad((32, 32, 32)), esm, rescale, onehot ]
transform = tio.Compose(transforms)

# SubjectsDataset is a subclass of torch.data.utils.Dataset

print('Generating subjects dataset.')
subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)
print('Done.')

#sampler = tio.data.UniformSampler((200, 200, 32))
sampler = tio.data.LabelSampler(patch_size=64, 
        label_probabilities={0: 0, 1: 1, 2: 1} )
#queue_length = 300
#samples_per_volume = 10
queue_length = 1
samples_per_volume = 1

print('Generating sampler queue.')
sampler_queue = tio.data.Queue(subjects_dataset, 
        queue_length, samples_per_volume, sampler, num_workers=4)
print('Done.')

print('Loading data loader.')
# Images are processed in parallel thanks to a PyTorch DataLoader
training_loader = DataLoader(sampler_queue, batch_size=1, num_workers=0)
print('Done.')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

in_channels = 1
out_classes = 3

#model = UNet3D(in_channels, out_classes, residual=True, num_encoding_blocks=3,
#        out_channels_first_layer=16)
## Move model to device before constructing optimizer.
#model = model.to(device)
#optimizer = optim.Adam(model.parameters(), lr=1e-4)
#
#loss = nn.BCEWithLogitsLoss()
# Training epoch
while True:
    print('Getting batch.')
    for subjects_batch in training_loader:
        inputs = subjects_batch['ct']['data']
        targets = subjects_batch['seg']['data']
        # The next line might not be necessary.
        # The OneHot operation outputs a (b, c, h, w, d) tensor where b is the
        # batch size and h, w, d are the resp. dimensions. c is the classes and
        # in this case c = 3. The (b, 0, h, w, d) is the complement of the labels.
        # That is, a given element is 1 if it was 0 in the labels (not 1 or 2).
        # This line inverts that matrix making it a union of the two labels.
        #targets[:, 0] = (~(targets[:, 0].int()))

        #inputs = inputs.to(device)
        #targets = targets.to(device).float()
        #outputs = model(inputs)
        #cur_loss = loss(outputs[:, 1:, :, :, :], targets[:, 1:, :, :, :])
        #cur_loss.backward()
        #optimizer.step()
        #print(cur_loss.data)
