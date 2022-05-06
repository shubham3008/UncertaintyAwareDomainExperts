import os
from random import seed
import numpy as np
import torch
import argparse
import torch.utils
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable

model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=True)
model1 = model1.cuda()

imgdata = 'linnaeus'
dir = os.path.join('Datasets',imgdata)

if imgdata == 'animal_classification' or imgdata == 'vehicle_classification':
    train_data = dset.ImageFolder(
        dir,
        transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ]))
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=len(train_data)//4, shuffle=False, pin_memory=False)

if imgdata == 'linnaeus':
    train_dir = os.path.join(dir,'train')
    valid_dir = os.path.join(dir,'test')
    train_data = dset.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ]))
    valid_data = dset.ImageFolder(
        valid_dir,
        transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=False, pin_memory=False)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=32, shuffle=False, pin_memory=False)

data_vector1 = torch.zeros((100,)).cuda()

for step, (input, target) in enumerate(train_queue):
    with torch.no_grad():
        input = Variable(input).cuda()
        data_vector1 += model1(input).mean(axis=0)

data_vector1 = data_vector1/(step+1)
data_vector1_str = imgdata + ': '+ str(data_vector1.cpu().numpy())
print(data_vector1_str)
# np.savetxt('data_vector1.txt', data_vector1_str)