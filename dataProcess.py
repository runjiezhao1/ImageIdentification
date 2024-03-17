import numpy as np
import torchvision.transforms as tfs

def image_normalize(images):
    for i in range(images.shape[0]):
        old_images = images[i,:,:,:]
        mean = np.mean(old_images)
        std = max([np.std(old_images), 1.0 / np.sqrt(images.shape[1]*images.shape[2]*images.shape[3])])
        #make it Gaussian Distribution
        new_image = (old_images - mean) / std
        images[i,:,:,:] = new_image
    return images

###################
#used for testing

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

model = models.ImprovedNet()

torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for i, data in enumerate(train_data_loader, 0):
    input, labels = data
    #print(type(newForm))
    updated_input = image_normalize(input.numpy())
    input = torch.from_numpy(updated_input)