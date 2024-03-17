import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64,128,3)
        self.conv3 = nn.Conv2d(128,256,3)
        self.fc1 = nn.Linear(256*2*2,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ImprovedNet_Type1(nn.Module):
    def __init__(self):
        super(ImprovedNet_Type1, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3, padding = 1)
        self.conv3 = nn.Conv2d(32,64,3, padding = 1)
        self.fc1 = nn.Linear(64*4*4,512)
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,10)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class ImprovedNet_Type2(nn.Module):
    def __init__(self):
        super(ImprovedNet_Type2, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64,128,3)
        self.conv3 = nn.Conv2d(128,256,3)
        self.fc1 = nn.Linear(256*2*2,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x