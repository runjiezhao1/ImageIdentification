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
    
class ImprovedNet_Typ3(nn.Module):
    def __init__(self):
        super(ImprovedNet_Typ3, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.batch1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64,128,3)
        self.batch2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,3)
        self.batch3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*2*2,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.drop1 = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.pool(F.relu(self.batch1(self.conv1(x))))
        x = self.pool(F.relu(self.batch2(self.conv2(x))))
        x = self.pool(F.relu(self.batch3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = self.fc3(x)
        return x
    
class ImprovedNet_Typ4(nn.Module):
    def __init__(self):
        super(ImprovedNet_Typ4, self).__init__()
        self.pool = nn.MaxPool2d(2,2,padding=1)

        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.batch1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128,128,3,padding=1)
        self.batch2 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128,128,3,padding=1)
        self.conv6 = nn.Conv2d(128,128,3,padding=1)
        self.conv7 = nn.Conv2d(128,128,1,padding=1)
        self.batch3 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128,256,3,padding=1)
        self.conv9 = nn.Conv2d(256,256,3,padding=1)
        self.conv10 = nn.Conv2d(256,256,1,padding=1)
        self.batch4 = nn.BatchNorm2d(256)

        self.conv11 = nn.Conv2d(256,512,3,padding=1)
        self.conv12 = nn.Conv2d(512,512,3,padding=1)
        self.conv13 = nn.Conv2d(512,512,1,padding=1)
        self.batch5 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512*4*4,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,10)
        self.drop1 = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.batch1(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.batch2(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool(x)
        x = self.batch3(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool(x)
        x = self.batch4(x)
        x = F.relu(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool(x)
        x = self.batch5(x)
        x = F.relu(x)

        x = x.view(-1,512*4*4)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = self.fc3(x)
        return x