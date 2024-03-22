import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import models
import dataProcess as dp
import matplotlib.pyplot as plt
import numpy as np

#apply transform to the data
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor() 
])

transform_valid = transforms.Compose([
    transforms.ToTensor() 
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_valid, download=True)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

#size of train dataset is 391
#size of test dataset is 79 

#choose the model
model = models.ImprovedNet()

#train on gpu
if not torch.cuda.is_available():
    print('CUDA not available')
else:
    print('CUDA available')

torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: ", torch.device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#record the history data
history = []

#iterate for 50 times
for epoch in range(50):
    # Train data
    running_loss = 0.0
    avg_loss = 0.0
    train_correct = 0.0
    train_total = 0.0
    for i, data in enumerate(train_data_loader, 0):
        #input size [128,3,32,32]
        input, labels = data
        #preprocessing
        updated_input = dp.image_normalize(input.numpy())
        input = torch.from_numpy(updated_input)
        #clear the gradient from the previous result
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #predict the value
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        if i == 390:
            print('Epoch: %d, step: %d, loss: %.3f' % (epoch + 1, i + 1, running_loss / 390))
            avg_loss = running_loss / 390.0
            running_loss = 0.0
    train_accuracy_rate = train_correct / train_total
    # Validation data
    correct = 0
    total = 0
    valid_loss = 0.0
    avg_valid_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_data_loader, 0):
            images, labels = data
            #preprocessing the data
            updated_input = dp.image_normalize(images.numpy())
            input = torch.from_numpy(updated_input)
            outputs = model(images)
            #get current loss
            loss = criterion(outputs, labels)
            #loss.backward()
            valid_loss += loss.item()
            if i == 78:
                avg_valid_loss = valid_loss / 78.0
                valid_loss = 0
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %% at epoch %d' % (100 * correct / total, epoch + 1))
    valid_accuracy_rate = correct / total
    history.append([avg_loss,avg_valid_loss,train_accuracy_rate,valid_accuracy_rate])

# Plot the figure
# Plot the loss
history = np.array(history)
plt.figure(figsize=(10, 10))
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
#Plot the accuracy rate
plt.figure(figsize=(10, 10))
plt.plot(history[:,2:4])
plt.legend(['Tr Accur', 'Val Accur'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy rate')
plt.show()

torch.save(model.state_dict(), "./temp.json")

# 0.58 -> 0.63 -> 0.71