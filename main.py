import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

#choose the model
model = models.ImprovedNet()


torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#iterate for 50 times
for epoch in range(50):
    running_loss = 0.0
    # i range from 0 to 2000
    for i, data in enumerate(train_data_loader, 0):
        #input size [128,3,32,32]
        input, labels = data
        #clear the gradient from the previous result
        print(input.shape[0])
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 0:
            print('Epoch: %d, step: %d, loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

correct = 0
total = 0
with torch.no_grad():
    for data in test_data_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#torch.save(model.state_dict(), "./temp.json")