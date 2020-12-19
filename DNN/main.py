import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from metrics import MetricTracker
from model import ConvNet
from dataset import FSDD

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

EPOCHS = 60
BATCH_SIZE = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = FSDD('./free-spoken-digit-dataset/recordings', transform=transform)

# 80-15-15 train, val, test split
trainset, valset, testset = torch.utils.data.random_split(dataset, [
                                                          2100, 450, 450])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)


def get_val_loss():
    with torch.no_grad():
        running_loss = 0.
        for i, (images, labels) in enumerate(valloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        running_loss /= i
    return running_loss


def get_acc(dataloader):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for (images, labels) in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


net = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

mt = MetricTracker()


for epoch in range(EPOCHS):
    net.train()

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 19))

            running_loss = 0.0

    # Append stats to metric tracker
    mt.append(loss.item(), 'train_loss')
    mt.append(get_val_loss(), 'val_loss')
    mt.append(get_acc(trainloader), 'train_acc')
    mt.append(get_acc(valloader), 'val_acc')

    # Validation Accuracy every 5 epochs
    if epoch % 5 == 4:
        print(f'[Val Acc] {(100 * get_acc(valloader))}%')


print('Finished Training')
mt.plot_acc()
mt.plot_loss()

print(f'[Test Acc] {(100 * get_acc(testloader))}%')
