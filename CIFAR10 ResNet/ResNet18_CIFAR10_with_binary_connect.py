

from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
import resnet
import binaryconnect

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def validation_step(model, batch):
    images, labels =  batch[0].to(device), batch[1].to(device)
    input_var = torch.autograd.Variable(images, volatile=True)
    target_var = torch.autograd.Variable(labels, volatile=True)

    out = model(input_var)                    # Generate predictions
    loss = F.cross_entropy(out.data, target_var)   # Calculate loss
    acc = accuracy(out.data, labels)      # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def epoch_end(model, epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['avg_lr'], result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))  # Update this line

    
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()      
    bin_op.binarization() 
    outputs = [validation_step(model, batch) for batch in val_loader]
    bin_op.restore()
    return validation_epoch_end(model, outputs)

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network.# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each
transform_train = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
normalize_scratch,
])

transform_test = transforms.Compose([
transforms.ToTensor(),
normalize_scratch,
])

### The data from CIFAR10 will be downloaded in the following folder
rootdir = '/users/local/m21cabon/data/cifar10'
c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

"""
#for testing
subset_indices = list(range(100))  # This will take the first 100 samples
c10train_subset = Subset(c10train, subset_indices)
trainloader = DataLoader(c10train_subset, batch_size=32, shuffle=True)
"""
trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32)

## number of target samples for the final dataset
num_train_examples = len(c10train)

## We set a seed manually so as to reproduce the results easily
seed = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)

print(f"Initial CIFAR10 dataset has {len(c10train)} samples")

### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train

hparam_currentvalue = 0
criterion = F.cross_entropy


# Initialize ResNet18
net = resnet.ResNet18()  # pretrained=True if you want to use pre-trained weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
bin_op=binaryconnect.BC(net)

epochs =200
max_lr = 0.01
weight_decay = 1e-4
grad_clip=0.1
lr=0.001
momentum=0.95

best_train_acc = 0
# Early stopping parameters
patience = 100  # Number of epochs to wait for improvement before stopping
patience_counter = 0

# Set up cutom optimizer with weight decay
optimizer = torch.optim.SGD(net.parameters(), lr,
                                momentum=momentum, 
                                weight_decay=weight_decay)

history = []
# Train the model
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_losses = []
    train_accuracies = []

    lrs = []
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(labels)
        bin_op.binarization()

        outputs = net(input_var)

        loss = criterion(outputs, target_var)
        train_losses.append(loss)
        acc = accuracy(outputs.data, labels)
        train_accuracies.append(acc)

        optimizer.zero_grad()
        loss.backward()
        bin_op.restore()
        optimizer.step()
        bin_op.clip()

        lrs.append(get_lr(optimizer))
    

        running_loss += loss.item()
        if i % 2000 == 1999:    
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # Evaluate the model on test data
    # Validation phase
    result = evaluate(net, testloader)
    result['train_loss'] = torch.stack(train_losses).mean().item()
    result['train_acc'] = torch.stack(train_accuracies).mean().item()  # Add this line
    result['avg_lr'] = sum(lrs) / len(lrs)
    epoch_end(net, epoch, result)
    history.append(result)

        # Check for early stopping
    if result['train_acc'] > best_train_acc:
        best_train_acc = result['train_acc']
        patience_counter = 0
        # Save the model if it's the best so far
        torch.save(net.state_dict(), '/users/local/m21cabon/cifar10-resnet18_bestBC.pth')
    else:
        patience_counter += 1
        if patience_counter > patience:
            print("Stopping early due to no improvement in validation loss")
            break


timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
directory = "/homes/m21cabon/EDL/CIFAR10_ResNet/results/"
filename = f"{directory}training_history_ResNet18_BC_{timestamp}.json"

# Convert the history to a JSON string and write it to a file
with open(filename, 'w') as f:
    json.dump(history, f, indent=4)

print(f"Training history saved to {filename}")