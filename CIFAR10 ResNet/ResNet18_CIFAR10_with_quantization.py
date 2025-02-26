

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
import torch.nn.utils.prune as prune
import copy
from torch.quantization import quantize_dynamic


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def validation_step(model, batch):
    images, labels = batch[0].to('cpu'), batch[1].to('cpu')  # Move data to CPU
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)  # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}

def validation_step_half(model, batch):
    images, labels = batch[0].half().to('cpu'), batch[1].half().to('cpu')  # Convert images to half precision
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)  # Calculate accuracy
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
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(model, outputs)

@torch.no_grad()
def evaluate_half(model, val_loader):
    model.eval()      
    outputs = [validation_step_half(model, batch) for batch in val_loader]
    return validation_epoch_end(model, outputs)

def prune_model(model, pruning_ratio):
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio,
    )
    
    # Make pruning permanent
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')


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
net.to('cpu')
net.load_state_dict(torch.load('/users/local/m21cabon/cifar10-resnet18_best.pth', map_location='cpu'))



#without quantization
print("###################### without quantization")
result = evaluate(net, testloader)
print(result)





#quantization 16bits
print("###################### with quantization 16bits")
# Apply dynamic quantization
net_quantized16 = quantize_dynamic(
    net,  # the original model
    {nn.Linear, nn.Conv2d},  # specify the layer types to quantize
    dtype=torch.float16  # specify the quantization data type
)
net_quantized16 = net_quantized16.to('cpu')
result = evaluate(net_quantized16, testloader)
print(result)



#quantization 8bits
print("###################### with quantization 8bits")
# Apply dynamic quantization
net_quantized = quantize_dynamic(
    net,  # the original model
    {nn.Linear, nn.Conv2d},  # specify the layer types to quantize
    dtype=torch.qint8  # specify the quantization data type
)
net_quantized = net_quantized.to('cpu')



result = evaluate(net_quantized, testloader)
print(result)

torch.save(net_quantized.state_dict(), '/users/local/m21cabon/cifar10-resnet18_with_quantization.pth')
