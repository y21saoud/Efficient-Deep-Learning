import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from torchvision.datasets.utils import download_url
from torchvision.datasets import CIFAR100
#from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.facecolor'] = '#ffffff'


project_name='resnet-practice-cifar100-resnet'

from torchvision.datasets.utils import download_url


# Dowload the dataset
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz"
download_url(dataset_url, '.')

# Extract from archive
with tarfile.open('./cifar100.tgz', 'r:gz') as tar:
    tar.extractall(path='./data')
    
# Look into the data directory
data_dir = './data/cifar100'
print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)

# Data transforms (normalization & data augmentation)
#stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) - cifar10
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))      #cifar100

train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         # tt.RandomRotate
                         # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                         # tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

# PyTorch datasets
#train_ds = ImageFolder(data_dir+'/train', train_tfms)
#valid_ds = ImageFolder(data_dir+'/test', valid_tfms)
train_ds = CIFAR100(root = 'data/', download = True, transform = train_tfms)
valid_ds = CIFAR100(root = 'data/', train = False, transform = valid_tfms)

print(train_ds)
print(valid_ds)
print(len(train_ds))
print(len(valid_ds))
print('total classes:', len(train_ds.classes))
print(train_ds.classes)

batch_size = 16

# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(images, *stats)
        ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0,1))
        break


show_batch(train_dl)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    

device = get_default_device()
device

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input
    

simple_resnet = to_device(SimpleResidualBlock(), device)

for images, labels in train_dl:
    out = simple_resnet(images)
    print(out.shape)
    break
    
del simple_resnet, images, labels
torch.cuda.empty_cache()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
        

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 3 x 32 x 32
        self.conv1 = conv_block(in_channels, 64)         # 64 x 32 x 32
        self.conv2 = conv_block(64, 128, pool=True)      # 128 x 16 x 16
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128))  # 128 x 16 x 16
        
        self.conv3 = conv_block(128, 256, pool=True)    # 256 x 8 x 8
        self.conv4 = conv_block(256, 512, pool=True)    # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(512, 512), 
                                  conv_block(512, 512))  # 512 x 4 x 4
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), # 512 x 1 x 1
                                        nn.Flatten(),     # 512
                                        nn.Dropout(0.2),  
                                        nn.Linear(512, num_classes)) # 100
        
    def forward(self, xb):
        out1 = self.conv1(xb)
        out2 = self.conv2(out1)
        out3 = self.res1(out2) + out2
        out4 = self.conv3(out3)
        out5 = self.conv4(out4)
        out6 = self.res2(out5) + out5
        out = self.classifier(out6)
        return out



model = to_device(ResNet9(3, 100), device)
model

model.load_state_dict(torch.load('/users/local/m21cabon/first_train_CIFAR100/cifar100-resnet9.pth'))
model.eval()
print("done")

# call model eval before doing any evaluation - good practice
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()      
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

history = [evaluate(model, valid_dl)]
history


result = evaluate(model, valid_dl)

print(result)

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

conv_layer_to_prune = model.conv1[0]  # Access the Conv2d layer inside the first sequential block
"""
# Apply pruning to this layer
prune.random_unstructured(conv_layer_to_prune, name="weight", amount=0.1)

# Evaluate the model after pruning
result_after_pruning = evaluate(model, valid_dl)
print(result_after_pruning)

prune.l1_unstructured(conv_layer_to_prune, name="bias", amount=1)
# Evaluate the model after pruning
result_after_pruning = evaluate(model, valid_dl)
print(result_after_pruning)
"""

"""
prune.ln_structured(conv_layer_to_prune, name="weight", amount=0.6, n=2, dim=0)
result_after_pruning = evaluate(model, valid_dl)
print(result_after_pruning)
"""


parameters_to_prune = (
    # Pruning the Conv2d layers in each conv block
    (model.conv1[0], 'weight'),  # First Conv2d in conv1 block
    (model.conv2[0], 'weight'),  # First Conv2d in conv2 block
    (model.conv3[0], 'weight'),  # First Conv2d in conv3 block
    (model.conv4[0], 'weight'),  # First Conv2d in conv4 block

    # Pruning the Conv2d layers in the residual blocks
    (model.res1[0][0], 'weight'), # First Conv2d in the first block of res1
    (model.res1[1][0], 'weight'), # First Conv2d in the second block of res1
    (model.res2[0][0], 'weight'), # First Conv2d in the first block of res2
    (model.res2[1][0], 'weight'), # First Conv2d in the second block of res2
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.99,
)

result = evaluate(model, valid_dl)

print(result)

"""
epochs = 10
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)



train_time='23:00'

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

plot_accuracies(history)


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');


plot_losses(history)

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');


torch.save(model.state_dict(), '/users/local/m21cabon/first_train_CIFAR100/cifar100-resnet9_pruning.pth')

"""





our_quant=32
quant_factors={1:32,8:4,16:2,32:1}
quant_factor= quant_factors[our_quant]
sparsity=0.

def count_conv2d(m, x, y):
    x = x[0] # remove tuple

    fin = m.in_channels
    fout = m.out_channels
    sh, sw = m.kernel_size

    # ops per output element
    kernel_mul = sh * sw * fin
    kernel_add = sh * sw * fin - 1
    bias_ops = 1 if m.bias is not None else 0
    kernel_mul = kernel_mul/quant_factor
    ops = (kernel_mul + kernel_add)/m.groups + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    #Nice Formatting
    print("{:<10}: S_c={:<4}, F_in={:<4}, F_out={:<4}, P={:<5}, params={:<10}, operations={:<20}".format("Conv2d",sh,fin,fout,x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # print("Conv2d: S_c={}, F_in={}, F_out={}, P={}, params={}, operations={}".format(sh,fin,fout,x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0] # remove tuple

    nelements = x.numel()
    total_sub = 2*nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])
    #Nice Formatting
    print("{:<10}: S_c={:<4}, F_in={:<4}, F_out={:<4}, P={:<5}, params={:<10}, operations={:<20}".format("Batch norm",'x',x.size(1),'x',x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # print("Batch norm: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("ReLU: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),0,int(total_ops)))



def count_avgpool(m, x, y):
    x = x[0]
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("AvgPool: S={}, F_in={}, P={}, params={}, operations={}".format(m.kernel_size,x.size(1),x.size()[2:].numel(),0,int(total_ops)))

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features/quant_factor
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    print("Linear: F_in={}, F_out={}, params={}, operations={}".format(m.in_features,m.out_features,int(m.total_params.item()),int(total_ops)))
    m.total_ops += torch.Tensor([int(total_ops)])

def count_sequential(m, x, y):
    print ("Sequential: No additional parameters  / op")

# custom ops could be used to pass variable customized ratios for quantization
def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            # Count non-zero parameters and ensure it's on the CPU
            non_zero_params = torch.count_nonzero(p).to('cpu')
            m.total_params += non_zero_params / quant_factor # Adjust for quantization


        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.AvgPool2d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, nn.Sequential):
            m.register_forward_hook(count_sequential)
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size).to(device)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params

    return total_ops, total_params

    # Resnet18 - Reference for CIFAR 10
ref_params = 5586981
ref_flops  = 834362880
    # WideResnet-28-10 - Reference for CIFAR 100
    # ref_params = 36500000
    # ref_flops  = 10490000000

print(model)
flops, params = profile(model, (16,3,32,32))
flops, params = flops.item(), params.item()

score_flops = flops / ref_flops
score_params = (params / ref_params)*(1-sparsity)
score = score_flops + score_params
print("Flops: {}, Params: {}".format(flops,params))
print("Score flops: {} Score Params: {}".format(score_flops,score_params))
print("Final score: {}".format(score))
