#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Resize
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


# In[14]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[25]:


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DeformableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.offsets = nn.Parameter(torch.Tensor(kernel_size, kernel_size,2))
        nn.init.normal_(self.offsets)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        shifted_x = self.apply_offsets(x, self.offsets)
        output = self.conv(shifted_x)
        return output
    def apply_offsets(self, x, offsets):
        batch_size = x.size(0)
        offsets = offsets.repeat(batch_size, 1, 1, 1)
        kernel_torch = torch.randn(self.kernel_size,self.kernel_size)
        grid = torch.stack(torch.meshgrid(torch.arange(kernel_torch.shape[0]), torch.arange(kernel_torch.shape[1])), dim=-1).float()
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        grid = grid.to(device)
        offsets = offsets.to(device)
        shifted_grid = grid + offsets
        normalized_grid = shifted_grid / (x.shape[-2] - 1)
        shifted_x = F.grid_sample(x, normalized_grid, align_corners=False)
        return shifted_x


# In[4]:


def Dataloader():
    transform = Compose([Resize((224,224)),ToTensor(),  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes = [0,1]
    train_data = [x for x in train_data if x[1] in classes]
    test_data = [x for x in test_data if x[1] in classes]
    return train_data,test_data


# In[5]:


train_data,test_data = Dataloader()


# In[6]:


train = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


# In[26]:


model2 = models.alexnet(weights=True)
print(model2)


# In[ ]:





# In[30]:


model2.features[0] = DeformableConv2d(3,64,kernel_size=5,stride=2,padding=2)
model2.features[3] = DeformableConv2d(64,192,kernel_size=5,stride=1,padding=2)
model2.features[6] = DeformableConv2d(192,384,kernel_size=3,stride=1,padding=1)
model2.features[8] = DeformableConv2d(384,256,kernel_size=3,stride=1,padding=1)
model2.features[10] = DeformableConv2d(256,256,kernel_size=3,stride=1,padding=1)


# In[31]:


#model2 = nn.DataParallel(model2, device_ids=[0, 1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model2 = model2.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model2.parameters(), lr=0.0006)
train_loss = []
train_accuracy = []


# In[32]:


import time
start_time1 = time.time()
for epoch in range(10):
    epoch_train_loss = 0.0
    correct_pred = 0
    total_samples = 0
    for images, labels in train:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model2(images)
        _, predicted_labels = torch.max(out.data,1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        total_samples += labels.size(0)
        correct_pred += (predicted_labels == labels).sum().item()
    epoch_train_accuracy = 100 * correct_pred / total_samples
    train_accuracy.append(epoch_train_accuracy)
    train_loss.append(epoch_train_loss)
    print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")

end_time1 = time.time()
execution_time = end_time1 - start_time1
print("Execution time with Deformable Convolution:", execution_time, "seconds")


# In[41]:


model = models.alexnet(weights=True)
model = model.to(device)
criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(model.parameters(), lr=0.0006)
train_loss2 = []
train_accuracy2 = []


# In[42]:


start_time2 = time.time()
for epoch in range(10):
    epoch_train_loss = 0.0
    correct_pred = 0
    total_samples = 0
    for images, labels in train:
        images = images.to(device)
        labels = labels.to(device)
        optimizer2.zero_grad()
        out = model(images)
        _, predicted_labels = torch.max(out.data,1)
        loss = criterion2(out, labels)
        loss.backward()
        optimizer2.step()
        epoch_train_loss += loss.item()
        total_samples += labels.size(0)
        correct_pred += (predicted_labels == labels).sum().item()
    epoch_train_accuracy = 100 * correct_pred / total_samples
    train_accuracy2.append(epoch_train_accuracy)
    train_loss2.append(epoch_train_loss)
    print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")

end_time2 = time.time()
execution_time = end_time2 - start_time2
print("Execution time with Deformable Convolution:", execution_time, "seconds")


# In[43]:





# In[ ]:





# In[49]:


import torch
import torch.nn as nn
import torchvision

class DCN(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super(DCN, self).__init__()
    self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_channels))
    else:
      self.register_parameter('bias', None)
    self.offset_channels = 2 * kernel_size * kernel_size
    self.mask_channels = kernel_size * kernel_size
    self.conv_offset_mask = nn.Conv2d(in_channels, self.offset_channels + self.mask_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.normal_(self.weight, std=0.01)
    nn.init.normal_(self.conv_offset_mask.weight, std=0.01)
    if self.bias is not None:
      nn.init.constant_(self.bias, 0)
    nn.init.constant_(self.conv_offset_mask.bias, 0)

  def forward(self, x):
    offset_mask = self.conv_offset_mask(x)
    offset = offset_mask[:, :self.offset_channels, :, :]
    mask = offset_mask[:, self.offset_channels:, :, :]
    mask = torch.sigmoid(mask)
    out = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.weight, bias=self.bias, mask=mask)
    return out


# In[50]:


model3.features[0]= DCN(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
model3.features[3] = DCN(in_channels=64, out_channels=192, kernel_size=11, stride=4, padding=2)
model3.features[3] = DCN(in_channels=192, out_channels=384, kernel_size=11, stride=4, padding=2)
model3.features[8] = DCN(in_channels=384, out_channels=256, kernel_size=11, stride=4, padding=2)
model3.features[10] = DCN(in_channels=256, out_channels=256, kernel_size=11, stride=4, padding=2)


# In[52]:


model3 = models.alexnet(weights=True)
model3 = model3.to(device)
criterion3 = nn.CrossEntropyLoss()
optimizer3 = optim.Adam(model3.parameters(), lr=0.0006)
train_loss3 = []
train_accuracy3 = []


# In[53]:


start_time3 = time.time()
for epoch in range(10):
    epoch_train_loss = 0.0
    correct_pred = 0
    total_samples = 0
    for images, labels in train:
        images = images.to(device)
        labels = labels.to(device)
        optimizer3.zero_grad()
        out = model3(images)
        _, predicted_labels = torch.max(out.data,1)
        loss = criterion3(out, labels)
        loss.backward()
        optimizer3.step()
        epoch_train_loss += loss.item()
        total_samples += labels.size(0)
        correct_pred += (predicted_labels == labels).sum().item()
    epoch_train_accuracy = 100 * correct_pred / total_samples
    train_accuracy2.append(epoch_train_accuracy)
    train_loss2.append(epoch_train_loss)
    print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")

end_time3 = time.time()
execution_time = end_time3 - start_time3
print("Execution time with Deformable Convolution:", execution_time, "seconds")


# In[ ]:




