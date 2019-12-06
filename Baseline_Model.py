#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import h5py

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import numpy as np

hdf5_filename = "testData/assignment1_data-1.hdf5"


# In[10]:


# Read files
with h5py.File(hdf5_filename, "r") as f:
    dset = f["mydataset"]


# In[11]:


print(dset.shape)


# In[3]:


torch.manual_seed(42)    # reproducible

# load data
label = [] # N by 1, where N is the number of samples
X = [] # N by F, where F is the number of features

# this is one way to define a network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

# set constants for network
net = Net(n_feature=3, n_hidden1=20, n_hidden2=10, n_output=1)     # define the network
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# train the network
for t in range(200):
    prediction = net(X)     # input x and predict based on x
    loss = loss_func(prediction, label)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    

