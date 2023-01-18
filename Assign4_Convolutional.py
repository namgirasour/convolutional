#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:08:54 2022

@author: Bi
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import torch.nn.functional as F
import numpy as np


train_dataset = datasets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = datasets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

batch_size = 64
num_epochs = int(10)
# print(num_epochs)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10) 

    def forward(self, x):

        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.log_softmax(out, dim=1)
        return out
    
model = CNN()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  


iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.requires_grad_()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        iter += 1
        
        labels = []
        if iter % 500 == 0:
        
            correct = 0
            total = 0
            predicted_list = []
            labels_list = []
           

            for images, labels in test_loader:
                images = images.requires_grad_()
         
                outputs = model(images)
                # outputs = outputs.detach().numpy()
                # print(outputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                correct += (predicted == labels).sum()
                
                predicted = predicted.detach().numpy()
                predicted_list = np.append(predicted_list, predicted)
                labels_list = np.append(labels_list, labels)
                print(labels_list)
                
                # print(predicted_list)
                # print(predicted)
                recall = recall_score(predicted_list, labels_list, average='macro')
                print(recall)
                precision = precision_score(predicted_list, labels_list, average='macro')
                print(precision)
                
                
                

            accuracy = 100 * correct / total
            


            
            print('Accuracy: {}'.format(accuracy))