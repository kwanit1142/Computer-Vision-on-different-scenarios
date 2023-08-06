import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import PIL
from tqdm import tqdm
import cv2
from sklearn.metrics import classification_report, roc_curve
import pandas as pd
import re
import string
from sklearn.preprocessing import label_binarize

def transform_fn(img):
  img = img.resize((128,128))
  tensor = transforms.ToTensor()(img)
  return tensor


def model_return(ckpt_path, model_obj):
  ckpt = torch.load(ckpt_path)
  model_obj.load_state_dict(ckpt['state_dict'])
  return model_obj.eval().cuda()

def confusion_matrix(pred,true):
  conf_matrix = np.zeros((10,10),dtype='uint8')
  for i in tqdm(range(len(pred))):
    conf_matrix[pred[i]][true[i]]+=1
  print(conf_matrix)

def accuracy(pred,true):
  class_vector = np.zeros((1,10))
  total_vector = np.zeros((1,10))
  total_correct = 0
  for i in range(len(pred)):
    if pred[i]==true[i]:
      total_correct+=1
      class_vector[0][true[i]]+=1
    total_vector[0][true[i]]+=1
  print("\nOverall Accuracy = ", 100*total_correct/len(pred), "%")
  print("\nClass-Wise Accuracies = ", 100*class_vector/total_vector)
  print(classification_report(pred,true))

def detect(test_loader, model_class):
  pred=[]
  true=[]
  proba=[]
  for i,j in tqdm(test_loader):
    i,j = i.cuda(), j.cuda()
    prob = model_class(i)
    out = prob.max(1, keepdim=True)[1]
    pred.append(out.detach().cpu().item())
    true.append(j.detach().cpu().item())
    proba.append(prob.detach().cpu().numpy())
  print("Confusion Matrix:\n")
  confusion_matrix(pred, true)
  accuracy(pred, true)
  del pred
  MultiClass_ROC(np.array(proba), true)
  del proba, true

def MultiClass_ROC(proba, true):
  fpr = {}
  tpr = {}
  thresh ={}
  n_class = 10
  plt.figure(figsize=(10,10))
  for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(true, [proba[j][0,i] for j in range(proba.shape[0])], pos_label=i)
    plt.plot(fpr[i], tpr[i], label='Class '+str(i)+' vs Rest')
  plt.title('Multiclass ROC curve')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive rate')
  plt.legend(loc='best')
  plt.show()

  class CNN_8Layer_Max(pl.LightningModule):
  
  def __init__(self, num_classes):
      super().__init__()
      self.model = nn.Sequential(
          nn.Conv2d(1,4,5),
          nn.MaxPool2d(2),
          nn.Conv2d(4,5,5),
          nn.MaxPool2d(2),
          nn.Conv2d(5,6,4),
          nn.MaxPool2d(2),
          nn.Flatten(),
          nn.Linear(1014,512),
          nn.Linear(512,num_classes)
      )
      self.loss = nn.CrossEntropyLoss()
      self.loss_accumulate=0
      self.loss_list=[]
  
  def forward(self, x):
      return self.model(x)
  
  def training_step(self, batch, batch_no):
      x, y = batch
      logits = self(x)
      loss = self.loss(logits, y)
      self.loss_accumulate+=loss
      if batch_no==599:
        self.loss_list.append(self.loss_accumulate)
        self.loss_accumulate=0
      return loss
  
  def loss_graph(self):
      plt.plot([x for x in range(0,20)],[y.detach().cpu() for y in self.loss_list])
      plt.show()
  
  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(),lr=0.001)

class CNN_8Layer_Power(pl.LightningModule):
  
  def __init__(self, num_classes):
      super().__init__()
      self.model = nn.Sequential(
          nn.Conv2d(1,4,5),
          nn.LPPool2d(2,2),
          nn.Conv2d(4,5,5),
          nn.LPPool2d(2,2),
          nn.Conv2d(5,6,4),
          nn.LPPool2d(2,2),
          nn.Flatten(),
          nn.Linear(1014,512),
          nn.Linear(512,num_classes)
      )
      self.loss = nn.CrossEntropyLoss()
      self.loss_accumulate=0
      self.loss_list=[]
  
  def forward(self, x):
      return self.model(x)
  
  def training_step(self, batch, batch_no):
      x, y = batch
      logits = self(x)
      loss = self.loss(logits, y)
      self.loss_accumulate+=loss
      if batch_no==599:
        self.loss_list.append(self.loss_accumulate)
        self.loss_accumulate=0
      return loss
  
  def loss_graph(self):
      plt.plot([x for x in range(0,20)],[y.detach().cpu() for y in self.loss_list])
      plt.show()
  
  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(),lr=0.001)

class CNN_8Layer_Avg(pl.LightningModule):
  
  def __init__(self, num_classes):
      super().__init__()
      self.model = nn.Sequential(
          nn.Conv2d(1,4,5),
          nn.AvgPool2d(2),
          nn.Conv2d(4,5,5),
          nn.AvgPool2d(2),
          nn.Conv2d(5,6,4),
          nn.AvgPool2d(2),
          nn.Flatten(),
          nn.Linear(1014,512),
          nn.Linear(512,num_classes)
      )
      self.loss = nn.CrossEntropyLoss()
      self.loss_accumulate=0
      self.loss_list=[]
  
  def forward(self, x):
      return self.model(x)
  
  def training_step(self, batch, batch_no):
      x, y = batch
      logits = self(x)
      loss = self.loss(logits, y)
      self.loss_accumulate+=loss
      if batch_no==599:
        self.loss_list.append(self.loss_accumulate)
        self.loss_accumulate=0
      return loss
  
  def loss_graph(self):
      plt.plot([x for x in range(0,20)],[y.detach().cpu() for y in self.loss_list])
      plt.show()
  
  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(),lr=0.001)

class Model(pl.LightningModule):

  def __init__(self, num_classes):
      super().__init__()
      self.model = models.resnet18(weights=None)
      self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      self.features = self.model.fc.in_features 
      self.model.fc = nn.Linear(self.features,num_classes)
      self.loss = nn.CrossEntropyLoss()
      self.loss_accumulate=0
      self.loss_list=[]
  
  def forward(self, x):
      return self.model(x)
  
  def training_step(self, batch, batch_no):
      x, y = batch
      logits = self(x)
      loss = self.loss(logits, y)
      self.loss_accumulate+=loss
      if batch_no==599:
        self.loss_list.append(self.loss_accumulate)
        self.loss_accumulate=0
      return loss
  
  def loss_graph(self):
      plt.plot([x for x in range(0,20)],[y.detach().cpu() for y in self.loss_list])
      plt.show()
  
  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(),lr=0.001)

train_MNIST = torch.utils.data.DataLoader(dataset=datasets.MNIST('../data', train=True, download=True, transform=transform_fn),batch_size=100, shuffle=True, pin_memory=True, num_workers=2)
test_MNIST = torch.utils.data.DataLoader(dataset=datasets.MNIST('../data', train=False, download=True, transform=transform_fn),batch_size=1, shuffle=True)

max_model = CNN_8Layer_Max(10).train().cuda()
trainer = pl.Trainer(accelerator='gpu', max_epochs=20, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Q1/MaxPool', benchmark=True)
trainer.fit(max_model, train_MNIST)
max_model.loss_graph()
del trainer, max_model

Model_Test = model_return('/content/drive/MyDrive/CV_Assignment_4/Q1/MaxPool/lightning_logs/version_1/checkpoints/epoch=19-step=12000.ckpt', CNN_8Layer_Max(10))
detect(test_MNIST, Model_Test)
del Model_Test

LP_model = CNN_8Layer_Power(10).train().cuda()
trainer = pl.Trainer(accelerator='gpu', max_epochs=20, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Q1/LPPool', benchmark=True)
trainer.fit(LP_model, train_MNIST)
LP_model.loss_graph()
del trainer, LP_model

Model_Test = model_return('/content/drive/MyDrive/CV_Assignment_4/Q1/LPPool/lightning_logs/version_0/checkpoints/epoch=19-step=12000.ckpt', CNN_8Layer_Power(10))
detect(test_MNIST, Model_Test)
del Model_Test

avg_model = CNN_8Layer_Avg(10).train().cuda()
trainer = pl.Trainer(accelerator='gpu', max_epochs=20, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Q1/AvgPool', benchmark=True)
trainer.fit(avg_model, train_MNIST)
avg_model.loss_graph()
del trainer, avg_model

Model_Test = model_return('/content/drive/MyDrive/CV_Assignment_4/Q1/AvgPool/lightning_logs/version_0/checkpoints/epoch=19-step=12000.ckpt', CNN_8Layer_Avg(10))
detect(test_MNIST, Model_Test)
del Model_Test

resnet = Model(10).train().cuda()
trainer = pl.Trainer(accelerator='gpu', max_epochs=20, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Q1/ResNet18', benchmark=True)
trainer.fit(resnet, train_MNIST)
resnet.loss_graph()
del trainer, resnet

Model_Test = model_return('/content/drive/MyDrive/CV_Assignment_4/Q1/ResNet18/lightning_logs/version_1/checkpoints/epoch=19-step=12000.ckpt', Model(10))
detect(test_MNIST, Model_Test)
del Model_Test