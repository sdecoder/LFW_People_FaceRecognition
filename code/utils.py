
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np
import os
import random
import sys

# hyper parameters definition
N = 16
lr = 1e-5
num_epoches = 50


class Cnn(nn.Module):
  def __init__(self):
    super(Cnn, self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 64, 5, 1, 2),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(2, 2),
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(64, 128, 5, 1, 2),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.MaxPool2d(2, 2),
    )
    self.conv3 = nn.Sequential(
      nn.Conv2d(128, 256, 3, 1, 1),
      nn.ReLU(),
      nn.BatchNorm2d(256),
      nn.MaxPool2d(2, 2),
    )
    self.conv4 = nn.Sequential(
      nn.Conv2d(256, 512, 3, 1, 1),
      nn.ReLU(),
      nn.BatchNorm2d(512),
    )
    self.fc1 = nn.Sequential(
      nn.Linear(131072, 1024),
      nn.ReLU(),
      nn.BatchNorm1d(1024),
      #nn.BatchNorm2d(1024),
    )
    self.fc2 = nn.Sequential(
      nn.Linear(2048, 1),
      nn.Sigmoid(),
    )

  def forward(self, x, y):

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    y = self.conv1(y)
    y = self.conv2(y)
    y = self.conv3(y)
    y = self.conv4(y)
    y = y.view(y.size(0), -1)
    y = self.fc1(y)
    f = torch.cat((x, y), 1)
    f = self.fc2(f)
    return f

class custom_dset(Dataset):
  def __init__(self,
               img_path,
               txt_path,
               img_transform1,
               img_transform2,
               ):
    with open(txt_path, 'r') as f:
      lines = f.readlines()
      self.img1_list = [
        os.path.join(img_path, i.split()[0]) for i in lines
      ]
      self.img2_list = [
        os.path.join(img_path, i.split()[1]) for i in lines
      ]
      self.label_list = [i.split()[2] for i in lines]
    self.img_transform1 = img_transform1
    self.img_transform2 = img_transform2

  def __getitem__(self, index):
    img1_path = self.img1_list[index]
    img2_path = self.img2_list[index]
    label = self.label_list[index]
    label = int(label)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = img1.astype(np.float) / 255
    img2 = img2.astype(np.float) / 255
    img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (128, 128), interpolation=cv2.INTER_AREA)
    img1 = self.img_transform1(img1)
    img2 = self.img_transform2(img2)

    return img1, img2, label

  def __len__(self):
    return len(self.label_list)


class Rescale(object):
  def __call__(self, img):
    if random.random() < 0.7:
      f = round(0.1 * random.randint(7, 13), 2)
      if f > 1:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        a = int(round((f * 128 - 128) / 2))
        img = img[a:a + 128, a:a + 128]
      else:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        a = int(round((128 - f * 128) / 2))
        temp = np.zeros([128, 128, 3], dtype=np.uint8)
        temp.fill(0)
        for i in range(img.shape[0]):
          for j in range(img.shape[1]):
            temp[i + a, j + a] = img[i, j]
        img = temp
    return img


class Flip(object):
  def __call__(self, img):
    if random.random() < 0.7:
      return cv2.flip(img, 1)
    return img


class Rotate(object):
  def __call__(self, img):
    if random.random() < 0.7:
      angle = random.random() * 60 - 30
      rows, cols, cn = img.shape
      M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
      img = cv2.warpAffine(img, M, (cols, rows))
      return img
    return img


class Translate(object):
  def __call__(self, img):
    if random.random() < 0.7:
      x = random.random() * 20 - 10
      y = random.random() * 20 - 10
      rows, cols, cn = img.shape
      M = np.float32([[1, 0, x], [0, 1, y]])
      img = cv2.warpAffine(img, M, (cols, rows))
    return img

class ToFloat32(object):
  def __call__(self, img):
    if img.dtype is torch.float32:
      return img
    return img.to(torch.float32)

transform1 = transforms.Compose([Rescale(), Flip(), Translate(), Rotate(), transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ToFloat32()])
transform2 = transforms.Compose([Rescale(), Flip(), Translate(), Rotate(), transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ToFloat32()])
train_set = custom_dset('../data/lfw_funneled', '../data/lfw_funneled/train.txt', transform1, transform2)
#train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=1)
train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=10)
