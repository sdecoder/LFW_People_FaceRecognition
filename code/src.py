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

from utils import num_epoches, train_loader, N, custom_dset, Cnn, lr

if len(sys.argv) > 1:
  if sys.argv[1] == '--load':
    weightname = sys.argv[2]
    tempna = '../models'
    name = os.path.join(tempna, weightname)
    test_only = 1
  else:
    weightname = sys.argv[2]
    tempna = './'
    name = tempna + weightname
    test_only = 0
else:
  weightname = 'cnn.pt'
  tempna = '../models'
  name = os.path.join(tempna, weightname)
  test_only = 0

def train(net):

  print(f'[trace] step into train function')
  optimizer = torch.optim.Adam(net.parameters(), lr)
  loss_func = nn.BCELoss()
  l_his = []
  for epoch in range(num_epoches):
    print('Epoch:', epoch + 1, 'Training...')
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      image1s, image2s, labels = data
      if torch.cuda.is_available():
        image1s = image1s.cuda()
        image2s = image2s.cuda()
        labels = labels.cuda()

      image1s, image2s, labels = Variable(image1s), Variable(image2s), Variable(labels.float())
      optimizer.zero_grad()
      f = net(image1s, image2s)
      f = torch.squeeze(f)
      loss = loss_func(f, labels)
      loss.backward()
      optimizer.step()
      if i % 20 == 19:
        l_his.append(loss.cpu().data)
        print(f'[trace] current dataset index: {i}')
      # print statistics
      running_loss += loss.data
      if i % 100 == 99:
        print(f'[trace] current dataset index: {i}')
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0

  print(f'[trace] Finished Training, saving network to {name}')
  torch.save(net.state_dict(), name)
  fig = plt.figure()
  ax = plt.subplot(111)
  ax.plot(l_his)
  plt.xlabel('Steps')
  plt.ylabel('Loss')
  fig.savefig('../resources/plotad.png')
  pass


def test(net):
  net.load_state_dict(torch.load(name))

  # test data
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  test_set = custom_dset('./lfw', './train.txt', transform, transform)
  test_loader = DataLoader(test_set, batch_size=N, shuffle=False, num_workers=2)

  correct = 0
  total = 0
  for i, data in enumerate(test_loader, 0):
    image1s, image2s, labels = data
    if torch.cuda.is_available():
      image1s = image1s.cuda()
      image2s = image2s.cuda()
      labels = labels.cuda()
    image1s, image2s, labels = Variable(image1s), Variable(image2s), Variable(labels.float())
    # print(labels)
    outputs = net(image1s, image2s)
    outputs = outputs.cpu()
    for j in range(outputs.size()[0]):
      if ((outputs.data.numpy()[j] < 0.5)):
        if labels.data.cpu().numpy()[j] == 0:
          correct += 1
          total += 1
        else:
          total += 1
      else:
        if labels.data.cpu().numpy()[j] == 1:
          correct += 1
          total += 1
        else:
          total += 1

  print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

  test_set = custom_dset('./lfw', './test.txt', transform, transform)
  test_loader = DataLoader(test_set, batch_size=N, shuffle=False, num_workers=2)

  correct = 0
  total = 0
  for i, data in enumerate(test_loader, 0):
    image1s, image2s, labels = data
    if torch.cuda.is_available():
      image1s = image1s.cuda()
      image2s = image2s.cuda()
      labels = labels.cuda()
    image1s, image2s, labels = Variable(image1s), Variable(image2s), Variable(labels.float())
    # print(labels)
    outputs = net(image1s, image2s)
    outputs = outputs.cpu()
    for j in range(outputs.size()[0]):
      if ((outputs.data.numpy()[j] < 0.5)):
        if labels.data.cpu().numpy()[j] == 0:
          correct += 1
          total += 1
        else:
          total += 1
      else:
        if labels.data.cpu().numpy()[j] == 1:
          correct += 1
          total += 1
        else:
          total += 1

  print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
  pass


def main():

  print(f'[trace] start main function')
  print(f'[trace] torch.version: {torch.__version__}')

  net = Cnn()
  if torch.cuda.is_available():
    net = net.cuda()

  if test_only == 0:
    train(net)
  else:
    test(net)
  pass


if __name__ == '__main__':
  main()
  pass
