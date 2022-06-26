from email.mime import image
from genericpath import exists
from pkgutil import get_data
from model import ClassifyModel
import torch
import torch.nn as nn
from model import DigitRecognitionModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from dataset import MnistSequenceDataset
transform = transforms.Compose([
  transforms.Resize(32),
  transforms.ToTensor(),
  transforms.Normalize((0.5), (0.5)),
])
device = torch.device('cuda')
data = MnistSequenceDataset(transform=transform, train=True, n_samples=10000)

train_loader = DataLoader(data, 4, True)

model = DigitRecognitionModel(depth=2)
print(model)

def train(model = None):
  model.train()
  device = torch.device("cuda")
  model.to(device)
  lr = 0.0001
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  epochs = 3
  total_loss = 0
  total_acc = 0
  for epoch in range(epochs):
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
      img = batch['image'].to(device)
      label = batch['label'].to(device).long()
      # print(label)
      optimizer.zero_grad()
      outputs = model(img)
      loss = criterion(outputs, label)
      # print(loss)
      loss.backward()
      optimizer.step()
      epoch_loss += loss
      if i % 100 == 0:
        print("Loss : {}".format(loss))
    print("Epoch : {}, Loss : {}".format(epoch, epoch_loss/len(train_loader)))
  torch.save(model.state_dict(), "saved_model/model.pth")
train(model=model)