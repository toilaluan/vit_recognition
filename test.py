from email.mime import image
from model import ClassifyModel
import torch
import torch.nn as nn
from model import DigitRecognitionModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from dataset import MnistSequenceDataset
import matplotlib.pyplot as plt
import numpy as np
transform = transforms.Compose([
  transforms.Resize(32),
  transforms.ToTensor(),
  transforms.Normalize((0.5), (0.5)),
])
device = torch.device('cpu')
data = MnistSequenceDataset(transform=transform, train=False, n_samples=10000)
index = np.random.randint(0, 500)
sample = data[index]['image']
print(data[index]['label'])
model = DigitRecognitionModel(depth=2)
model.load_state_dict(torch.load("saved_model/model.pth"))
sample = torch.unsqueeze(sample, 0)
print(sample.shape)
output = model(sample)
print(torch.argmax(output, dim=1))
plt.imshow(sample[0][0])
plt.show()
