from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
idx2label = {x : str(x) for x in range(10)}
idx2label[10] = 'pad'
print(idx2label)
class MnistSequenceDataset(Dataset):
    def __init__(self, transform=None, train = True, n_samples = 48000, max_length = 16):
        self.data = datasets.EMNIST(root='data', train=train, download=True, transform=transform, split="letters")
        self.dataloader = DataLoader(self.data, batch_size = 8, shuffle=True)
        self.n_samples = n_samples
        self.max_length = max_length
        image_size = self.data.data.shape[-1]
        self.blank = torch.zeros((32, 32))
        self.sequene_data = self.make_sequence()
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"image" : self.sequene_data[0][idx], "label" : self.sequene_data[1][idx]}
        return sample
    def make_sequence(self):
        sequence_image = []
        sequence_labels = []
        while len(sequence_image) < self.n_samples:
            
            for images, labels in self.dataloader:

                if len(sequence_image) == self.n_samples:
                    break
                # print(labels.shape)
                print(labels)
                break
                tuple_img = (img for img in images)
                sequence_label = labels
                sequence_img = torch.concat([img for img in images], dim=2)
                while len(sequence_label) != self.max_length:
                    sequence_label = torch.cat((sequence_label, torch.tensor([10])), dim =0) 
                # print(sequence_label)
                random_slot = np.random.randint(0, 7, size=2)
                for index in random_slot:
                    # print(index)
                    # print(sequence_img[:,:, index*32 : (index+1)*32].shape)
                    # print(sequence_img.shape)
                    sequence_img[:,:, index*32 : (index+1)*32] = self.blank
                    sequence_label[index:index+1] = torch.tensor(10)
                sequence_image.append(sequence_img)
                sequence_labels.append(sequence_label)
                # plt.imshow(sequence_img[0])
                # plt.show()
                # print(sequence_img.shape)
        return sequence_image, sequence_labels
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
# dataset = MnistSequenceDataset(transform=transform, train=True, n_samples=1000)
# train_loader = DataLoader(dataset, 8, True)

# img, label = next(iter(train_loader))['image'], next(iter(train_loader))['label']

# print(img.shape)