#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2023/12/13 17:11:16
@Author  :   Carry
@Version :   1.0
@Desc    :   None
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchaudio.transforms as T
import torchaudio
import glob
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


def load_and_preprocess_data(path):
    """
    加载音频文件并对其进行预处理
    """
    waveform, sample_rate = torchaudio.load(path)

    desired_length = 16000 * 3  # 统一截取3秒
    current_length = waveform.shape[1]
    if current_length < desired_length:
        # 填充
        padding = torch.zeros((1, desired_length - current_length))
        padded_waveform = torch.cat((waveform, padding), dim=1)
    else:
        # 截断
        padded_waveform = waveform[:, :desired_length]
    return padded_waveform


class GenderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []
        self.transform = transform

        self._load_data()

    def _load_data(self):
        male_files = glob.glob(os.path.join(self.root_dir + "/male/*/*.wav"))
        female_files = glob.glob(os.path.join(self.root_dir + "/female/*/*.wav"))

        self.file_paths.extend(male_files)
        self.labels.extend([0] * len(male_files))  # 0 represents male

        self.file_paths.extend(female_files)
        self.labels.extend([1] * len(female_files))  # 1 represents female
        a = 0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        # label = torch.tensor(self.labels[idx])
        label = self.labels[idx]

        padded_waveform = load_and_preprocess_data(audio_path)
        if self.transform:
            padded_waveform = self.transform(padded_waveform)
            print(f"after transform: {padded_waveform.shape}")  # 128, 241
        return padded_waveform, label


class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 60, 128)  # 修改这里
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # 2 classes for male and female

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 60)  # 修改这里
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Result: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


# Example of using transforms
transform = transforms.Compose(
    [
        T.MelSpectrogram(16000),  # 这个转换将音频波形转换为梅尔频谱图。sample_rate 参数指定了音频的采样率。
        T.FrequencyMasking(freq_mask_param=30),  # 这个转换对梅尔频谱图进行频率掩码。freq_mask_param 参数控制掩码的频率范围。
        T.TimeMasking(time_mask_param=100),  # 这个转换对梅尔频谱图进行时间掩码。time_mask_param 参数控制掩码的时间范围。
    ]
)
# Create datasets and dataloaders
train_dataloader = GenderDataset(root_dir="/datasets/gender_dataset/train", transform=transform)
test_dataloader = GenderDataset(root_dir="/datasets/gender_dataset/test", transform=transform)

train_loader = DataLoader(train_dataloader, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataloader, batch_size=32, shuffle=False)


model = GenderClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def main():
    epochs = 20
    best = 0.0
    for t in tqdm(range(epochs)):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        accuracy, avg_loss = test(test_loader, model, loss_fn)
        if float(accuracy) > best:
            best = float(accuracy)
            torch.save(model.state_dict(), f"model_epoch_{t+1}_acc_{round(accuracy, 4)}.pth")


if __name__ == "__main__":
    main()
