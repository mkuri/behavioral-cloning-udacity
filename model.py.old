import pandas as pd
import numpy as np
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import time

from dataset import ImageSteeringDataset


class Nvidia(nn.Module):
    def __init__(self):
        super(Nvidia, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, (5, 5), stride=2)
        self.conv2 = nn.Conv2d(24, 36, (5, 5), stride=2)
        self.conv3 = nn.Conv2d(36, 48, (5, 5), stride=2)
        self.conv4 = nn.Conv2d(48, 64, (3, 3), stride=1)
        self.conv5 = nn.Conv2d(64, 64, (3, 3), stride=1)
        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc4(x)
        return x
                

def split_train_test_dataset(dataset, ratio):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size])
    return train_dataset, test_dataset


def train(model, dataset):
    print('>>> Train model ...')
    train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=4,
            shuffle=True, num_workers=2)
    n_epoch = 10
    lr = 0.0001

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epoch):
        running_loss = 0.0

        for i, (features, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('>>> outputs')
                print(outputs)
                print('>>> labels')
                print(labels)
                print(loss)
                print('[%d, %5d] loss: %.6f' %
                        (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0

    return model


def main():
    print('>>> Initialize ...')
    dataset = ImageSteeringDataset('./data/driving_log.csv', True)

    train_dataset, test_dataset = split_train_test_dataset(
            dataset, ratio=0.7)

    model = Nvidia()
    model = train(model, train_dataset)
        


if __name__ == "__main__":
    main()
