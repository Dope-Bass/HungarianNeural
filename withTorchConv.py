import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose

import sys
from HungarianNeural import parse

m_size = 5
device = torch.device("cpu")


class newToTensor(object):
    def __call__(self, data):
        np_data = np.array(data)

        return torch.from_numpy(np_data).to(device)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.conv3 = nn.Conv2d(64, 128, 2)

    def forward(self, x):
        x = torch.relu(self.fullyConnectedLayer1(x))
        x = torch.relu(self.fullyConnectedLayer2(x))
        x = torch.relu(self.fullyConnectedLayer3(x))
        x = self.fullyConnectedLayer4(x)
        return f.log_softmax(x, dim=1)


class MatrixDataset(Dataset):

    def __init__(self, net_index, samples, transform=None):
        super().__init__()
        self.rows = []
        self.labels = []
        for sample in range(samples):
            i_file = str(sys.path[0]) + '\\dataset\\in_sample_{0}.txt'.format(sample)
            o_file = str(sys.path[0]) + '\\dataset\\out_sample_{0}.txt'.format(sample)

            matrix = parse.parse_from_file(i_file, m_size)
            res_mx = parse.parse_from_file(o_file, m_size)

            self.rows.append(np.array(matrix[net_index]))
            self.labels.append(res_mx[net_index].index(1))

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        row = self.rows[item]

        if self.transform is not None:
            row = self.transform(row)

        label = self.labels[item]

        return row, label


if __name__ == '__main__':

    datasetBunch = []
    netBunch = []
    optBunch = []

    # for nnet in range(m_size):
    #     net = Net()
    #     optimizer = opt.Adam(net.parameters(), lr=0.001)
    #
    #     netBunch.append(net)
    #     optBunch.append(optimizer)

    net = Net()
    optimizer = opt.Adam(net.parameters(), lr=0.001)

    print("------------- Network generated -------------")

    dst = MatrixDataset(0, 200000, newToTensor())
    print(dst[0])
    train_loader = DataLoader(
        dataset=dst,
        batch_size=10,
        drop_last=True,
        shuffle=True,  # want to shuffle the dataset
        num_workers=2)

    print("------------- Dataset ready -------------")

    loss = None
    for epoch in range(3):
        for data in train_loader:
            smpl, lbl = data
            # print(data)
            # break
            net.double()
            output = net(smpl.view(-1, 5))  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = f.nll_loss(output, lbl)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
            # break
        print(loss)
