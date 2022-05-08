import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
from HungarianNeural import parse

size = 25
m_size = 500*500
device = torch.device("cpu")


class newToTensor(object):
    def __call__(self, data):
        np_data = np.array(data)

        return torch.from_numpy(np_data).to(device)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fullyConnectedLayer1 = nn.Linear(5*5, 20)
        self.fullyConnectedLayer2 = nn.Linear(20, 20)
        self.fullyConnectedLayer3 = nn.Linear(20, 20)
        self.fullyConnectedLayer4 = nn.Linear(20, 120)

    def forward(self, x):
        x = torch.sigmoid(self.fullyConnectedLayer1(x))
        x = torch.sigmoid(self.fullyConnectedLayer2(x))
        x = torch.sigmoid(self.fullyConnectedLayer3(x))
        x = self.fullyConnectedLayer4(x)
        return f.log_softmax(x, dim=1)


class MatrixDataset(Dataset):

    def __init__(self, samples, transform=None):
        super().__init__()
        self.mxs = []
        self.labels = []
        for sample in range(samples):
            i_file = str(sys.path[0]) + '\\dataset\\in_sample_{0}.txt'.format(sample)
            o_file = str(sys.path[0]) + '\\dataset\\out_sample_{0}.txt'.format(sample)

            matrix = parse.parse_from_file(i_file, m_size)
            res_mx = parse.parse_from_file(o_file, m_size)

            # self.rows.append(np.array(matrix[net_index]))
            # self.labels.append(res_mx[net_index].index(1))

            self.labels.append(self.make_label(res_mx))
            # print(self.make_label(res_mx))
            self.mxs.append(matrix)

        self.transform = transform

        print(self.mxs)
        print(self.labels)

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def make_label(res_matrix):
        label = ''
        for row in res_matrix:
            label += str(row.index(1))

        return label

    def __getitem__(self, item):
        mx = self.mxs[item]

        if self.transform is not None:
            mx = self.transform(mx)

        label = self.labels[item]

        return mx, label


if __name__ == '__main__':

    net = Net()
    loss_function = nn.BCELoss()
    optimizer = opt.Adam(net.parameters(), lr=0.001)

    print("------------- Network generated -------------")

    dst = MatrixDataset(100000, newToTensor())
    train_loader = DataLoader(
        dataset=dst,
        batch_size=10,
        drop_last=True,
        shuffle=True,  # want to shuffle the dataset
        num_workers=2)

    print("------------- Dataset ready -------------")

    loss = None
    for epoch in range(3):  # 3 full passes over the data
        print("------------- Epoch {0} -------------".format(epoch))
        for data in train_loader:  # `data` is a batch of data
            X, y = data  # X is the batch of features, y is the batch of targets.
            # print(X)
            # print(y)
            net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = net(X.view(-1, 25))  # pass in the reshaped batch (recall they are 28x28 atm)
            print("------------- Output ready -------------")
            loss = f.binary_cross_entropy_with_logits(output, y.view(-1, m_size))  # calc and grab the loss value
            print("------------- Loss calculated -------------")
            loss.backward()  # apply this loss backwards thru the network's parameters
            print("------------- Backward processed -------------")
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
            print("------------- One data element processed -------------")
        print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!
