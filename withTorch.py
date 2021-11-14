import torch

import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt

import sys
from HungarianNeural import parse

size = 500
m_size = 500*500


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fullyConnectedLayer1 = nn.Linear(500*500, 20000)
        self.fullyConnectedLayer2 = nn.Linear(20000, 20000)
        self.fullyConnectedLayer3 = nn.Linear(20000, 20000)
        self.fullyConnectedLayer4 = nn.Linear(20000, 500*500)

    def forward(self, x):
        x = torch.sigmoid(self.fullyConnectedLayer1(x))
        x = torch.sigmoid(self.fullyConnectedLayer2(x))
        x = torch.sigmoid(self.fullyConnectedLayer3(x))
        x = self.fullyConnectedLayer4(x)
        return f.log_softmax(x, dim=1)


if __name__ == '__main__':

    wholeDataSet = []
    net = Net()
    loss_function = nn.BCELoss()
    optimizer = opt.Adam(net.parameters(), lr=0.001)

    print("------------- Network generated -------------")

    for sample in range(1000):
        preTensorFt = []
        preTensorLbl = []

        # for mx in range(10):
        file = str(sys.path[0]) + '\\dataset\\in_sample_{0}.txt'.format(sample)
        o_file = str(sys.path[0]) + '\\dataset\\out_sample_{0}.txt'.format(sample)

        matrix = parse.parse_from_file(file, size)
        res_mx = parse.parse_from_file(o_file, size)

        preTensorFt.append(matrix)
        preTensorLbl.append(res_mx)

        if (sample % 5 == 0) and (sample != 0):
            tensorFt = torch.FloatTensor(preTensorFt)
            tensorLbl = torch.FloatTensor(preTensorLbl)

            # print(tensorFt)
            # print(tensorLbl)

            oneBatch = [tensorFt, tensorLbl]

            # print(oneBatch[0][0][0])
            wholeDataSet.append(oneBatch)
            # break

    print(len(wholeDataSet))
    print("------------- Dataset ready -------------")

    loss = None
    for epoch in range(3):  # 3 full passes over the data
        print("------------- Epoch {0} -------------".format(epoch))
        for data in wholeDataSet:  # `data` is a batch of data
            X, y = data  # X is the batch of features, y is the batch of targets.
            # print(X)
            # print(y)
            net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = net(X.view(-1, m_size))  # pass in the reshaped batch (recall they are 28x28 atm)
            print("------------- Output ready -------------")
            loss = f.binary_cross_entropy_with_logits(output, y.view(-1, m_size))  # calc and grab the loss value
            print("------------- Loss calculated -------------")
            loss.backward()  # apply this loss backwards thru the network's parameters
            print("------------- Backward processed -------------")
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
            print("------------- One data element processed -------------")
        print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!
