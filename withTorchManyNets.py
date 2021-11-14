import torch

import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt

import sys
from HungarianNeural import parse

m_size = 500


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fullyConnectedLayer1 = nn.Linear(500, 50)
        self.fullyConnectedLayer2 = nn.Linear(50, 50)
        self.fullyConnectedLayer3 = nn.Linear(50, 50)
        self.fullyConnectedLayer4 = nn.Linear(50, 500)

    def forward(self, x):
        x = torch.sigmoid(self.fullyConnectedLayer1(x))
        x = torch.sigmoid(self.fullyConnectedLayer2(x))
        x = torch.sigmoid(self.fullyConnectedLayer3(x))
        x = self.fullyConnectedLayer4(x)
        return f.log_softmax(x, dim=1)


def matrixToTensor(matrix, index, res):

    return [matrix[index], res[index].index(1)+1]


if __name__ == '__main__':

    datasetBunch = []
    netBunch = []
    optBunch = []

    preTensorFt = []
    preTensorLbl = []

    for nnet in range(m_size):
        net = Net()
        optimizer = opt.Adam(net.parameters(), lr=0.001)

        netBunch.append(net)
        optBunch.append(optimizer)
        datasetBunch.append([])
        preTensorFt.append([])
        preTensorLbl.append([])

    print("------------- Network generated -------------")

    '''
    datasetBunch = 
        [ 
            row1 = [
                batch1 = [ tensorFt(tensor) = [ trRow1_1 = [] ... trRow1_10 = [] ], tensorLbl(tensor) ]
                batch2 = [ tensorFt(tensor) = [ trRow1_11 = [] ... trRow1_20 = [] ], tensorLbl(tensor) ]
                ...
                batch100 = [ tensorFt(tensor) = [ trRow1_991 = [] ... trRow1_1000 = [] ], tensorLbl(tensor) ]
            ]
            row2 = [
                batch1 = [ tensorFt(tensor) = [ trRow2_1 = [] ... trRow2_10 = [] ], tensorLbl(tensor) ]
                batch2 = [ tensorFt(tensor) = [ trRow2_11 = [] ... trRow2_20 = [] ], tensorLbl(tensor) ]
                ...
                batch100 = [ tensorFt(tensor) = [ trRow2_991 = [] ... trRow2_1000 = [] ], tensorLbl(tensor) ]
            ]
            ...
            row500 = [
                batch1 = [ tensorFt(tensor) = [ trRow500_1 = [] ... trRow500_10 = [] ], tensorLbl(tensor) ]
                batch2 = [ tensorFt(tensor) = [ trRow500_11 = [] ... trRow500_20 = [] ], tensorLbl(tensor) ]
                ...
                batch100 = [ tensorFt(tensor) = [ trRow500_991 = [] ... trRow500_1000 = [] ], tensorLbl(tensor) ]
            ]  
        ]
    '''

    for sample in range(1000):

        file = str(sys.path[0]) + '\\dataset\\in_sample_{0}.txt'.format(sample)
        o_file = str(sys.path[0]) + '\\dataset\\out_sample_{0}.txt'.format(sample)

        matrix = parse.parse_from_file(file, m_size)
        res_mx = parse.parse_from_file(o_file, m_size)

        for row in range(m_size):
            preTensorFt[row].append(matrix[row])
            preTensorLbl[row].append(res_mx[row].index(1))

        if sample % 10 == 9:
            for row in range(m_size):
                tensorFt = torch.FloatTensor(preTensorFt[row])
                tensorLbl = torch.LongTensor(preTensorLbl[row])

                datasetBunch[row].append([tensorFt, tensorLbl])

            preTensorFt = []
            preTensorLbl = []

            for row in range(m_size):
                preTensorFt.append([])
                preTensorLbl.append([])

    print(datasetBunch)

    # preTensorFt.append(matrix)
    # preTensorLbl.append(res_mx)

    # if (sample % 10 == 9) and (sample != 0):
    #     tensorFt = torch.FloatTensor(preTensorFt)
    #     tensorLbl = torch.FloatTensor(preTensorLbl)

    # print(tensorFt)
    # print(tensorLbl)

    # oneBatch = [tensorFt, tensorLbl]

    # print(oneBatch[0][0][0])
    # wholeDataSet.append(oneBatch)
    # break

    # print(len(wholeDataSet))
    print("------------- Dataset ready -------------")

    loss = None
    for net in range(len(netBunch)):
        neural = netBunch[net]
        optim = optBunch[net]

        for epoch in range(3):  # 3 full passes over the data
            print("------------- Epoch {0} -------------".format(epoch))
            for data in datasetBunch[net]:  # `data` is a batch of data
                X, y = data  # X is the batch of features, y is the batch of targets.
                # print(X)
                # print(y)
                neural.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                output = neural(X.view(-1, m_size))  # pass in the reshaped batch (recall they are 28x28 atm)
                print("------------- Output ready -------------")
                loss = f.nll_loss(output, y)  # calc and grab the loss value
                print("------------- Loss calculated -------------")
                loss.backward()  # apply this loss backwards thru the network's parameters
                print("------------- Backward processed -------------")
                optim.step()  # attempt to optimize weights to account for loss/gradients
                print("------------- One data element processed -------------")
            print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!
        break
