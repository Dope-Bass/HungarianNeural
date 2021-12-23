import torch

import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
from HungarianNeural import parse

m_size = 500


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fullyConnectedLayer1 = nn.Linear(500, 100)
        self.fullyConnectedLayer2 = nn.Linear(100, 100)
        self.fullyConnectedLayer3 = nn.Linear(100, 100)
        self.fullyConnectedLayer4 = nn.Linear(100, 500)

    def forward(self, x):
        x = torch.relu(self.fullyConnectedLayer1(x))
        x = torch.relu(self.fullyConnectedLayer2(x))
        x = torch.relu(self.fullyConnectedLayer3(x))
        x = self.fullyConnectedLayer4(x)
        return f.log_softmax(x, dim=1)


class MatrixDataset(Dataset):

    def __init__(self, rows, labels, transform=None, target_transform=None):
        super().__init__()
        self.m_rows = rows
        self.m_labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.m_labels)

    def __getitem__(self, item):
        row = self.m_rows[item]
        label = self.m_labels[item]
        if self.transform:
            row = torch.tensor(row, dtype=torch.float32, device='cuda')
        if self.target_transform:
            label = torch.tensor(label, dtype=torch.long, device='cuda')
        return row, label

    def append(self, rows, value):
        if rows:
            self.m_rows.append(value)
        else:
            self.m_labels.append(value)


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

    print("------------- Network generated -------------")

    # inp = torch.randn(1, 300, 1, 500)
    # out = netBunch[0](inp)
    # print(out)

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

    print("------------- Dataset ready -------------")

    loss = None
    loss_function = nn.CrossEntropyLoss()
    for net in range(len(netBunch)):
        neural = netBunch[net]
        optim = optBunch[net]
        neural.cuda()

        for epoch in range(3):  # 3 full passes over the data
            print("------------- Epoch {0} -------------".format(epoch))

            preTensorFt = []
            preTensorLbl = []
            batch = []
            dataset = MatrixDataset([], [], transform=transforms.Compose([transforms.ToTensor()]),
                                    target_transform=transforms.Compose([transforms.ToTensor()]))

            for sample in range(80000):
                i_file = str(sys.path[0]) + '\\dataset\\in_sample_{0}.txt'.format(sample)
                o_file = str(sys.path[0]) + '\\dataset\\out_sample_{0}.txt'.format(sample)

                matrix = parse.parse_from_file(i_file, m_size)
                res_mx = parse.parse_from_file(o_file, m_size)

                # preTensorFt.append(matrix[net])
                # preTensorLbl.append(res_mx[net].index(1))

                dataset.append(True, matrix[net])
                dataset.append(False, res_mx[net].index(1))

                if (sample % 1000 == 999) and (sample != 0):
                    # tensorFt = []
                    # tensorLbl = []
                    # for t in range(len(preTensorFt)):
                    # tensorFt = torch.tensor(preTensorFt, dtype=torch.float32, device='cuda')
                    # tensorLbl = torch.tensor(preTensorLbl, dtype=torch.long, device='cuda')
                    #
                    # batch.append([tensorFt, tensorLbl])

                    # print(preTensorLbl)
                    # print(preTensorFt)

                    # print(tensorFt[0])
                    # print(tensorLbl[0])

                    # preTensorFt = []
                    # preTensorLbl = []
                    # print(batch)

                    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

                    for data in train_dataloader:
                        # for element in range(64):
                        # print(pair[0][element])
                        # print(pair[1][element])

                        X, y = data

                        # print(pair)

                        # print(X)
                        # print(y)

                        neural.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                        output = neural(X)  # pass in the reshaped batch (recall they are 28x28 atm)
                        # print("------------- Output ready -------------")
                        loss = loss_function(output, y)  # calc and grab the loss value
                        # print("------------- Loss calculated -------------")
                        loss.backward()  # apply this loss backwards thru the network's parameters
                        # print("------------- Backward processed -------------")
                        optim.step()  # attempt to optimize weights to account for loss/gradients
                        # print("------------- One data element processed -------------")
                    batch = []
                    dataset = MatrixDataset([], [], transform=transforms.Compose([transforms.ToTensor()]),
                                            target_transform=transforms.Compose([transforms.ToTensor()]))
                    print('Loss after batch calc = {0}'.format(loss))
            print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!
        break
