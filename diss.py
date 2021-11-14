import sys
from HungarianNeural import parse
from fastai.tabular.all import *
from fastai.tabular.data import *
import numpy as np

import torch
from torch.nn import functional

from fastai.basics import Learner
from fastai.tabular.model import TabularModel
from fastai.tabular.data import TabularDataLoaders
from fastai.metrics import accuracy
from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import fit_one_cycle


# fastai 1.0.61
# fastai 2.5.2


class MyCrossEntropy(torch.nn.CrossEntropyLoss):

    def forward(self, input, target):
        target = torch.squeeze(target.long())  # ADDED
        return torch.nn.functional.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                                                 reduction=self.reduction)


if __name__ == '__main__':

    dfs = []
    for tab in range(1000):
        file = str(sys.path[0]) + '\\dataset\\in_sample_{0}.txt'.format(tab)
        o_file = str(sys.path[0]) + '\\dataset\\out_sample_{0}.txt'.format(tab)

        matrix = parse.parse_from_file(file, 500)
        res_mx = parse.parse_from_file(o_file, 500)
        for row in range(len(res_mx)):
            for res in range(len(res_mx[row])):
                if res_mx[row][res] == 1:
                    # print('row {0} col {1}'.format(row, res))
                    pass
        df = parse.matrix_to_dataframe(matrix, res_mx)
        # print(df.head())
        dfs.append(df)
    # df.transpose()
    dep_var = 'ExecutorForAssignment'
    # cat_names = [] + [df.columns.tolist()[-1]]
    # print(cat_names)
    # print(cont_names)

    # df.to_excel('output.xlsx')
    res_df = pd.concat(dfs, ignore_index=True)
    cont_names = [] + res_df.columns.tolist()[0:len(res_df.columns.tolist()) - 1]
    print(res_df)
    # to = TabularPandas(res_df, procs=[Categorify],
    #                    cont_names=cont_names, y_names=dep_var)
    # print(to.xs.iloc[:500000])
    # dls = to.dataloaders(bs=5)
    # learn = tabular_learner(dls)
    # learn.fit(n_epoch=1)
    # row, clas, probs = learn.predict(df.iloc[0])
    # print(row)
    # dls.show_batch()
    #
    # learn = tabular_learner(dls, metrics=accuracy)
    # learn.fit(4)
    # learn.show_results()

    # dl = TabularDataLoaders.from_df(res_df,
    #                                 cont_names=cont_names,
    #                                 y_names=dep_var,
    #                                 bs=5,
    #                                 )
    # res_df[dep_var] = res_df[dep_var].astype(np.int64).values
    # emb_s = [(4, 2), (17, 8)]
    # model = TabularModel(emb_szs=[], n_cont=len(cont_names), out_sz=2, layers=[512, 256])
    # learn = Learner(dl, model, loss_func=MyCrossEntropy(), metrics=[accuracy], cbs=ProgressCallback())
    # learning_rate = 1.32e-2
    # epochs = 5
    # fit_one_cycle(learn, n_epoch=epochs, lr_max=learning_rate)

    x = torch.Tensor([5, 3])
    y = torch.Tensor([2, 1])

    print