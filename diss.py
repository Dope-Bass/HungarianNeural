import cflearn
from cfdata.tabular import *
import sys
import pandas as pd
from HungarianNeural import parse


if __name__ == '__main__':

    file = str(sys.path[0]) + '\\dataset\\in_sample_0.txt'
    o_file = str(sys.path[0]) + '\\dataset\\out_sample_0.txt'

    matrix = parse.parse_from_file(file, 499)
    res_mx = parse.parse_from_file(o_file, 499)
    for row in range(len(res_mx)):
        for res in range(int(len(res_mx[row]))):
            if res_mx[row][res]:
                print('row {0} col {1}'.format(row, res))
    dataframe = parse.matrix_to_dataframe(matrix, res_mx)
    print(dataframe['Executor for Assignment'])



