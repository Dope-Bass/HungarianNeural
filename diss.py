import cflearn
from cfdata.tabular import *
import sys
import pandas as pd
from HungarianNeural import parse


if __name__ == '__main__':

    file = str(sys.path[0]) + '\\dataset\\in_sample_1.txt'
    o_file = str(sys.path[0]) + '\\dataset\\out_sample_1.txt'

    matrix = parse.parse_from_file(file, 500)
    res_mx = parse.parse_from_file(o_file, 500)
    parse.matrix_to_dataframe(matrix, res_mx)



