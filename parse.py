import sys
import os
import pandas as pd


def parse_from_file(file, row_size):

    with open(file, 'rt') as file:
        text = file.readline().split('.')
        text.pop(-1)
        rows = []
        for row in text:
            # print(len(row))
            int_row = []
            for el in row.split(','):
                if el.isalnum():
                    int_row.append(int(el))
            rows.append(int_row)

    return rows


def matrix_to_dataframe(mx, res_mx):
    columns = ['Assignments']
    for col in range(len(mx[0])):
        columns.append('Executor #{}'.format(col))
    columns.append('Executor for Assignment')
    for row in range(len(res_mx)):
        mx[row].insert(0, 'Assignment #{}'.format(row))
        for res in range(len(res_mx[row])):
            if res_mx[row][res] == 1:
                mx[row].append('Executor #{}'.format(res))
                break
    return pd.DataFrame(mx, columns=columns)
