import sys
import os
import pandas as pd


def parse_from_file(file, row_size):

    with open(file, 'rt') as file:
        text = file.readline().split(',')

    numbers = []
    rows = []
    for element in text:
        if len(numbers) != row_size:
            if element.isalnum():
                numbers.append(int(element))
        else:
            rows.append(numbers)
            numbers = []

    return rows


def matrix_to_dataframe(mx, res_mx):
    columns = ['Assignments']
    df_dict = {}
    for col in range(len(mx[0])):
        columns.append('Executor #{}'.format(col))
    columns.append('Executor for Assignment')
    executors = 0
    for row in range(len(res_mx)):
        mx[row].insert(0, 'Assignment #{}'.format(row))
        for res in range(len(res_mx[row])):
            if res_mx[row][res] == 1:
                executors += 1
                mx[row].append(columns[res])
                break
    print(executors)
    return pd.DataFrame(mx, columns=columns)
