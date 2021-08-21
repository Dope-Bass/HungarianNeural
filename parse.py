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
    for col in range(len(mx[0])):
        columns.append('Executor #{}'.format(col))
    columns.append('Executor for Assignment')
    for row in range(len(mx)):
        mx[row].insert(0, 'Assignment #{}'.format(row))

    print(columns)
    print(mx[0])