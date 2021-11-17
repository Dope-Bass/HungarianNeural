import time
import random
from munkres import *
import sys
from threading import Thread

count = 5000


def make_matrix(rows, cols):
    mx = []

    for row in range(rows):
        mx.append([])
        for col in range(cols):
            mx[row].append(random.randint(1, 300))

    return mx


def write_elem(elem, last, file):
    if last:
        file.write(str(elem))
        file.write('.')
    else:
        file.write(str(elem))
        file.write(',')


def write_mx_to_file(mx, name):
    with open(name, 'wt') as file:
        for row in range(len(mx)):
            for col in range(len(mx[row])):
                write_elem(mx[row][col], True if col == len(mx[row])-1 else False, file)

    print('Just wrote down file ' + name)


def write_res_to_file(res_mx, mx, name):
    with open(name, 'wt') as file:
        for row in range(len(mx)):
            for col in range(len(mx[row])):
                write_elem('1' if (row, col) in res_mx else '0', True if col == len(mx[row])-1 else False, file)

    print('Just wrote down file ' + name)


def calc_time_hung(matrix, alg_func):
    import timeit

    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row += [sys.maxsize - col]
        cost_matrix += [cost_row]

    start = timeit.default_timer()

    res = alg_func(cost_matrix)

    stop = timeit.default_timer()
    print('Time for 500x500 matrix by Hungarian Algorithm: ', stop - start)

    return res


if __name__ == '__main__':

    for _ in range(60057, 70000):
        mx_ = make_matrix(500, 500)
        result = calc_time_hung(mx_, Munkres().compute)
        write_mx_to_file(mx_, str(sys.path[0] + '\\dataset\\in_sample_{}.txt'.format(_)))
        write_res_to_file(result, mx_, str(sys.path[0] + '\\dataset\\out_sample_{}.txt'.format(_)))
