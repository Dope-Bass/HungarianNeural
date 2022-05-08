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
                    int_row.append(float(el))
            rows.append(int_row)

    return rows


def matrix_to_dataframe(mx, res_mx):
    # columns = ['Assignments']
    # columns = []
    # for col in range(len(mx[0])):
    #     columns.append('Executor#{}'.format(col))
    # columns.append('ExecutorForAssignment')
    # for row in range(len(res_mx)):
    #     mx[row].insert(0, 'Assignment#{}'.format(row))
    #     # mx[row].insert(0, row)
    #     for res in range(len(res_mx[row])):
    #         if res_mx[row][res] == 1:
    #             mx[row].append('Executor#{}'.format(res))
    #             break
    # return pd.DataFrame(mx, columns=columns)

    # data_dict = {'Assignments': []}
    data_dict = {}
    for col in range(len(mx[0])):
        data_dict.update({'Executor#{}'.format(col): []})
    data_dict.update({'ExecutorForAssignment': []})
    for row in range(len(res_mx)):
        # data_dict.update({'Assignments': 'Assignment#{}'.format(row)})
        # data_dict['Assignments'].append('Assignment#{}'.format(row))
        for res in range(len(res_mx[row])):
            if res_mx[row][res] == 1:
                # data_dict.update({'ExecutorForAssignment': 'Executor#{}'.format(res)})
                data_dict['ExecutorForAssignment'].append('Executor#{}'.format(res))
            data_dict['Executor#{}'.format(res)].append(mx[row][res])
    # print(data_dict)

    return pd.DataFrame.from_dict(data_dict)
