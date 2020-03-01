#!/usr/bin/env python
"""
Description: generate baselines from TADPOLE_D${1_2, 3}.csv
Email: gzynus@gmail.com
Author: Zongyi Guo
"""
import os
import csv
import pickle
import datetime


def gen_baselines(data_path):
    """
    generate baseline_dates.pkl
    :param data_path:
    :return:
    """
    data_csv_path = os.path.join(data_path, 'data.csv')
    baseline_dates = {}
    data_dict = {}
    with open(data_csv_path) as data:
        data_csv = csv.reader(data)
        data_header = next(data_csv)
        for row in data_csv:
            if int(row[0]) in data_dict.keys():
                data_dict[int(row[0])].append(row)
            else:
                data_dict[int(row[0])] = []
                data_dict[int(row[0])].append(row)

    for sub in sorted(data_dict.keys()):
        baseline_date = data_dict[sub][0][1]
        baseline_date = datetime.datetime.strptime(baseline_date,
                                                   "%Y-%m-%d")
        baseline_dates[sub] = baseline_date

    # save baseline_dates as pkl
    baseline_dates_path = os.path.join(data_path, 'zongyi_baseline_dates.pkl')
    file = open(baseline_dates_path, 'wb')
    pickle.dump(baseline_dates, file)
    file.close()
    print('Finish generating baseline_dates!')

if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    gen_baselines(data_path)