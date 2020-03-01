#!/usr/bin/env python
"""
Description: self-defined functions
Email: gzynus@gmail.com
Author: Zongyi Guo
"""
import pandas as pd
import datetime


def list2csv(list, csv_path, columns_name):
    """
    functions for writing list to a csv file
    :param list:
    :param csv_path:
    :param columns_name:
    :return:
    """
    df = pd.DataFrame(list, columns=columns_name)
    df.to_csv(csv_path, index=False, sep=',')


def sort_dates(list):
    """
    sort the records in list according to the dates
    :param list:
    :return:
    """
    return sorted(list, key=(lambda x: datetime.datetime.strptime(x[1],
                                                                  "%Y-%m-%d")))   # sort according to time


if __name__ == '__main__':
    list = [[3, '2003-01-09'], [1, '2001-02-08'], [2, '2002-09-16']]
    print(sort_dates(list))
