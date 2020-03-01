#!/usr/bin/env python
"""
Description: reference data for model's evaluation
Email: gzynus@gmail.com
Author: Zongyi Guo
"""

import os
import csv
import datetime
import numpy as np
from lib.helper import list2csv
from lib.helper import sort_dates
from lib.calc_date import compare_date
from lib.load_data import loading_subject_list


def gen_ref_data_old(data_path, raw_data_path, output_path, split_date, flag):
    """
    generate reference data for model evaluation
    :param data_path:
    :param output_path:
    :param split_date:
    :param flag:
    :return:
    """
    assert flag in ('test', 'dev')
    # loading data.csv
    data_csv_path = os.path.join(data_path, 'data.csv')
    # loading subjects list
    sub_list = loading_subject_list(raw_data_path, flag)
    # get accurate sub_list
    acc_sub_list = []
    data_dict = {}
    with open(data_csv_path) as data:
        data_csv = csv.reader(data)
        data_header = next(data_csv)
        for row in data_csv:
            if int(row[0]) in sub_list:
                if int(row[0]) in data_dict.keys():
                    data_dict[int(row[0])].append(row)
                else:
                    data_dict[int(row[0])] = []
                    data_dict[int(row[0])].append(row)
    for sub in data_dict.keys():
        before = False
        after = False
        for record in data_dict[sub]:
            date = datetime.datetime.strptime(record[1],
                                               "%Y-%m-%d")
            if compare_date(date, split_date):
                after = True
            else:
                before = True
        if before and after:
            acc_sub_list.append(sub)

    ref_data = []
    with open(data_csv_path) as data:
        data_csv = csv.reader(data)
        data_header = next(data_csv)
        for row in data_csv:
            if int(row[0]) in acc_sub_list:

                date = datetime.datetime.strptime(row[1],
                                                  "%Y-%m-%d")
                if compare_date(date, split_date):
                    clip_row = []
                    clip_row.append(row[0])
                    clip_row.append(row[1])
                    if row[2] == 'Nan':
                        clip_row.append('')
                    elif int(row[2]) == 0:
                        clip_row.append('CN')
                    elif int(row[2]) == 1:
                        clip_row.append('MCI')
                    else:
                        clip_row.append('AD')
                    clip_row.append(row[5])
                    clip_row.append(row[1])
                    if row[16] is not 'Nan' and row[18] is not 'Nan':
                        clip_row.append(float(row[18]) / float(row[16]))
                    else:
                        clip_row.append('Nan')
                    clip_row = ['' if x == 'Nan' else x for x in clip_row]
                    ref_data.append(clip_row)
    # save as a csv file
    ref_data_path = os.path.join(output_path, flag + '_ref_fix_date.csv')
    columns_name = ["RID", "CognitiveAssessmentDate", "Diagnosis",
                    "ADAS13", "ScanDate", "Ventricles"]
    list2csv(ref_data, ref_data_path, columns_name)
    print('Finish generating reference for' + flag)


def gen_ref_data(data_path, raw_data_path, out_path, flag):
    """
    generate $_ref.csv for model evaluation
    :param data_path:
    :param raw_data_path:
    :param out_path:
    :param flag:
    :return:
    """
    # load data.csv
    data_csv_path = os.path.join(data_path, 'data.csv')
    # load subject list
    sub_list = loading_subject_list(raw_data_path, flag)
    ref_list = []
    ref_data = {}
    for sub in sub_list:
        ref_data[sub] = []
    with open(data_csv_path) as data:
        data_csv = csv.reader(data)
        data_header = next(data_csv)
        for row in data_csv:
            if int(row[0]) in sub_list:
                clip_row = []
                clip_row.append(row[0])
                clip_row.append(row[1])
                # print(clip_row)
                if row[2] == 'Nan':
                    clip_row.append('')
                elif int(row[2]) == 0:
                    clip_row.append('CN')
                elif int(row[2]) == 1:
                    clip_row.append('MCI')
                else:
                    clip_row.append('AD')
                # print(clip_row)
                clip_row.append(row[5])
                clip_row.append(row[1])
                if row[16] is not 'Nan' and row[18] is not 'Nan':
                    clip_row.append(float(row[18]) / float(row[16]))
                else:
                    clip_row.append('Nan')
                # print(clip_row)
                clip_row = [np.nan if x == 'Nan' else x for x in clip_row]   # ? 为何clip_row那儿没有影响
                ref_data[int(row[0])].append(clip_row)
    # only extract the second half as reference
    for sub in sorted(sub_list):
        data_lists = sort_dates(ref_data[sub])
        numPoints = len(data_lists)
        if numPoints != 1:
            if numPoints % 2 == 0:
                start_point = numPoints / 2
            else:
                start_point = (numPoints + 1) / 2
            for tp in range(int(start_point), numPoints):
                ref_list.append(data_lists[tp])
    # save ref_list to csv file
    ref_list_path = os.path.join(out_path, flag + '_ref.csv')
    columns_name = ["RID", "CognitiveAssessmentDate", "Diagnosis",
                    "ADAS13", "ScanDate", "Ventricles"]
    list2csv(ref_list, ref_list_path, columns_name)
    # print(ref_list_path)
    print('Finish generating ' + flag + '_dev.csv!')



if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    raw_data_path = os.path.join(root_path, 'raw_data')
    out_path = os.path.join(root_path, 'evaluation')
    gen_ref_data(data_path, raw_data_path, out_path, 'dev')
    gen_ref_data(data_path, raw_data_path, out_path, 'test')
    # gen_ref_data_old(data_path, raw_data_path, out_path, '2010-05', 'dev')
    # gen_ref_data_old(data_path, raw_data_path, out_path, '2010-05', 'test')
