#!/usr/bin/env python
"""
Description: preprocessing phase: extract 23 features from raw dataset
Email: gzynus@gmail.com
Author: Zongyi Guo
"""
import os
import csv
from lib.helper import list2csv
from lib.helper import sort_dates


def extract_features(raw_data_path, data_path):
    """
    extract features from raw csv file
    :param raw_data_path:
    :param data_path:
    :return:
    """
    D1_D2_path = os.path.join(raw_data_path, 'TADPOLE_D1_D2.csv')
    features_dict = {}
    with open(D1_D2_path) as D1_D2:
        D1_D2_csv = csv.reader(D1_D2)
        D1_D2_header = next(D1_D2_csv)
        for row in D1_D2_csv:
            if int(row[0]) in features_dict.keys():
                features_dict[int(row[0])].append(row)
            else:
                features_dict[int(row[0])] = []
                features_dict[int(row[0])].append(row)
    all_list = []
    for sub in sorted(features_dict.keys()):
        sub_list = []
        for row in features_dict[sub]:
            list = []
            RID = row[0]
            list.append(RID)
            ExamDate = row[8]
            list.append(ExamDate)
            DX = row[54]
            if DX == '':
                DX = 'Nan'
            if DX == 'NL':
                DX = 0
            if DX == 'NL to MCI':
                DX = 1
            if DX == 'MCI to NL':
                DX = 0
            if DX == 'MCI':
                DX = 1
            if DX == 'Dementia to MCI':
                DX = 1
            if DX == 'MCI to Dementia':
                DX = 2
            if DX == 'Dementia':
                DX = 2
            if DX == 'NL to Dementia':
                DX = 2
            list.append(DX)
            CDRSB = row[21]
            if CDRSB == '':
                CDRSB = 'Nan'
            list.append(CDRSB)
            ADAS11 = row[22]
            if ADAS11 == '':
                ADAS11 = 'Nan'
            list.append(ADAS11)
            ADAS13 = row[23]
            if ADAS13 == '':
                ADAS13 = 'Nan'
            list.append(ADAS13)
            MMSE = row[24]
            if MMSE == '':
                MMSE = 'Nan'
            list.append(MMSE)
            RAVLT_immediate = row[25]
            if RAVLT_immediate == '':
                RAVLT_immediate = 'Nan'
            list.append(RAVLT_immediate)
            RAVLT_learning = row[26]
            if RAVLT_learning == '':
                RAVLT_learning = 'Nan'
            list.append(RAVLT_learning)
            RAVLT_forgetting = row[27]
            if RAVLT_forgetting == '':
                RAVLT_forgetting = 'Nan'
            list.append(RAVLT_forgetting)
            RAVLT_perc_forgetting = row[28]
            if RAVLT_perc_forgetting == '':
                RAVLT_perc_forgetting = 'Nan'
            list.append(RAVLT_perc_forgetting)
            MOCA = row[30]
            if MOCA == '':
                MOCA = 'Nan'
            list.append(MOCA)
            FAQ = row[29]
            if FAQ == '':
                FAQ = 'Nan'
            list.append(FAQ)
            Entorhinal = row[50]
            if Entorhinal == '':
                Entorhinal = 'Nan'
            list.append(Entorhinal)
            Fusiform = row[51]
            if Fusiform == '':
                Fusiform = 'Nan'
            list.append(Fusiform)
            Hippocampus = row[48]
            if Hippocampus == '':
                Hippocampus = 'Nan'
            list.append(Hippocampus)
            ICV = row[53]
            if ICV == '':
                ICV = 'Nan'
            list.append(ICV)
            MidTemp = row[52]
            if MidTemp == '':
                MidTemp = 'Nan'
            list.append(MidTemp)
            Ventricles = row[47]
            if Ventricles == '':
                Ventricles = 'Nan'
            list.append(Ventricles)
            WholeBrain = row[49]
            if WholeBrain == '':
                WholeBrain = 'Nan'
            list.append(WholeBrain)
            AV45 = row[20]
            if AV45 == '':
                AV45 = 'Nan'
            list.append(AV45)
            FDG = row[18]
            if FDG == '':
                FDG = 'Nan'
            list.append(FDG)
            ABETA = row[1902]
            if ABETA == '' or ABETA == ' ':
                ABETA = 'Nan'
            if ABETA == '<200':
                ABETA = 200
            list.append(ABETA)
            TAU = row[1903]
            if TAU == '' or TAU == ' ':
                TAU = 'Nan'
            if TAU == '<80':
                TAU = 80
            if TAU == '>1300':
                TAU = 1300
            list.append(TAU)
            PTAU = row[1904]
            if PTAU == '' or PTAU == ' ':
                PTAU = 'Nan'
            if PTAU == '<8':
                PTAU = 8
            if PTAU == '>120':
                PTAU = 120
            list.append(PTAU)
            if list.count('Nan') != 23:
                sub_list.append(list)
        all_list += sort_dates(sub_list)
    raw_data_path = os.path.join(data_path, 'data.csv')
    columns_name = ['RID', 'Exam Date', 'DX', 'CDRSB', 'ADAS11',
                    'ADAS13', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning',
                    'RAVLT_forgetting', 'RAVLT_perc_forgetting',
                    'MOCA', 'FAQ', 'Entorhinal', 'Fusiform',
                    'Hippocampus', 'ICV', 'MidTemp', 'Ventricles',
                    'WholeBrain', 'AV45', 'FDG', 'A-BETA', 'TAU', 'PTAU']
    list2csv(all_list, raw_data_path, columns_name)
    print('Finish extracting all features!')


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    raw_data_path = os.path.join(root_path, 'raw_data')
    extract_features(raw_data_path, data_path)

