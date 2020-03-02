#!/usr/bin/env python
"""
Description: self-defined functions
Email: gzynus@gmail.com
Author: Zongyi

"""
import os
import pickle
import pandas as pd
import numpy as np
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_forcast_date(start_date, month):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    date = start_date + relativedelta(months=month)

    return datetime.strftime(date, "%Y-%m-%d")


def load_feature(feature_file_path):
    """
    Load list of features from a text file
    Features are separated by newline
    """
    return [l.strip() for l in open(feature_file_path)]


def load_train_npy(data_path, test_fold, label, window):
    """

    :param train_set_path:
    :return:
    """
    # get npy file path
    npy_name = 'fold' + str(test_fold) + '_train_' + label + '_' + str(
        window) + '.npy'
    npy_path = os.path.join(data_path, str(test_fold), npy_name)

    return np.load(npy_path)


def load_pred_npy(data_path, test_fold, label, window, isTest=False):
    # get npy file path
    if isTest:
        npy_name = 'fold' + str(test_fold) + '_test_' + label + '_' + str(
            window) + '.npy'
    else:
        npy_name = 'fold' + str(test_fold) + '_val_' + label + '_' + str(
            window) + '.npy'
    npy_path = os.path.join(data_path, str(test_fold), npy_name)

    return np.load(npy_path)


def build_pred_frame(id_array, pred_array, args, model_name):
    """
    save prediction into a csv file
    :param id_array:
    :param pred_array:
    :param args:
    :return:
    """
    # we need to load prediction start date for each subject
    pred_start_date_path = os.path.join(args.data_path, 'pred_start_date.pkl')
    with open(pred_start_date_path, 'rb') as fobj:
        pred_start_date = pickle.load(fobj)
    fobj.close()
    # read means and stds to recover prediction
    means_stds_path = os.path.join(args.data_path, str(args.test_fold),
                                   'fold' + str(args.test_fold) +
                                   '_means_stds.pkl')
    with open(means_stds_path, 'rb') as fobj:
        means_stds = pickle.load(fobj)
    fobj.close()
    # inverse z normalization
    if args.label in ['adas13', 'ventricles']:
        label = 'curr_' + args.label
        mean = means_stds[label]['mean']
        std = means_stds[label]['std']
        pred_array[:, 0] = (pred_array[:, 0] * std) + mean
    # create a prediction dataframe
    columns = ['RID', 'Forecast Month', 'Forecast Date',
               'CN relative probability', 'MCI relative probability',
               'AD relative probability', 'ADAS13', 'Ventricles_ICV']
    fill_array = np.full([id_array.shape[0], len(columns)], np.nan)
    fill_array[:, 0] = id_array[:, 0]
    fill_array[:, 1] = id_array[:, 1]
    forcast_date = []
    for i in range(id_array.shape[0]):
        start_date = pred_start_date[id_array[i, 0]]
        date = get_forcast_date(start_date, id_array[i, 1])
        forcast_date.append(date)
    if args.label == 'adas13':
        fill_array[:, 6] = pred_array[:, 0]
    elif args.label == 'ventricles':
        fill_array[:, 7] = pred_array[:, 0]
    elif args.label == 'dx':
        fill_array[:, 3:6] = pred_array[:, :3]
    pred_df = pd.DataFrame(data=fill_array, columns=columns)
    pred_df.iloc[:, 2] = np.array(forcast_date)
    # save it to a csv file
    save_path = os.path.join(args.pred_path, str(args.test_fold),
                             'val_' + model_name + '.csv')
    pred_df.to_csv(save_path, sep=',', index=False)


def get_model_name(args):
    # get the model name for certain args
    model_name = '{Label:' + args.label \
                 + '_Window:' + str(args.window) \
                 + '_LR:' + str(args.lr) \
                 + '_L2:' + str(args.l2) \
                 + '_I_drop:' + str(args.i_ratio) \
                 + '_H_drop:' + str(args.h_ratio) \
                 + '_Nb_layer:' + str(args.nb_layer) \
                 + '_H0:' + str(args.h0) \
                 + '_H1:' + str(args.h1) \
                 + '_H2:' + str(args.h2) \
                 + '_H3:' + str(args.h3) \
                 + '_H4:' + str(args.h4) \
                 + '_EPOCH:' + str(args.epochs)

    return model_name


def get_pred_ref_path(args):
    """
    get prediction and ref csv path
    :param args:
    :return:
    """
    # ref_path = os.path.join(args.data_path, str(args.test_fold),
    #                         'fold' + str(args.test_fold) + '_val_gt_'
    #                         + args.label + '_' + str(args.window) + '.csv')
    ref_path = os.path.join(args.data_path, str(args.test_fold),'fold3_test.csv')
    model_name = get_model_name(args)

    pred_path = os.path.join(args.pred_path, str(args.test_fold),
                             'val_' + model_name + '.csv')
    return ref_path, pred_path


def Diagnosis_conv(value):
    '''Convert diagnosis from string to float '''
    if value == 'CN':
        return 0.
    if value == 'MCI':
        return 1.
    if value == 'AD':
        return 2.
    return float('NaN')

def str2date(string):
    """ Convert string to datetime object """
    return datetime.strptime(string, '%Y-%m-%d')


CONVERTERS = {
    'CognitiveAssessmentDate': str2date,
    'ScanDate': str2date,
    'Forecast Date': str2date,
    'EXAMDATE': str2date,
    'Diagnosis': Diagnosis_conv
}


def read_csv(fpath):
    """ Load CSV with converters """
    return pd.read_csv(fpath, converters=CONVERTERS)
