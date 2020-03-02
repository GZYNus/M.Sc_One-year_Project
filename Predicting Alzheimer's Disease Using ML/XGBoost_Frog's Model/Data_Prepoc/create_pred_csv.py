#!/usr/bin/env python
"""
Description: Create Prediction csv file, corresponding to tadpole_d4.RData
Email:gzynus@gmail.com
Author: Zongyi Guo
"""
import os
import time
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool


def month_between(end, start):
    """ Get duration (in months) between *end* and *start* dates """
    # assert end >= start
    end = datetime.strptime(end, "%Y-%m-%d")
    start = datetime.strptime(start, "%Y-%m-%d")
    diff = relativedelta(end, start)
    months = 12 * diff.years + diff.months
    to_next = relativedelta(end + relativedelta(months=1, days=-diff.days),
                            end).days
    to_prev = diff.days
    return months + (to_next < to_prev)


def create_long_features(df_array, dates, jj, kk):
    # create
    past_mask = (dates <= dates[jj])
    past = df_array[past_mask[: jj + 1]]
    past_dates = dates[: jj + 1]
    # get non nan
    if np.sum(~np.isnan(past)) > 0:
        non_nan_past = past[~np.isnan(past)]
        non_nan_dates = past_dates[~np.isnan(past)]
        most_recent = non_nan_past[-1]
        # time_since_most_recent = month_between(dates[kk], non_nan_dates[-1])
        time_since_most_recent = dates[kk] - non_nan_dates[-1]
        lowest = np.min(non_nan_past)
        # time_since_lowest = month_between(dates[kk], non_nan_dates[non_nan_past == lowest][0])
        time_since_lowest = dates[kk] - non_nan_dates[non_nan_past == lowest][0]
        highest = np.max(non_nan_past)
        # time_since_highest = month_between(dates[kk], non_nan_dates[non_nan_past == highest][0])
        time_since_highest = dates[kk] - non_nan_dates[non_nan_past == highest][0]
        if np.sum(~np.isnan(past)) > 1:
            mr_change = non_nan_past[-1] - non_nan_past[-2]
            # time_diff = month_between(dates[kk], non_nan_dates[-2])
            time_diff = dates[kk] - non_nan_dates[-2]
            mr_change /= time_diff
        else:
            mr_change = np.nan

        return [most_recent, time_since_most_recent, mr_change, lowest,
                time_since_lowest, highest, time_since_highest]

    else:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]


def create_dx_features(df_array, dates, jj, kk):
    # create a expanded dx features
    expanded = create_long_features(df_array, dates, jj, kk)
    # we need to remove mr_change and add milder, time_since_milder
    expanded.pop(2)  # remove mr_change
    past_mask = dates <= dates[jj]
    past = df_array[past_mask[: jj + 1]]
    past_dates = dates[: jj + 1]
    non_nan_past = past[~np.isnan(past)]
    non_nan_dates = past_dates[~np.isnan(past)]

    most_recent = expanded[0]
    tmp = non_nan_past[non_nan_past < most_recent]
    tmp_dates = non_nan_dates[non_nan_past < most_recent]
    if len(tmp) == 0:
        expanded.append(0)
        expanded.append(999)
    else:
        expanded.append(1)
        # expanded.append(month_between(dates[kk], tmp_dates[-1]))
        expanded.append(dates[kk] - tmp_dates[-1])

    return expanded


def feature_names(feature):
    """
    generate a list of expanded feature names
    :param feature:
    :return:
    """
    if feature == 'DX':
        return ['mr_dx', 'time_since_mr_dx', 'best_dx', 'time_since_best_dx',
                'worst_dx', 'time_since_worst_dx', 'milder',
                'time_since_midler']
    else:
        return ['mr_' + feature, 'time_since_mr_' + feature,
                'mr_change_' + feature, 'low_' + feature, 'time_since_low_' +
                feature, 'high_' + feature, 'time_since_high_' + feature]


def create_pred_csv(data_path, test_fold):
    """
    create prediction set, in corresponding with tadpole_d4.Data
    1.we need to get prediction start date
    2. For each subject, making prediction for consecutive 100 months
    3. create same input array as we did for train_long.csv
    """
    # we need to load pred start date
    pred_start_date_path = os.path.join(data_path, 'pred_start_date.pkl')
    with open(pred_start_date_path, 'rb') as file:
        pred_start_date = pickle.load(file, encoding='iso-8859-1')
    # we need to read folds.npy to get test subjects and val subjects
    folds = np.load(os.path.join(data_path, 'folds.npy'))
    # get test subjects and val subjects
    val_fold = (test_fold + 1) % len(folds)
    val_subjs = folds[val_fold]
    test_subjs = folds[test_fold]
    # read {test_fold}_train_basic.csv
    basic_df_path = os.path.join(data_path, str(test_fold),
                                  'fold' + str(test_fold) + '_train_basic.csv')
    basic_df = pd.read_csv( basic_df_path)
    subjs = np.unique(basic_df.RID)
    # create a train_long dataframe
    column_names = ['RID', 'Forecast_month', 'APOE', 'GENDER', 'EDUC',
                    'MARRY', 'am_positive', 'curr_age', 'Month_bl']
    # adding longitudinal features name
    column_names += feature_names('DX')
    column_names += feature_names('ADAS13')
    column_names += feature_names('Ventricles')
    column_names += feature_names('Fusiform')
    column_names += feature_names('WholeBrain')
    column_names += feature_names('Hippocampus')
    column_names += feature_names('MidTemp')
    column_names += feature_names('ICV')
    column_names += feature_names('MMSE')
    column_names += feature_names('CDRSB')
    column_names += feature_names('FDG')
    column_names += feature_names('ADAS11')
    column_names += feature_names('RAVLT_im')
    column_names += feature_names('RAVLT_learn')
    column_names += feature_names('RAVLT_forget')
    val_pred = pd.DataFrame(columns=column_names)
    test_pred = pd.DataFrame(columns=column_names)
    count = 0
    for subj in subjs:
        if subj in val_subjs:
            count += 1
            print('val subject', count, val_subjs.shape[0], 'for test fold',
                  test_fold)
            # we need to get month_bl
            subj_mask = (basic_df.RID == subj)
            subj_data = basic_df[subj_mask].sort_values('EXAMDATE', ascending=True)
            dates = np.sort(basic_df.EXAMDATE[subj_mask])
            pred_month_bl = month_between(pred_start_date[subj], dates[0])
            months = np.sort(basic_df.Month_bl[subj_mask])
            jj = months.shape[0] - 1
            init_age = np.sort(basic_df.curr_age[subj_mask])[0]
            gender = np.array(basic_df.PTGENDER[subj_mask])[0]
            apoe = np.array(basic_df.APOE4[subj_mask])[0]
            educ = np.array(basic_df.PTEDUCAT[subj_mask])[0]
            marrage = np.array(basic_df.PTMARRY[subj_mask])[0]
            am_positive = np.array(basic_df.am_positive[subj_mask])[0]
            for pred_month in range(pred_month_bl, pred_month_bl + 100):
                months = np.append(months, pred_month)
                kk = months.shape[0] - 1
                # making prediction for 100 months
                row = []
                row.append(subj)  # RID
                row.append(pred_month - pred_month_bl)  # Forcast_month
                row.append(apoe)
                row.append(gender)
                row.append(educ)
                row.append(marrage)
                row.append(am_positive)
                row.append(init_age + pred_month / 12.0)
                row.append(pred_month)
                row += create_dx_features(np.array(subj_data['curr_dx_numeric']), months, jj, kk)
                # append longitudinal features, mapped into 7 categories
                row += create_long_features(np.array(subj_data['ADAS13']),months, jj, kk)
                row += create_long_features(np.array(subj_data['Ventricles']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['Fusiform']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['WholeBrain']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['Hippocampus']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['MidTemp']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['MMSE']), months, jj, kk)
                row += create_long_features(np.array(subj_data['CDRSB']), months, jj, kk)
                row += create_long_features(np.array(subj_data['FDG']), months, jj, kk)
                row += create_long_features(np.array(subj_data['ADAS11']), months, jj, kk)
                row += create_long_features(np.array(subj_data['RAVLT_immediate']), months, jj, kk)
                row += create_long_features(np.array(subj_data['RAVLT_learning']), months, jj, kk)
                row += create_long_features(np.array(subj_data['RAVLT_perc_forgetting']), months, jj, kk)
                val_pred = val_pred.append(pd.DataFrame([row], columns=column_names), ignore_index=True)
    # save val pred
    val_pred.to_csv(os.path.join(data_path, str(test_fold), 'fold' + str(
        test_fold) + '_val_pred.csv'),
                    sep=',', index=False)
    count = 0
    for subj in subjs:
        if subj in test_subjs:
            count += 1
            print('test subject', count, test_subjs.shape[0], 'for test fold',
                  test_fold)
            # we need to get month_bl
            subj_mask = (basic_df.RID == subj)
            subj_data = basic_df[subj_mask]
            dates = np.sort(basic_df.EXAMDATE[subj_mask])
            pred_month_bl = month_between(pred_start_date[subj], dates[0])
            months = np.sort(basic_df.Month_bl[subj_mask])
            jj = months.shape[0] - 1
            init_age = np.sort(basic_df.curr_age[subj_mask])[0]
            gender = np.array(basic_df.PTGENDER[subj_mask])[0]
            apoe = np.array(basic_df.APOE4[subj_mask])[0]
            educ = np.array(basic_df.PTEDUCAT[subj_mask])[0]
            marrage = np.array(basic_df.PTMARRY[subj_mask])[0]
            am_positive = np.array(basic_df.am_positive[subj_mask])[0]
            for pred_month in range(pred_month_bl, pred_month_bl + 100):
                months = np.append(months, pred_month)
                kk = months.shape[0] - 1
                # making prediction for 100 months
                row = []
                row.append(subj)  # RID
                row.append(pred_month - pred_month_bl)  # Forcast_month
                row.append(apoe)
                row.append(gender)
                row.append(educ)
                row.append(marrage)
                row.append(am_positive)
                row.append(init_age + pred_month / 12.0)
                row.append(pred_month)
                row += create_dx_features(np.array(subj_data['curr_dx_numeric']), months, jj, kk)
                # append longitudinal features, mapped into 7 categories
                row += create_long_features(np.array(subj_data['ADAS13']),months, jj, kk)
                row += create_long_features(np.array(subj_data['Ventricles']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['Fusiform']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['WholeBrain']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['Hippocampus']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['MidTemp']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['MMSE']), months, jj, kk)
                row += create_long_features(np.array(subj_data['CDRSB']), months, jj, kk)
                row += create_long_features(np.array(subj_data['FDG']), months, jj, kk)
                row += create_long_features(np.array(subj_data['ADAS11']), months, jj, kk)
                row += create_long_features(np.array(subj_data['RAVLT_immediate']), months, jj, kk)
                row += create_long_features(np.array(subj_data['RAVLT_learning']), months, jj, kk)
                row += create_long_features(np.array(subj_data['RAVLT_perc_forgetting']), months, jj, kk)
                test_pred = test_pred.append(pd.DataFrame([row], columns=column_names), ignore_index=True)
    # save test pred
    test_pred.to_csv(os.path.join(data_path, str(test_fold), 'fold'  + str(
        test_fold) +'_test_pred.csv'),
                     sep=',', index=False)


def get_pred_start_date(data_path, save_path):
    """
    get the prediction start date for each subject and save for looking up
    :param data_path:
    :param save_path:
    :return:
    """
    # read original frog_tadpole_basic_censored.csv
    original_df_path = os.path.join(data_path,
                                    'frog_tadpole_basic_censored.csv')
    orginal_df = pd.read_csv(original_df_path)
    subjs = np.unique(orginal_df.RID)
    # for each subject, get it's prediction start date
    pred_start_date_dict = {}
    for subj in subjs:
        pred_start_date_dict[subj] = {}
        subj_mask = (orginal_df.RID == subj)
        # get the prediction start date
        ref_date = np.sort(orginal_df.EXAMDATE[subj_mask])[subj_mask.sum() // 2]
        pred_start_date_dict[subj] = ref_date
    # save it to a pkl file
    file = open(save_path, 'wb')
    pickle.dump(pred_start_date_dict, file)
    file.close()


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    data_path_list = []
    test_fold_list = []
    for test_fold in range(10):
        data_path_list.append(data_path)
        test_fold_list.append(test_fold)
    cores = multiprocessing.cpu_count()
    pool = Pool(processes=cores)
    pool.map(create_pred_csv, data_path_list, test_fold_list)
