#!/usr/bin/env python
"""
Description: This is for expanding frog's tadpole basic csv file
Email:gzynus@gmail.com
Author: Zongyi Guo
"""
import os
import pandas as pd
import numpy as np


def create_basic_csv(basic_csv_path, folds_path, save_path):
    """
    Create basic csv files as Frog's way
    We need to do,
    1. pick train subjects' timepoints, val and test subject's first half
    timepoints
    2. Change D2 value as in Frog
    3. creay
    3. Create for each fold
    Note that we are creating 10 folds cross validation
    """
    # load basic csv file
    basic_df = pd.read_csv(basic_csv_path)
    # load folds.npy
    folds = np.load(folds_path)
    for test_fold in range(len(folds)):
        # create 10 fold cross validation
        print('Processing with test fold', test_fold)
        val_fold = (test_fold + 1) % len(folds)
        val_subjs = folds[val_fold]
        test_subjs = folds[test_fold]
        train_folds = [
            i for i in range(len(folds)) if (i != test_fold and i != val_fold)
        ]
        train_basic_csv_path = os.path.join(save_path, str(test_fold),
                                            'fold' + str(test_fold) + '_train_basic.csv')
        val_truth_path = os.path.join(save_path, str(test_fold),
                                      'fold' + str(test_fold) + '_val.csv')
        test_truth_path = os.path.join(save_path, str(test_fold),
                                      'fold' + str(test_fold) + '_test.csv')
        # now we need to create train_basic.csv
        _basic_df = basic_df.copy(deep=True)  # copy for modification
        _basic_df['D1'] = 0.
        _basic_df['D2'] = 0.
        # here we use D1, D2 to denote:
        # D1 = 1: this subject is in val fold
        # D2 = 1: this subject is in test fold
        # modify D1 and D2 value
        subjs = np.unique(_basic_df.RID)
        for subj in subjs:
            if subj in val_subjs:
                # we need to modify D1's value to 1
                _basic_df.loc[_basic_df[_basic_df.RID == subj].index, 'D1'] = 1
            else:
                # we need to modify D1's value to 0
                _basic_df.loc[_basic_df[_basic_df.RID == subj].index, 'D1'] = 0

            if subj in test_subjs:
                # we need to modify D2's value to 1
                _basic_df.loc[_basic_df[_basic_df.RID == subj].index, 'D2'] = 1
            else:
                # we need to modify D2's value to 0
                _basic_df.loc[_basic_df[_basic_df.RID == subj].index, 'D2'] = 0
        # remove test subjects and val subjects
        for subj in subjs:
            if subj in test_subjs or subj in val_subjs:
                # get ref date
                subj_mask = (_basic_df.RID == subj)
                ref_date = np.sort(_basic_df.EXAMDATE[subj_mask])[subj_mask.sum() // 2]
                mask = (_basic_df.RID == subj) & (_basic_df.EXAMDATE >= ref_date)
                _basic_df.drop(_basic_df[mask].index, inplace=True)
        _basic_df = _basic_df.reset_index(drop=True)
        # save train_train_basic.csv
        _basic_df.to_csv(train_basic_csv_path, index=False, sep=',')
        # val pred truth
        val_predgt_df = basic_df.copy(deep=True)
        # val_predgt_df['has_data'] = ~val_predgt_df[['DXCHANGE', 'ADAS13',
        #                                             'Ventricles', 'ICV']].isnull().apply(np.all, axis=1)
        # val_predgt_df = val_predgt_df[val_predgt_df.has_data]
        for subj in subjs:
            subj_mask = (val_predgt_df.RID == subj)
            if subj in val_subjs:
                # only remove first half half
                ref_date = np.sort(val_predgt_df.EXAMDATE[subj_mask])[
                    subj_mask.sum() // 2]
                mask = (val_predgt_df.RID == subj) & (
                            val_predgt_df.EXAMDATE < ref_date)
                val_predgt_df.drop(val_predgt_df[mask].index, inplace=True)
            else:
                val_predgt_df.drop(val_predgt_df[subj_mask].index, inplace=True)
        val_predgt_df = val_predgt_df.reset_index(drop=True)
        create_pred_truth(val_predgt_df, val_truth_path)
        # test pred truth
        test_predgt_df = basic_df.copy(deep=True)
        # test_predgt_df['has_data'] = ~test_predgt_df[['DXCHANGE', 'ADAS13',
        #                                              'Ventricles', 'ICV']].isnull().apply(
        #     np.all, axis=1)
        # test_predgt_df = test_predgt_df[test_predgt_df.has_data]
        for subj in subjs:
            subj_mask = (test_predgt_df.RID == subj)
            if subj in test_subjs:
                # only remove first half
                ref_date = np.sort(test_predgt_df.EXAMDATE[subj_mask])[
                    subj_mask.sum() // 2]
                mask = (test_predgt_df.RID == subj) & (
                        test_predgt_df.EXAMDATE < ref_date)
                test_predgt_df.drop(test_predgt_df[mask].index, inplace=True)
            else:
                test_predgt_df.drop(test_predgt_df[subj_mask].index,
                                    inplace=True)
        test_predgt_df = test_predgt_df.reset_index(drop=True)
        create_pred_truth(test_predgt_df, test_truth_path)


def create_pred_truth(df, save_path):
    # create pred truth csv file for val fold or test fold
    columns = [
        'RID', 'CognitiveAssessmentDate', 'Diagnosis', 'ADAS13', 'ScanDate'
    ]
    pred_gt = pd.DataFrame(columns=columns)
    # we need to remove some rows do not have Diagnosis, ADAS13 and Ventricels

    pred_gt[columns] = df[['RID', 'EXAMDATE', 'DXCHANGE', 'ADAS13', 'EXAMDATE']]
    pred_gt['Ventricles'] = df['Ventricles'] / df['ICV']
    # mapping
    mapping = {
        1: 'CN',
        7: 'CN',
        9: 'CN',
        2: 'MCI',
        4: 'MCI',
        8: 'MCI',
        3: 'AD',
        5: 'AD',
        6: 'AD'
    }
    pred_gt.replace({'Diagnosis': mapping}, inplace=True)
    pred_gt.reset_index(drop=True, inplace=True)

    pred_gt.to_csv(save_path, index=False, sep=',')


if __name__ == '__main__':
    # get root path
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    basic_csv_path = os.path.join(root_path, 'data',
                                  'frog_tadpole_basic_censored.csv')
    folds_path = os.path.join(root_path, 'data', 'folds.npy')

    save_path = os.path.join(root_path, 'data')

    create_basic_csv(basic_csv_path, folds_path, save_path)