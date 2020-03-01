from __future__ import print_function, division
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import evaluation.MAUC as MAUC


def str2date(string):
    return datetime.strptime(string, '%Y-%m-%d')


def nearest(series, target):
    return (series - target).abs().idxmin()


def mask(pred, true):
    index = ~np.isnan(true)
    ret = pred[index], true[index]
    assert ret[0].shape[0] == ret[0].shape[0]
    return ret

def parse_data(_ref_frame, _pred_frame):
    true_label_and_prob = []
    pred_diag = np.full(len(_ref_frame), -1, dtype=int)
    pred_adas = np.full(len(_ref_frame), -1, dtype=float)
    pred_vent = np.full(len(_ref_frame), -1, dtype=float)

    for i in range(len(_ref_frame)):
        # print(i)
        cur_row = _ref_frame.iloc[i]
        # print(cur_row)
        subj_data = _pred_frame[_pred_frame.RID == cur_row.RID].reset_index(drop=True)
        dates = subj_data['Forecast Date']
        # print('subj_data',subj_data)
        # print(dates)
        # print(cur_row.CognitiveAssessmentDate)

        matched_row = subj_data.iloc[nearest(dates, cur_row.CognitiveAssessmentDate)]
        # print('match_row',matched_row)
        prob = matched_row[['CN relative probability', 'MCI relative probability', 'AD relative probability']].values
        pred_diag[i] = np.argmax(prob)
        pred_adas[i] = matched_row['ADAS13']


        # for the mri scan find the forecast closest to the scan date,
        # which might be different from the cognitive assessment date
        pred_vent[i] = subj_data.iloc[nearest(dates, cur_row.ScanDate)]['Ventricles_ICV']

        if not np.isnan(cur_row.Diagnosis):
            true_label_and_prob += [(cur_row.Diagnosis, prob)]

    pred_diag, true_diag = mask(pred_diag, _ref_frame.Diagnosis)
    pred_adas, true_adas = mask(pred_adas, _ref_frame.ADAS13)
    pred_vent, true_vent = mask(pred_vent, _ref_frame.Ventricles)
    # print('std adas',np.std(true_adas))
    # print('std vent',np.std(true_vent))
    return true_label_and_prob, pred_diag, pred_adas, pred_vent, true_diag, true_adas, true_vent

def calcBCA(estimLabels, trueLabels, nrClasses):
    bcaAll = []

    for c0 in range(nrClasses):
        # c0 can be either CTL, MCI or AD

        # one example when c0=CTL
        # TP - label was estimated as CTL, and the true label was also CTL
        # FP - label was estimated as CTL, but the true label was not CTL (was either MCI or AD).
        TP = np.sum((estimLabels == c0) & (trueLabels == c0))
        TN = np.sum((estimLabels != c0) & (trueLabels != c0))
        FP = np.sum((estimLabels == c0) & (trueLabels != c0))
        FN = np.sum((estimLabels != c0) & (trueLabels == c0))

        # sometimes the sensitivity of specificity can be NaN, if the user doesn't forecast one of the classes.
        # In this case we assume a default value for sensitivity/specificity
        if (TP + FN) == 0:
            sensitivity = 0.5
        else:
            sensitivity = (1. * TP) / (TP + FN)

        if (TN + FP) == 0:
            specificity = 0.5
        else:
            specificity = (1. * TN) / (TN + FP)

        bcaCurr = 0.5 * (sensitivity + specificity)
        bcaAll += [bcaCurr]

    return np.mean(bcaAll)



def eval_submission(ref_frame, pred_frame):
    diagLabels = {'CN': 0, 'MCI': 1, 'AD': 2}
    ref_frame.replace({'Diagnosis': diagLabels}, inplace=True)
    # print('ref_frame:',ref_frame)
    # print('pred_frame:', pred_frame)

    true_labels_and_prob, p_diag, p_adas, p_vent, t_diag, t_adas, t_vent = parse_data(ref_frame, pred_frame)

    mauc = MAUC.MAUC(true_labels_and_prob, num_classes=len(diagLabels))
    # print('p_diag.shape',p_diag.shape)
    # print('t_diag:',t_diag.astype(int))
    bca = calcBCA(p_diag, t_diag.astype(int), nrClasses=len(diagLabels))
    # get true mae
    adas_mae = np.mean(np.abs(p_adas - t_adas))
    # print(len(p_vent))
    vent_mae = np.mean(np.abs(p_vent - t_vent))
    # get z-normalized mae
    z_adas_mae = np.mean(np.abs(p_adas - t_adas)) / 14.16166476358704
    z_vent_mae = np.mean(np.abs(p_vent - t_vent)) / 0.012544718874075404
    return mauc, bca, z_adas_mae, adas_mae, z_vent_mae, vent_mae




