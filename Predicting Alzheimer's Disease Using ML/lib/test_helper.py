# pylint: disable=too-many-arguments
"""
Description: self-defined functions
Email: gzynus@gmail.com
Author: Zongyi Guo
"""
import datetime
import pandas as pd
import numpy as np
from lib.load_data import loading_means_stds
from lib.calc_date import calc_date
from evaluation.evalOneSubmission import eval_submission
from evaluation.evalOneSubmission import str2date


def gen_pred_dict():
    """
    generate a prediction dict
    :return:
    """
    result = {}
    result['RID'] = []
    result['Forecast_Month'] = []
    result['Forecast_Date'] = []
    result['CN_relative_probability'] = []
    result['MCI_relative_probability'] = []
    result['AD_relative_probability'] = []
    result['ADAS13'] = []
    result['ADAS13_CI_lower'] = []
    result['ADAS13_CI_upper'] = []
    result['Ventricles_ICV'] = []
    result['Ventricles_ICV_CI_lower'] = []
    result['Ventricles_ICV_CI_upper'] = []

    return result


def save_pred(pred, diagnosis_prob, diagnosis_pred, continuous_pred):
    """
    save prediction to a tensor
    """
    pred[0, :3] = diagnosis_prob
    pred[0, 3] = diagnosis_pred
    pred[0, 4:] = continuous_pred

    return pred


def pred2list(pred, sub_id, month, forecast_month, baseline, data_path, flag):
    """
    write prediction to list, each list is a row in csv file
    """
    # define list
    list = []
    # load means, stds, fileds_list
    means, stds = loading_means_stds(data_path, flag)
    list.append(sub_id)
    list.append(forecast_month)
    forecast_date = calc_date(month + 1, baseline)
    list.append(str(datetime.datetime.date(forecast_date)))
    list.append(pred[0, 0].cpu().data.numpy())
    list.append(pred[0, 1].cpu().data.numpy())
    list.append(pred[0, 2].cpu().data.numpy())
    adas13 = (pred[0, 6] * stds['ADAS13']
              + means['ADAS13']).cpu().data.numpy()
    adas13 = np.clip(adas13, 0, 85)
    list.append(adas13)
    list.append(adas13 - 1.0)
    list.append(adas13 + 1.0)
    v_icv = (pred[0, 19] * stds['Ventricles'] + means['Ventricles']) \
            / (pred[0, 17] * stds['ICV'] + means['ICV'])
    list.append(v_icv.cpu().data.numpy())
    list.append(v_icv.cpu().data.numpy() - 0.001)
    list.append(v_icv.cpu().data.numpy() + 0.001)

    return list


def list2csv(list, csv_path):
    columns_name = ["RID", "Forecast Month", "Forecast Date",
                    "CN relative probability", "MCI relative probability",
                    "AD relative probability", "ADAS13", "ADAS13 50% CI lower",
                    "ADAS13 50% CI upper", "Ventricles_ICV",
                    "Ventricles_ICV 50% CI lower", "Ventricles_ICV 50% CI upper"]
    df = pd.DataFrame(list, columns=columns_name)
    df.to_csv(csv_path, index=False, sep=',')


def evaluate_model(dev_pred_list, ref_dev_path, pred_dev_path):
    """
    evaluate model's performance
    """
    # save dev_pred_list as csv file
    list2csv(dev_pred_list, pred_dev_path)
    val_ref_frame = pd.read_csv(
        ref_dev_path,
        converters={
            'CognitiveAssessmentDate': str2date,
            'ScanDate': str2date
        })
    val_pred_frame = pd.read_csv(
        pred_dev_path, converters={'Forecast Date': str2date})

    return eval_submission(val_ref_frame, val_pred_frame)
