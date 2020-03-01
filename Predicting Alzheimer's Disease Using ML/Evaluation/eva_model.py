import datetime
import pandas as pd
import numpy as np
from datetime import datetime

from evaluation.evalOneSubmission import eval_submission
from evaluation.evalOneSubmission import str2date

def str2date(string):
    return datetime.strptime(string, '%Y-%m-%d')

def list2csv(list, csv_path):
    columns_name = ["RID", "Forecast Month", "Forecast Date",
                    "CN relative probability", "MCI relative probability",
                    "AD relative probability", "ADAS13", "ADAS13 50% CI lower",
                    "ADAS13 50% CI upper", "Ventricles_ICV",
                    "Ventricles_ICV 50% CI lower", "Ventricles_ICV 50% CI upper"]
    df = pd.DataFrame(list, columns=columns_name)
    df.to_csv(csv_path, index=False, sep=',')


def evaluate_model(ref_dev_path, pred_dev_path):
    """
    evaluate model's performance
    """
    # save dev_pred_list as csv file
    val_ref_frame = pd.read_csv(ref_dev_path)
    # val_ref_frame = val_ref_frame.drop(index = [0,1,2])
    # val_ref_frame = val_ref_frame.drop(index=[48,53,97,195,219,232,358,367,415,416])
    val_ref_frame.to_csv(ref_dev_path,index=None)
    val_ref_frame = pd.read_csv(
        ref_dev_path,
        converters={
            'CognitiveAssessmentDate': str2date,
            'ScanDate': str2date
        })
    val_pred_frame = pd.read_csv(
        pred_dev_path, converters={'Forecast Date': str2date})

    return eval_submission(val_ref_frame, val_pred_frame)


if  __name__  == "__main__":
    fold = 0
    pre_path = '/home/zyguo/storage/LSTM/data/'+str(fold) + '/zongyi_dev_linear.csv'
    ref_test_path = '/home/zyguo/storage/LSTM/data/0/dev_ref.csv'

    mauc, bca, z_adasmae, adas, z_ventmae, vent = evaluate_model(ref_test_path,pre_path)
    print('mAUC:',mauc, '\nBCA:',bca, '\nADAS:',adas, '\nVentical_ICV:',vent)



