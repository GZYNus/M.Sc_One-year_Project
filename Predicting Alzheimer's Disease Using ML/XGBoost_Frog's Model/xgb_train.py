"""
Description: Train XGBoost
Email: gzynus@gmail.com
Arthur: Zongyi Guo
"""

import numpy as np
import xgboost as xgb
import pickle as pk
import math
import os
from lib.load_data import get_model_name


def xgb_train(args, xtr, ytr, xval):
    xtr = xtr.cpu().detach().numpy()
    ytr = ytr.cpu().detach().numpy().tolist()
    xval = xval.cpu().detach().numpy()
    j=[]
    y_notnan = []
    z_mean = np.mean(xtr, axis=0)
    z_std = np.std(xtr, axis=0)

    np.savetxt('/home/zyguo/storage/LSTM_Exp/LSTM_XGB/z_param.txt',(z_mean,z_std))

    xtr = (xtr - z_mean) / z_std
    xval = (xval - z_mean) / z_std
    for i in range(len(ytr)):
        if not math.isnan(ytr[i]):
            j.append(i)
            y_notnan.append(ytr[i])
    print('train shape:',xtr[j].shape)
    dtrain = xgb.DMatrix(xtr[j], label=y_notnan)
    dval = xgb.DMatrix(xval)
    param = {
        # 'objective':'multi:softprob',
        # 'num_class':3,
        # 'eval_metric':'mlogloss',
        'objective':'reg:squarederror',
        'max_depth':10,
        'lambda':0.1,
        'subsample':0.9,
        'eta':0.01,
        'colsample_bytree':0.9,
        'eval_metric': 'mae',
        'seed':'17',
        'nthread':8,
        'gpu_id':0,
        'tree_method':'gpu_hist'
    }
    watchlist = [(dtrain,'train')]
    num_rounds = 1000
    plst = param.items()
    gbm = xgb.train(plst,dtrain,num_rounds, watchlist)

    pred = gbm.predict(dval)
    model_name = get_model_name(args)
    model_save_path = os.path.join(args.checkpoints, str(args.test_fold), 'xgb_'+model_name+'.pkl')
    pk.dump(gbm, open(model_save_path, 'wb'))

    return pred







