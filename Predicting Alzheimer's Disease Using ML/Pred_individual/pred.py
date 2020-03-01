import pandas as pd
import torch
import torch.nn as nn
import _pickle as pk
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")


def predIndividual(pred_path, LSTM, pred_times, H_N, C_N):
    pred_CN = []
    pred_MCI = []
    pred_AD = []
    ADAS13 = []
    ICV = []
    Ventricles = []
    predListName = []

    pred_data = pk.load(open(pred_path, 'rb'), encoding='iso-8859-1')
    list_name =list(pred_data.keys())
    list_name = sorted(list_name)

    #normalization
    for i in list_name:
        df_list = pd.DataFrame(pred_data[i]['input'][:,1:])
        # df_list = df_list.dropna(axis='rows',how = 'all')
        df_list = df_list.values
        pred_data[i]['input'] = df_list
        a = pred_data[i]['input']
        a = np.array(a)
        if i == list_name[0]:
            dat = a
        else:
            dat = np.vstack((dat, a))

    pred_MEAN = np.nanmean(dat[:, 1:], axis=0)
    pred_STD = np.nanstd(dat[:, 1:], axis=0)

    for i in list_name:
        H_N2 = H_N          #every subject should have their own h_n & c_n
        C_N2 = C_N
        x = pred_data[i]['input']
        list_loader = np.full([110,23],np.nan)
        for r in range(len(x[:,0])):
            list_loader[r] = x[r]
        #normalization
        x1 = list_loader[:,0]
        x1 = x1[:,np.newaxis]
        x2 = (list_loader[:,1:] - pred_MEAN)/pred_STD
        x = np.hstack((x1,x2))
        x = torch.from_numpy(x)
        x = x.float()
        x = x.to(device)
        predListName.append(i)
        # print(predListName)
        for j in range(pred_times):
            month = x[j]
            month = month.unsqueeze(0)
            out, H_N2, C_N2 = LSTM(month, H_N2, C_N2)
            A = torch.max(out[:, :3], 1)[1].float()
            month_next = x[j+1]
            month_next = month_next.unsqueeze(0)
            nan_index = torch.isnan(month_next)

            month_next[:,0][nan_index[:,0]] = A[nan_index[:,0]]
            month_next[:,1:][nan_index[:,1:]] = out[:, 3:][nan_index[:,1:]]
            x[j+1] = month_next
            out = out.cpu()
            sof = F.softmax(out[:, :3], dim=1)
            pred = torch.max(sof, 1)[1].numpy().tolist()
            sof = sof.detach().numpy().tolist()
            pred_CN.append(sof[0][0])
            pred_MCI.append(sof[0][1])
            pred_AD.append(sof[0][2])
            ADAS13 += out[:, 5].detach().numpy().tolist()
            ICV += out[:, 16].detach().numpy().tolist()
            Ventricles += out[:, 18].detach().numpy().tolist()

    return predListName, pred_CN, pred_MCI, pred_AD, ADAS13, ICV, Ventricles, pred_MEAN, pred_STD


