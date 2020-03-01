import torch
import random
import numpy as np
import os
import torch.nn as nn
import torch.utils.data as Data
from lib.load_data import *
# from model.lstmnet import LSTMcell
from model.lstmnet import *
from torch.autograd import Variable
from evaluation.eva_model import *
from pred_individual.pred import predIndividual


def initHidden(n, hidden_size, num_layers):
    hidden = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(num_layers):
        hidden.append(Variable(torch.ones(n, hidden_size)/2).to(device))
    return hidden


def initCell(n, hidden_size, num_layers):
    C_N = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(num_layers):
        C_N.append(Variable(torch.ones(n, hidden_size) / 2).to(device))
    return C_N


class LSTM:
    def __init__(self, model_args):
        self.args = model_args
        # set random seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        # set GPU id
        torch.cuda.set_device(self.args.gpu)
        self.device = torch.device('cuda')
        # forward-filling or mixed filling
        self.filling_method = self.args.filling_method
        self.model_name = '{Hidden_units:' + str(self.args.hidden_size) \
                          + '}_{Filling_method:' + self.args.filling_method \
                          + '}_{Learning_rate:' + str(self.args.lr) \
                          + '}_{Net_layers:' + str(self.args.nb_layers) \
                          + '}_{L2Reg:' + str(self.args.l2) + '}'

    def train(self, flag='split_train'):
        """
        Training part
        """
        train_path = os.path.join(self.args.data_path, str(self.args.test_fold), 'zongyi_split_train_linear.pkl')

        MEAN, STD, list_name = loadMS(train_path)
        data_loader, mask_loader = loadNormData(self.args, MEAN, STD, train_path, list_name)
        lstm_model = LSTMnet(self.args.input_size, self.args.hidden_size, self.args.nb_layers, self.args.output_size).to(self.device)
        optimizer = torch.optim.Adam(
            lstm_model.parameters(), lr=self.args.lr, weight_decay=self.args.l2)
        trainset = TrainDataSet(data_loader, mask_loader)
        trainloader = Data.DataLoader(
            dataset=trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=16)
        for epoch in range(self.args.epochs):
            # print('Epoch: ', epoch)
            for step, DATALOADER in enumerate(trainloader):
                DATA_LOADER = DATALOADER[0].to(self.device)
                mask_batch = DATALOADER[1].to(self.device)
                lstm_model.zero_grad()
                H_N = initHidden(self.args.batch_size, self.args.hidden_size, self.args.nb_layers)
                C_N = initCell(self.args.batch_size, self.args.hidden_size, self.args.nb_layers)
                LOSS1 = 0
                LOSS2 = 0
                c_e = nn.CrossEntropyLoss().to(self.device)
                mae = nn.L1Loss(reduction='sum').to(self.device)
                iter_len = get_batch_max_length(mask_batch, self.args.input_size)
                for i in range(iter_len - 1):
                    month = DATA_LOADER[:, i, :]
                    # month = month.view(1, self.args.batch_size, self.args.input_size)
                    # print(month)
                    out = lstm_model(month, H_N, C_N)
                    A = torch.max(out[:, :3], 1)[1].float()
                    month_next = DATA_LOADER[:, i + 1, :]

                    nan_index = torch.isnan(month_next)
                    month_next[:, 0][nan_index[:, 0]] = A[nan_index[:, 0]]
                    month_next[:, 1:][nan_index[:, 1:]] = out[:, 3:][nan_index[:, 1:]]
                    DATA_LOADER[:, i + 1, :] = month_next

                    dx_index = torch.eq(mask_batch[:, i + 1, 0], 1)
                    if dx_index.sum() > 0:
                        LOSS1 += c_e(out[:, :3][dx_index],
                                     month_next[:, 0][dx_index].long())

                    mae_index = torch.eq(mask_batch[:, i + 1, 1:], 1)
                    if mae_index.sum() > 0:
                        LOSS2 += mae(out[:, 3:][mae_index],
                                     month_next[:, 1:][mae_index].float())

                LOSS = (LOSS1 + LOSS2) / self.args.batch_size
                # print('lOSS:', LOSS)
                LOSS.backward(retain_graph=True)
                optimizer.step()
        model_name = get_model_name(self.args)
        # print()
        torch.save(lstm_model,os.path.join(self.args.checkpoints, str(self.args.test_fold), model_name +'.pkl'))

    def test(self):
        self.args.batch_size = 1
        H_N = initHidden(self.args.batch_size, self.args.hidden_size, self.args.nb_layers)
        C_N = initCell(self.args.batch_size, self.args.hidden_size, self.args.nb_layers)
        model_name = get_model_name(self.args)
        lstm_model = torch.load(os.path.join(self.args.checkpoints, str(self.args.test_fold), model_name +'.pkl'))
        lstm_model.eval()
        pred_path = './data/' + str(self.args.test_fold) + '/zongyi_dev_linear.pkl'
        predListName, pred_CN, pred_MCI, pred_AD, ADAS13, ICV, Ventricles, pred_MEAN, pred_STD \
            = predIndividual(pred_path, lstm_model, self.args.pred_times, H_N, C_N)
        pre_save_path = './prediction/' + str(self.args.test_fold) + '/zongyi_dev_linear.csv'
        pred_to_csv(pre_save_path, predListName, pred_CN, pred_MCI, pred_AD, ADAS13, ICV, Ventricles, pred_MEAN,
                    pred_STD)
        ref_test_path = './data/'+ str(self.args.test_fold) + '/dev_ref.csv'

        mauc, bca, z_adasmae, adas, z_ventmae, vent = evaluate_model(ref_test_path, pre_save_path)
        # print('mAUC:', mauc, '\nBCA:', bca, '\nADAS:', adas, '\nVentical_ICV:', vent)
        return mauc, bca, z_adasmae, adas, z_ventmae, vent







