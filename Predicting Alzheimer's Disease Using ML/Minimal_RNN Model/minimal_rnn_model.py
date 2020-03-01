# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-callable
"""
Train RNN model, evaluate model's performance and make prediction
Writen by Lijun AN and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
from model.minimal_rnn_net import MinimalRNNNet
from lib.load_data import loading_znormalized_data
from lib.load_data import TrainDataSet
from lib.train_helper import padding_data
from lib.train_helper import get_batch_max_length
from lib.train_helper import update_nan_next_month
from lib.test_helper import pred2list
from lib.test_helper import save_pred
from lib.test_helper import evaluate_model


class ADRnn:
    """
    RNN model for AD prediction
    """
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
                          + '}_{Input_dropput:' + str(self.args.indrop) \
                          + '}_{Recurrent_dropout:' + str(self.args.redrop) \
                          + '}_{Net_layers:' + str(self.args.layers) \
                          + '}_{L2Reg:' + str(self.args.l2) + '}'

    def train_forward(self, model, hiddens, batch_inputs, batch_masks, dropout_masks):
        """
        forward pass of training, return loss
        """
        # loss functions
        loss_func1 = nn.NLLLoss(reduction='sum').to(self.device)
        loss_func2 = nn.L1Loss(reduction='sum').to(self.device)
        loss_ce, loss_mae = 0, 0
        # find the maximum time length in this batch
        b_max_length = get_batch_max_length(batch_masks)
        batch_inputs = batch_inputs.to(self.device)
        batch_masks = batch_masks.to(self.device)
        for month in range(b_max_length - 1):
            cur_month = batch_inputs[month, :, :].float()
            nex_month = batch_inputs[month + 1, :, :]
            diagnosis_prob, diagnosis_pred, continuous_pred, hiddens = model(Variable(cur_month), hiddens, dropout_masks)
            # update Nan in next month
            batch_inputs[month + 1, :, :] = \
                update_nan_next_month(nex_month, diagnosis_pred, continuous_pred)
            # calculate loss
            dx_index = torch.eq(batch_masks[month + 1, :, 0], 1)
            if dx_index.sum() > 0:
                loss_ce += loss_func1(torch.log(diagnosis_prob)[dx_index],
                                      nex_month[:, 0][dx_index].long())
            mae_index = torch.eq(batch_masks[month + 1, :, 1:], 1)
            if mae_index.sum() > 0:
                loss_mae += loss_func2(continuous_pred[mae_index],
                                       nex_month[:, 1:][mae_index].float())
        return loss_ce/batch_masks.shape[1], loss_mae/batch_masks.shape[1]

    def train(self, flag='split_train'):
        """
        Training part
        """
        # loading data after z-normalization
        data_list, mask_list, _, _, _, _ = loading_znormalized_data(
            data_path=self.args.data_path,
            flag=flag,
            method=self.args.filling_method)
        # padding with Nan to match maximum length
        data_tensor, mask_tensor = padding_data(data_list, mask_list)
        trainset = TrainDataSet(data_tensor, mask_tensor)
        trainloader = Data.DataLoader(
            dataset=trainset,                                #上面的trainset就在为这里的dataset做准备  trainset一定要继承了dataset的两个函数
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=16)
        # construct model from MinimalRnnNet
        model = MinimalRNNNet(self.args).to(self.device)
        optimizier = torch.optim.Adam(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.l2)
        # begin training
        for epoch in range(self.args.epochs):
            print('Epoch: ', epoch)
            for _, batch_data in enumerate(trainloader):
                # hidden state initialization and dropmasks
                hiddens, dropout_masks = model.net_init()
                # transpose shape to (max_length, batch_size, num_features]
                batch_inputs = batch_data[0].transpose(0, 1)
                batch_masks = batch_data[1].transpose(0, 1)
                # zero grad
                model.zero_grad()
                # calculate loss of this batch
                loss_ce, loss_mae = ADRnn.train_forward(
                    self, model, hiddens, batch_inputs, batch_masks, dropout_masks)
                loss = loss_ce + loss_mae
                loss.backward(retain_graph=True)
                optimizier.step()
        # saving trained model
        torch.save(model.state_dict(), os.path.join(self.args.model_path,
                                                    self.model_name))

    def test(self, model_name=None, flag='dev'):
        """
        making prediction
        """
        # load trained model
        if model_name is not None:
            self.model_name = model_name
        # loading trained model
        self.args.batch_size = 1
        self.args.isTraining = False
        model = MinimalRNNNet(self.args).to(self.device)
        model.load_state_dict(
            torch.load(os.path.join(self.args.model_path, self.model_name)))
        model.eval()
        return ADRnn.test_forward(self, model, flag)

    def test_forward(self, model, flag):
        """
        making prediction
        """
        # list for saving prediction
        test_pred_list = []
        # loading data
        te_inputs_list, _, te_baseline_list, te_id_list, te_length_list, \
            te_split_index_list = loading_znormalized_data(self.args.data_path,
                                                           self.args.filling_method,
                                                           flag=flag)
        for sub, te_input in enumerate(te_inputs_list):
            te_input_tensor = torch.tensor(te_input)
            te_input_tensor = te_input_tensor.to(self.device).view(-1, 1, 23)   #3维tensor (seq_len,batch，features)
            test_sub_pred_list = \
                ADRnn.test_forward_individual(self, model,
                                              te_input_tensor,
                                              te_baseline_list[sub],
                                              te_id_list[sub],
                                              te_length_list[sub],
                                              te_split_index_list[sub],
                                              flag)
            test_pred_list += test_sub_pred_list

        ref_path = os.path.join(self.args.eval_path, flag + '_ref.csv')
        pred_path = os.path.join(self.args.pred_path, flag + self.model_name
                                 + '.csv')
        return evaluate_model(test_pred_list, ref_path, pred_path)

    def test_forward_individual(self, model, te_input_tensor, te_baseline,
                                te_id, te_length, te_split_index, flag):
        """
        making prediction for each subject
        """
        test_sub_pred_list = []
        hiddens, dropout_masks = model.net_init()
        exceed_end_date = False
        month = 0
        forecast_month = 0
        pred = np.full((1, 26), np.nan)
        pred = torch.tensor(pred).to(self.device)
        while not exceed_end_date:
            # exceed_start_date = (month >= te_split_index - 2)
            exceed_end_date = (month >= te_length)
            if month < te_input_tensor.shape[0]:
                cur_month = te_input_tensor[month, :, :].float()
                diagnosis_prob, diagnosis_pred, continuous_pred, hiddens = \
                    model(cur_month, hiddens, dropout_masks)
                # update the Nan in nex_month
                if month < te_input_tensor.shape[0] - 1:
                    te_input_tensor[month + 1, :, :] = \
                        update_nan_next_month(te_input_tensor[month + 1, :, :],
                                              diagnosis_pred, continuous_pred)

            else:
                cur_month = pred[:, 3:].float()
                diagnosis_prob, diagnosis_pred, continuous_pred, hiddens = model(
                    cur_month, hiddens, dropout_masks)

            pred = save_pred(pred, diagnosis_prob, diagnosis_pred, continuous_pred)
            # if exceed_start_date:
            # write to a sub_pred_list
            forecast_month += 1
            row = pred2list(pred, te_id, month,
                            forecast_month, te_baseline, self.args.data_path, flag)
            test_sub_pred_list.append(row)
            month += 1
        return test_sub_pred_list


def get_args():
    """
    arguments feeding
    """
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    model_path = os.path.join(root_path, 'checkpoints')
    log_path = os.path.join(root_path, 'log')
    pred_path = os.path.join(root_path, 'prediction')
    eval_path = os.path.join(root_path, 'evaluation')

    model_parser = argparse.ArgumentParser()
    # general parameters
    model_parser.add_argument('--data_path', type=str, default=data_path)
    model_parser.add_argument('--batch_size', type=int, default=128)
    model_parser.add_argument('--isTraining', type=bool, default=True)
    model_parser.add_argument('--epochs', type=int, default=10)
    model_parser.add_argument('--model_path', '-o', default=model_path)
    model_parser.add_argument('--gpu', type=int, default=1)
    model_parser.add_argument('--input_size', type=int, default=23)
    model_parser.add_argument('--log_path', type=str, default=log_path)
    model_parser.add_argument('--pred_path', type=str, default=pred_path)
    model_parser.add_argument('--eval_path', type=str, default=eval_path)
    model_parser.add_argument('--seed', type=int, default=17)
    model_parser.add_argument('--filling_method', type=str, default='linear')
    model_parser.add_argument('--pred_start_date', type=str, default='2010-05')
    model_parser.add_argument('--pred_end_date', type=str, default='2017-04')

    # hyper parameter
    model_parser.add_argument('--lr', type=float, default=8.483429e-04)
    model_parser.add_argument('--l2', type=float, default=6.105402e-07)
    model_parser.add_argument('--indrop', type=float, default=0.007143)
    model_parser.add_argument('--redrop', type=float, default=0.092857)
    model_parser.add_argument('--hidden_size', type=int, default=169)
    model_parser.add_argument('--layers', type=int, default=2)

    return model_parser.parse_args()


if __name__ == '__main__':
    DEMO = ADRnn(model_args=(get_args()))
    DEMO.train(flag='train')
    print(DEMO.test(flag='test'))
    # print(DEMO.test(model_name='model', flag='dev'))

"""
 --lr 8.483429e-04 --l2 6.105402e-07 --redrop 0.092857 --indrop 0.007143 --hidden_size 169 --layers 2
 (0.9594038904976403, 0.8914861992901022, 0.3619878738403385, 4.361439035136514, 0.12547753082425062, 0.0016988484775445618)
 Best solution found: {'lr': '9.329189e-05', 'l2': '5.389202e-07', 'redrop': '0.000000', 'indrop': '0.000000', 'hidden_size': '247', 'layers': '4'}
(0.9538394197769199, 0.8683927128773917, 0.3308125028084924, 3.98582015401032, 0.13052199676245485, 0.001767145831045619)
"""

