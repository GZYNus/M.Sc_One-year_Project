"""
Description: a wrapper function for training and validation, used for HORD search
Email: gzynus@gmail.com
Arthur: Zongyi Guo
"""
import warnings
import os
import argparse
import random
import numpy as np
import torch
from model.LSTM_net import LSTM


warnings.filterwarnings('ignore')


def train(args):
    # print('train')
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = LSTM(args)
    model.train(flag='split_train')
    # return model.test()
    a = model.test()
    tmp = np.array(list(a))
    return tmp


def get_args():
    """
    arguments feeding
    """
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    root_path = os.path.join(root_path, 'hord_LSTM')
    data_path = os.path.join(root_path, 'data')
    checkpoints = os.path.join(root_path, 'checkpoints')
    log_path = os.path.join(root_path, 'log')
    pred_path = os.path.join(root_path, 'prediction')
    eval_path = os.path.join(root_path, 'evaluation')

    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument('--test_fold', type=int, default=0)
    parser.add_argument('--NUM_SUB', type=int, default=1407)
    parser.add_argument('--MAX_LEN', type=int, default=130)
    parser.add_argument('--data_path', type=str, default=data_path)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--isTraining', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--checkpoints', '-o', default=checkpoints)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=23)
    parser.add_argument('--log_path', type=str, default=log_path)
    parser.add_argument('--pred_path', type=str, default=pred_path)
    parser.add_argument('--eval_path', type=str, default=eval_path)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--label', type=str, default='all')
    parser.add_argument('--window', type=int, default=1)

    parser.add_argument('--filling_method', type=str, default='linear')
    parser.add_argument('--out_path', type=str, default='./out')
    parser.add_argument('--pred_times', type=int, default=100)
    parser.add_argument('--output_size', type=int, default=25)

    # hyperparameter
    parser.add_argument('--lr', type=float, default=3.011901e-04)
    parser.add_argument('--l2', type=float, default=1.196042e-07)
    parser.add_argument('--hidden_size', type=int, default=863)
    parser.add_argument('--nb_layers', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    print(*train(get_args()))
