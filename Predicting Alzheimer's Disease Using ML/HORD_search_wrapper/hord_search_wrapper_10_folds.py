#!/usr/bin/env python
"""
Description:
Date: 11/7/19
Email: anlijuncn@gmail.com
Writen by Lijun AN and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""
import os
import argparse
from hord_search import hord_search


def get_args():
    """
    arguments feeding
    """
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    root_path = os.path.join(root_path, 'hord_LSTM')
    data_path = os.path.join(root_path, 'data')
    model_path = os.path.join(root_path, 'checkpoints')
    log_path = os.path.join(root_path, 'log')
    pred_path = os.path.join(root_path, 'prediction')
    eval_path = os.path.join(root_path, 'evaluation')
    checkpoint_path = os.path.join(root_path, 'checkpoints')

    parser = argparse.ArgumentParser()
    parser.add_argument('--nthreads', type=int, default=4)
    parser.add_argument('--maxeval', type=int, default=60)
    parser.add_argument('--experiment', default='lstm_mod')
    parser.add_argument('--logstem', default='log')

    parser.add_argument('--label', type=str, default='all')
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--nb_layers', type=int, default=2)
    # general parameters
    parser.add_argument('--test_fold', type=int, default=0)
    parser.add_argument('--NUM_SUB', type=int, default=1407)
    parser.add_argument('--MAX_LEN', type=int, default=130)
    parser.add_argument('--data_path', type=str, default=data_path)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--isTraining', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--model_path', '-o', default=model_path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=23)
    parser.add_argument('--log_path', type=str, default=log_path)
    parser.add_argument('--pred_path', type=str, default=pred_path)
    parser.add_argument('--eval_path', type=str, default=eval_path)
    parser.add_argument('--checkpoint', type=str, default=checkpoint_path)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--filling_method', type=str, default='linear')
    parser.add_argument('--out_path', type=str, default='./out')
    parser.add_argument('--pred_times', type=int, default=100)
    parser.add_argument('--output_size', type=int, default=25)

    # hyperparameter
    parser.add_argument('--lr', type=float, default=9.810491e-04)
    parser.add_argument('--l2', type=float, default=9.146395e-07)
    parser.add_argument('--hidden_size', type=int, default=473)
    # parser.add_argument('--nb_layers', type=int, default=1)

    return parser.parse_args()


for test_fold in range(2,10):
    nub_subj_list = [1407, 1407, 1408, 1409, 1409, 1409, 1409, 1409, 1409, 1408]
    args = get_args()
    args.NUM_SUB = nub_subj_list[test_fold]
    print(args.NUM_SUB)
    args.test_fold = test_fold
    hord_search(args)


