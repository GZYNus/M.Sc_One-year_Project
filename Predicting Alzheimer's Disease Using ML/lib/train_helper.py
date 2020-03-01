# pylint: disable=not-callable
"""
Description: self-defined functions
Email: gzynus@gmail.com
Author: Zongyi Guo
"""
import torch
import numpy as np


def padding_data(data_list, mask_list, num_features=23):
    """
    Padding Nan to make data_list and mask_list match maximum length
    :param data_list: uninterpolated data_list
    :param mask_list: mask_list, True means feature exists, False means feature doesn't exist
    :return: in format of tensor, in shape of [num_subjects,max_length, num_features]
    """
    # get the maximum length
    max_length = 0
    for item in enumerate(data_list):
        if np.array(item[1]).shape[0] > max_length:
            max_length = np.array(item[1].shape[0])

    data_array = np.zeros((len(data_list), max_length, num_features))
    mask_array = np.zeros((len(data_list), max_length, num_features))

    for item in enumerate(data_list):
        if np.array(item[1]).shape[0] < max_length:
            padding_length = max_length - np.array(item[1]).shape[0]
            pad_array = np.full((padding_length, num_features), np.nan)
            # zero_array = np.full((padding_length, num_features), 0)
            data_array[item[0], :, :] = np.concatenate(                   #这里data_array 和 mask_array同时padding
                (np.array(item[1]), pad_array), axis=0)
            mask_array[item[0], :, :] = np.concatenate(
                (np.array(mask_list[item[0]]), pad_array), axis=0)
    # convert ndarray to tensor
    data_tensor = torch.tensor(data_array)
    mask_tensor = torch.tensor(mask_array)

    return data_tensor, mask_tensor


def get_batch_max_length(batch_masks):
    """
    get the maximum length we need to use
    until all features are padded to match maximum length
    :param batch_masks:
    :return:
    """
    b_max_length = 1
    for i in range(batch_masks.shape[0]):
        if torch.isnan(batch_masks[i, :, :]).sum() == batch_masks.shape[1] * 23:
            break
        else:
            b_max_length = i + 1

    return b_max_length


def update_nan_next_month(next_month, diagnosis_pred, continuous_pred):
    """
    update Nan value in next month with predictin
    :param next_month:
    :param diagnosis_pred:
    :param continuous_pred:
    :return:
    """
    # get the Nan index of nex_month
    nan_index_nex = torch.isnan(next_month)
    # update the Nan in nex_month
    next_month[:, 0][nan_index_nex[:, 0]] = \
        diagnosis_pred[nan_index_nex[:, 0]].double()
    next_month[:, 1:][nan_index_nex[:, 1:]] = \
        continuous_pred[nan_index_nex[:, 1:]].double()

    return next_month

