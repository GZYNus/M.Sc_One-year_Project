#!/usr/bin/env python
"""
Description: generate 10_fold cross-validation
Email: gzynus@gmail.com
Author: Zongyi Guo
"""
import os
import csv
import random
import sys
from sklearn.model_selection import *
from lib.load_data import loading_subject_list



def get_all_subjects_list(data_path, output_path):
    """
    get lists for all subjects from data.csv
    :param data_path:
    :param output_path:
    :return:
    """
    data_csv_path = os.path.join(data_path, 'data.csv')
    subs_dict = {}
    with open(data_csv_path) as file:
        data_csv = csv.reader(file)
        data_header = next(data_csv)
        for row in data_csv:
            if int(row[0]) not in subs_dict.keys():
                subs_dict[int(row[0])] = 0
    sub_list = sorted(subs_dict.keys())
    # write sub_list to a txt file
    save_path = os.path.join(output_path, 'full_subject_list.txt')
    fileObject = open(save_path, 'w')
    for sub in sub_list:
        fileObject.write(str(sub))
        fileObject.write('\n')
    fileObject.close()
    print('Finish generating all subject lists! Total: ', len(sub_list))


def gen_k_fold_splits(data_path, output_path, k):
    """
    generate k-fold split for k-fold cross validation
    :param data_path:
    :param output_path:
    :param k: num of folds
    :return:
    """
    full_subs = loading_subject_list(data_path, flag='full')
    # using sklearn to generate k-fold splits
    kf = KFold(n_splits=k)
    fold = 0
    for train_index, test_index in kf.split(full_subs):
        fold += 1
        train_list = list(train_index)
        test_list = list(test_index)
        # randomly pick 10% subjects as validation subjects in train_list
        dev_list = random.sample(train_list, int(0.1 * len(train_list)))
        assert (len(train_list) + len(test_list) + len(dev_list))\
            == len(full_subs)
        # save them in form of txt file
        train_list_name = str(fold) + '_train_subject_list.txt'
        save_k_fold_splits(train_list, output_path, train_list_name, fold)

        dev_list_name = str(fold) + '_train_subject_list.txt'
        save_k_fold_splits(dev_list, output_path, dev_list_name, fold)

        test_list_name = str(fold) + '_test_subject_list.txt'
        save_k_fold_splits(test_list, output_path, test_list_name, fold)


def save_k_fold_splits(list, output_path, list_name, k):
    """
    save splits in form of txt file
    """
    # create folder for saving k-th split
    if os.path.isdir(os.path.join(output_path, str(k))):
        pass
    else:
        os.mkdir(os.path.join(output_path, str(k)))
    save_path = os.path.join(output_path, str(k), list_name)
    fileObject = open(save_path, 'w')
    for sub in list:
        fileObject.write(str(sub))
        fileObject.write('\n')
    fileObject.close()


if __name__ == '__main__':

    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    output_path = os.path.join(root_path, 'splits')
    # get_all_subjects_list(data_path, output_path)
    # gen_split(output_path, output_path)
    gen_k_fold_splits(output_path, output_path, 10)


