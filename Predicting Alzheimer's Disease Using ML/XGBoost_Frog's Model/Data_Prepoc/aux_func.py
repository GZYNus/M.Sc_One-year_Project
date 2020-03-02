#!/usr/bin/env python
"""
Description: augmented functions used in expanding into tadpole_train.csv
Email:gzynus@gmail.com
Author: Zongyi Guo
"""
import os
import numpy as np
from datetime import datetime
from dateutil import relativedelta
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


def month_between(end, start):
    """ Get duration (in months) between *end* and *start* dates """
    # assert end >= start
    end = datetime.strptime(end, "%Y-%m-%d")
    start = datetime.strptime(start, "%Y-%m-%d")
    diff = relativedelta(end, start)
    months = 12 * diff.years + diff.months
    to_next = relativedelta(end + relativedelta(months=1, days=-diff.days),
                            end).days
    to_prev = diff.days
    return months + (to_next < to_prev)


def create_long_features(df_array, dates, jj, kk):
    # create
    past_mask = (dates <= dates[jj])
    past = df_array[past_mask]
    past_dates = dates[: jj + 1]
    # get non nan
    if np.sum(~np.isnan(past)) > 0:
        non_nan_past = past[~np.isnan(past)]
        non_nan_dates = past_dates[~np.isnan(past)]
        most_recent = non_nan_past[-1]
        time_since_most_recent = month_between(dates[kk], non_nan_dates[-1])
        lowest = np.min(non_nan_past)
        time_since_lowest = month_between(dates[kk], non_nan_dates[non_nan_past == lowest][0])
        highest = np.max(non_nan_past)
        time_since_highest = month_between(dates[kk], non_nan_dates[non_nan_past == highest][0])
        if np.sum(~np.isnan(past)) > 1:
            mr_change = non_nan_past[-1] - non_nan_past[-2]
            time_diff = month_between(dates[kk], non_nan_dates[-2])
            mr_change /= time_diff
        else:
            mr_change = np.nan

        return [most_recent, time_since_most_recent, mr_change, lowest,
                time_since_lowest, highest, time_since_highest]

    else:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]


def create_dx_features(df_array, dates, jj, kk):
    # create a expanded dx features
    expanded = create_long_features(df_array, dates, jj, kk)
    # we need to remove mr_change and add milder, time_since_milder
    expanded.pop(2)  # remove mr_change
    past_mask = dates <= dates[jj]
    past = df_array[past_mask]
    past_dates = dates[: jj + 1]
    non_nan_past = past[~np.isnan(past)]
    non_nan_dates = past_dates[~np.isnan(past)]

    most_recent = expanded[0]
    tmp = non_nan_past[non_nan_past < most_recent]
    tmp_dates = non_nan_dates[non_nan_past < most_recent]
    if len(tmp) == 0:
        expanded.append(0)
        expanded.append(999)
    else:
        expanded.append(1)
        expanded.append(month_between(dates[kk], tmp_dates[-1]))

    return expanded


def feature_names(feature):
    """
    generate a list of expanded feature names
    :param feature:
    :return:
    """
    if feature == 'DX':
        return ['mr_dx', 'time_since_mr_dx', 'best_dx', 'time_since_best_dx',
                'worst_dx', 'time_since_worst_dx', 'milder',
                'time_since_midler']
    else:
        return ['mr_' + feature, 'time_since_mr_' + feature,
                'mr_change_' + feature, 'low_' + feature, 'time_since_low_' +
                feature, 'high_' + feature, 'time_since_high_' + feature]





