#!/usr/bin/env python
"""
Description: calculate the nearest date and the month intervals
Email: gzynus@gmail.com
Author: Zongyi Guo
"""
import datetime
from dateutil.relativedelta import relativedelta


def calc_date(month, baseline):
    """
    Calcuate date for kth month
    For example, if month = 3, baseline='2010-05'
    the date is '2010-08'
    :param month:
    :param baseline:
    :return:
    """
    date = baseline + relativedelta(months=month)
    return date


def compare_date(date, baseline):
    """
    Compare whether year-month-day exceeds baseline
    """
    baseline_year = datetime.datetime.strptime(baseline, '%Y-%m').year
    baseline_month = datetime.datetime.strptime(baseline, '%Y-%m').month
    year = date.year
    month = date.month

    if year < baseline_year:
        status = False
    elif year == baseline_year and month < baseline_month:
        status = False
    else:
        status = True
    return status


def month_between(end, start):
    # assert end > start
    diff = relativedelta(end, start)
    return diff.years * 12 + diff.months + (diff.days > 0)

if __name__ == '__main__':
    start = datetime.datetime.strptime('2005-9-8', "%Y-%m-%d")
    end = datetime.datetime.strptime('2015-9-22', "%Y-%m-%d")
    print(month_between(end, start))