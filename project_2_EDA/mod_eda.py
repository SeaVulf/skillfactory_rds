# Importing required libraries
import pandas as pd
import numpy as np
import math

from itertools import combinations
from scipy.stats import ttest_ind

def get_boards(series):
    """Get outlier's boards by calculating Interquartile range in pandas.DataFrame.Series 

    Arguments:
        series {pandas.DataFrame.Series} -- Series for finding outlier's boards

    Returns:
        list -- Two values list with outlier's boards
    """
    Q1 = pd.Series.quantile(series,q=0.25)
    Q3 = pd.Series.quantile(series,q=0.75)
    IQR = Q3 - Q1
    outlier_boards = [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    return outlier_boards

def find_nulls(series):
    """Find nulls like space symbols and None

    Arguments:
        series {pandas.DataFrame.Series} -- Series for finding nulls

    Returns:
        list -- Text information and count of None-values and skips.
    """
    #Число пустых значений:
    isna_count = len(series[series.isna()])

    #Имеются ли None - элементы
    if isna_count == 0:
        answer = 'None-значения отсутствуют. '
    else:
        answer = f'{str(isna_count)} - количество none-значений. '

    skips_count = len(series[series.astype('str').str.strip() == ''])

    #Имеются ли пробельные пропуски
    if skips_count == 0:
        answer = answer + 'Пробельные пропуски отсутствуют.'
    else:
        answer = answer + f'{str(skips_count)} - число пробельных пропусков.' 
    return [answer, isna_count, skips_count]

#Тестовый датафрейм
#df = pd.DataFrame({'foo': [4,8,25,None,16,''],
                    #'bar': ['Текст', ' ', '', None, 'My', ' f'],
                    #'ideal': [5, 8, 25, 6, 45, 20]})
#df.head()

