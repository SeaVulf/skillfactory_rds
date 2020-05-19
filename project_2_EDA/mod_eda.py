
# %%

# Importing required libraries
import pandas as pd
import numpy as np
import math

from itertools import combinations
from scipy.stats import ttest_ind
from random import random

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
df = pd.DataFrame({'foo': [10,4,8, None, 6, 16, 4, 4, None],
                  'bar': ['T', 'T', 'A', None, 'T', 'A', 'A', 'T', 'A']})
df.head()

# %
# ФУНКЦИЯ ЗАПОЛНЕНИЯ ПРОПУСКОВ
def fill_rand(main_series, sub_series=pd.Series([])):
    # Подсчёт пропорции
    main_prop = main_series.value_counts(normalize=
                                                True).sort_values(ascending = False) 

    # Начальные условия 
    dict = {}
    start_val = 0
    end_val = 0

    #Составление таблицы распределения вероятностей основной колонки
    for i in range(len(main_prop)):
        if i < len(main_prop) - 1:
            end_val = end_val + main_prop.iloc[i]
        else:
            end_val = 1
        dict[main_prop.index[i]] = [start_val, end_val]
        start_val = end_val
    df_main_prop = pd.DataFrame(dict).T
    
    # Если заполнение происходит только на основании пропорции главной серии
    if len(sub_series) == 0: 

        # Получение пропущенных значений главного столбца
        main_nan = main_series[main_series.isna()]

        # Получение списка для заполнения
        fill_list = []
        for i in range(len(main_nan)):
            rnd = random()
            fill_list.append(df_main_prop[(df_main_prop[0] < rnd) & (df_main_prop[1] > rnd)].index[0])        
        
        # Заполнение пропусков
        main_series[main_series.isna()] = fill_list

        return main_series
    
    
    # Случай, когда имеется дополнительный Series для учёта.
    else:
        # Подсчёт пропорции по второму столбцу
        sub_prop = sub_series.value_counts(normalize=
                                                True).sort_values(ascending = False)

        # Начальные условия 
        dict = {}
        start_val = 0
        end_val = 0

        #Составление таблицы распределения вероятностей вспомогательной колонки
        for i in range(len(sub_prop)):
            if i < len(sub_prop) - 1:
                end_val = end_val + sub_prop.iloc[i]
            else:
                end_val = 1
            dict[sub_prop.index[i]] = [start_val, end_val]
            start_val = end_val
        df_sub_prop = pd.DataFrame(dict).T

        # Вычислим условную вероятность
            # Подсчёт количества значений главного столбца
        main_count = main_series.value_counts().sort_values(ascending=False).reset_index()
        main_count.columns=['main','main_count']

            # Объединение заданных колонок в один датафрейм
        serv_df = pd.DataFrame({'main': main_series, 'sub': sub_series})

            # Распределение количества значений, соответствующих паре: "основное, дополнительное" значения
        distr = pd.DataFrame(serv_df.groupby(['main'])['sub'].value_counts())
        distr.columns = ['sub_count']
        distr.reset_index(inplace=True)

            # Получение столбца пропорций
        distr = distr.merge(main_count, on='main', how='left')
        distr['cond_prop'] = distr.sub_count/distr.main_count

        # Добавление вероятностей, вычисленных ранее
        main_prop = main_prop.reset_index()
        main_prop.columns=['main','main_prop'] 

        sub_prop = sub_prop.reset_index()
        sub_prop.columns=['sub','sub_prop'] 

        distr = distr.merge(main_prop, on='main', how='left')
        distr = distr.merge(sub_prop, on='sub', how='left')


        # Вычислим вероятность наступления события и добавим в общую таблицу
        event_prop = pd.DataFrame()
        for sub_val in distr['sub'].unique():
            event_prop[sub_val] = [(distr[distr['sub'] == sub_val]['main_prop']* \
                                distr[distr['sub'] == sub_val]['cond_prop']).sum()]
        
        event_prop = event_prop.T
        event_prop.reset_index(inplace=True)
        event_prop.columns=['sub','event_prop']

        distr = distr.merge(event_prop, on='sub', how='left')

        # Вычислим апостериорную вероятность
        distr['apost_prop'] = distr['main_prop']*distr['cond_prop']/distr['event_prop']

        # Получение значений вспомогательного столбца, соответствующих значениям главного столбца
        serv_nan = serv_df[serv_df['main'].isna()]

        # Получение списка для заполнения
        fill_list = []
        for i in range(len(serv_nan)):
            # TODO: РЕШИТЬ ПРОБЛЕМУ связанную с наличием ОБОИХ
            # Блок кода ниже справедлив в случае, когда соответствующее значение 'sub' не None:
            if pd.notna(serv_nan.iloc[i]['sub']):
                part_distr = distr[distr['sub'] == serv_nan.iloc[i]['sub']]

                # Начальные условия 
                dict = {}
                start_val = 0
                end_val = 0

                #Составление таблицы распределения апостериорных вероятностей 
                for j in range(len(part_distr)):
                    if j < len(part_distr) - 1:
                        end_val = end_val + part_distr.iloc[j]['apost_prop']
                    else:
                        end_val = 1
                    dict[part_distr.iloc[j]['main']] = [start_val, end_val]
                    start_val = end_val
                df_real_prop = pd.DataFrame(dict).T

                rnd = random()

                fill_list.append(df_real_prop[(df_real_prop[0] < rnd) & (df_real_prop[1] > rnd)].index[0])
            
            # Если пропущено значение в колонке "sub"
            else:
                # Получение списка для заполнения
                rnd = random()
                fill_list.append(df_main_prop[(df_main_prop[0] < rnd) & (df_main_prop[1] > rnd)].index[0])        
            
        # Заполнение пропусков
        main_series[main_series.isna()] = fill_list

        return main_series
    
    return 0



# %
# fill_rand(df['foo'], df['bar'])

