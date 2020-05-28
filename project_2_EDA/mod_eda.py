# Importing required libraries
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from scipy.stats import ttest_ind
from scipy.stats import norm
from random import random

# Функция определения границ выбросов
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

# Функция поиска None и пробельных пропусков
def find_nulls(series):
    """Find nulls like space symbols and None

    Arguments:
        series {pandas.DataFrame.Series} -- Series for finding nulls

    Returns:
        dictionary -- count of None-values and skips, their sum, rate of total length of series.
    """
    answer_dict = {
            'skip_nan': 0, 
            'skip_space': 0,
            'all_skips': 0,
            'frac_skips': 0
            }
    if len(series) > 0:
        #Число пустых значений:
        answer_dict['skip_nan'] = len(series[series.isna()])
        # Число пробельных пропусков:
        answer_dict['skip_space'] = len(series[series.astype('str').str.strip() == ''])
        # Суммарное количество пропусков
        answer_dict['all_skips'] = answer_dict['skip_nan'] + answer_dict['skip_space']
        # Доля пропусков от размера Серии
        answer_dict['frac_skips'] = answer_dict['all_skips']/len(series)

    return answer_dict

# Функция заполнения пропусков
def fill_rand(main_series, sub_series=pd.Series([])):

    """Filling gaps in main_series with imputation method 
    by using the proportion of values in this series and (optional)
    by Thomas Bayes formula for posterior probability

    Arguments:
        main_series {pandas.DataFrame.Series} -- Series for filling gaps in it

    Keyword Arguments:
        sub_series {pandas.DataFrame.Series} -- Auxiliary series which has correlation 
        with first one (default: {pd.Series([])})

    Returns:
        pandas.DataFrame.Series -- main_series with filled gaps
    """
    # Подсчёт пропорции значений в главной колонке
    main_prop = main_series.value_counts(normalize=
                                                True).sort_values(ascending = False)

    # Начальные условия для составления таблицы ниже
    dict = {}
    start_val = 0
    end_val = 0

    # Составление таблицы разбиения диапазона (0, 1) на доли-диапазоны, соответствующие уникальным 
    # значениям главной колонки. Ниже, при попадании случайного числа в определённый диапазон
    # выводится уникальной значение главной колонки, соответствующее этому диапазону.
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
        # Получение списка пропущенных значений главного столбца
        main_nan = main_series[main_series.isna()]

        # Получение списка для заполнения на основе таблицы разбиения на доли (df_main_prop)
        fill_list = []
        for i in range(len(main_nan)):
            rnd = random()
            fill_list.append(df_main_prop[(df_main_prop[0] < rnd) & (df_main_prop[1] > rnd)].index[0])        
        
        # Заполнение пропусков
        main_series[main_series.isna()] = fill_list

        return main_series

    # Когда имеется дополнительный Series для "умного" заполнения пробелов основного .
    else:
        # Вычислим УСЛОВНУЮ вероятность, то есть вероятность выпадения уникального 
        # значения вспомогательной колонки при выпадении значения главной колонки
            # Подсчёт частоты встречаемости уникальных значений главного столбца
        main_count = main_series.value_counts().sort_values(ascending=False).reset_index()
        main_count.columns=['main','main_count']

            # Объединение заданных колонок в один датафрейм
        serv_df = pd.DataFrame({'main': main_series, 'sub': sub_series})

            # Распределение уникальных значений дополнительной колонки по уникальным 
            # значениям основной колонки. 
            # Каждая строка - уникальная пара значений основной и вспомогательной колонок
        distr = pd.DataFrame(serv_df.groupby(['main'])['sub'].value_counts())
        distr.columns = ['sub_count']
        distr.reset_index(inplace=True)

            # Получение столбца условных вероятностей (объяснение выше)
        distr = distr.merge(main_count, on='main', how='left')
        distr['cond_prop'] = distr.sub_count/distr.main_count

        # Добавление пропорций главного столбца, вычисленных ранее
        main_prop = main_prop.reset_index()
        main_prop.columns=['main','main_prop'] 
        distr = distr.merge(main_prop, on='main', how='left')

        # Вычислим полную вероятность наступления события (выпадения конкретного 
        # значения вспомогательной колонки) и добавим в общую таблицу
        event_prop = pd.DataFrame()
        for sub_val in distr['sub'].unique():
            event_prop[sub_val] = [(distr[distr['sub'] == sub_val]['main_prop']* \
                                distr[distr['sub'] == sub_val]['cond_prop']).sum()]
        
        event_prop = event_prop.T
        event_prop.reset_index(inplace=True)
        event_prop.columns=['sub','event_prop']

        distr = distr.merge(event_prop, on='sub', how='left')

        # Вычислим апостериорную вероятность, то есть вероятность выпадения конкретного значения 
        # основной колонки при выпадении уникального значения вспомогательной колонки
        distr['apost_prop'] = distr['main_prop']*distr['cond_prop']/distr['event_prop']

        # Получение строк, соответствующих пустым значениям главного столбца
        serv_nan = serv_df[serv_df['main'].isna()]

        # Получение списка для заполнения
        fill_list = []
        for i in range(len(serv_nan)):
            # Блок кода ниже справедлив в случае, когда соответствующее значение 'sub' не None:
            if pd.notna(serv_nan.iloc[i]['sub']):
                # Извлечение из созданного выше датафрейма строк, 
                # соответствующих текущему значению дополнительной колонки
                # в рассматриваемой строке с пропуском
                part_distr = distr[distr['sub'] == serv_nan.iloc[i]['sub']]

                # Начальные условия 
                dict = {}
                start_val = 0
                end_val = 0
                # Составление таблицы разбиения диапазона (0, 1) на доли-диапазоны, соответствующие  уникальным 
                # значениям главной колонки при одном конкретном значении вспомогательной колонки. 
                # Ниже, при попадании случайного числа в определённый диапазон
                # выводится уникальной значение главной колонки, соответствующее этому диапазону.
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
                # Получение значения, соответствующего доле-диапазону уникальных значений главной колонки, 
                # в которую попало случайное число
                rnd = random()
                fill_list.append(df_main_prop[(df_main_prop[0] < rnd) & (df_main_prop[1] > rnd)].index[0])        
            
        # Заполнение пропусков
        main_series[main_series.isna()] = fill_list

        return main_series

    return 0

# Функция определения p_value гипотезы о разности выборочных пропорций
def get_prop_dif(res_df):
    """ The function for determining the p_value of the hypothesis of the difference in selective proportions

    Arguments:
        res_df {pandas.DataFrame} -- Table of selective proportions

    Returns:
        float -- P_value of the hypothesis of the difference in selective proportions
    """
    # Оценочная пропорция
    p = res_df['loc_count'].sum()/res_df['sub_count'].sum()
    
    # Подсчёт z-статистики
    denom = math.sqrt(p*(1-p)*(1/res_df['sub_count'].iloc[0] + 1/res_df['sub_count'].iloc[1]))
    if (denom != 0):
        z_st = (res_df['prop'].iloc[0] - res_df['prop'].iloc[1])/denom
    else:
        return 1
    
    # Определение p_value
    if z_st > 0:
        p_value = 1 - norm.cdf(z_st)
    else:
        p_value = norm.cdf(z_st)

    return p_value

# Функция определения статистически значимой зависимости между двумя колонками датафрейма
def get_col_depend(df, column, sub_column, alpha=0.1, logs=False):
    """The function for determining statistically significant difference 
    between column and sub_column

    Arguments:
        df {pandas.DataFrame} -- DataFrame with columns for determining 
        statistically significant difference
        column {str} -- Main column
        sub_column {str} -- Additional column for finding level of dependence 
        on main column

    Keyword Arguments:
        alpha {float} -- Level of significance (default: {0.1})
        logs {boolean} -- True if it's necessary to recieve log-data

    Returns:
        dictionary -- dictionary of answers with column name, boolean of dependence, 
        relative amount of difference, max logarifm of difference, 
        logarifm of mean value of difference, [Logs]
        
    """
    
    # Ответ в виде словаря с параметрами: наименование колонки, 
    # имеется ли статистически значимая взаимосвязь с изучаемой колонкой,
    # доля пар колонок с расхождением от общего числа комбинаций, 
    # максимальное значение логарифма расхождения, 
    # логарифм среднего значения расхождения 
    answer_dict = {
        'col_name': sub_column,
        'depends': False,
        'rel_amount': 0,
        'max_log_frac': 0,
        'log_mean_frac': 0,
        }
    # Если требуются логи, выводить их 
    if logs:
        answer_dict['logs'] = []

    # Список p_value
    p_val_list = []
    # Список величин относительных расхождений (alpha/p_value)
    diff_list = []
    # Лист длинн всех комбинаций
    all_comb_count_lt = []


    # Датафрейм с подсчитанным распределением значений вспомогательного параметра (колонки) 
    # по значениям основного параметра (индексы) (нужен для оценки размера выборки)
    compare_df = pd.crosstab(df[column],df[sub_column])

    # Датафрейм аналогичный предыдущему, но в долях от количества значений в колонке
    research_df = pd.crosstab(df[column],df[sub_column],normalize='columns')

    # Список проверяемых значений изучаемой колонки
    index_list = list(research_df.index)
    for index in index_list: 
        
        # Оставим значения, соответствующие только рассматриваемому index изучаемой колонки:
        resrch_df = research_df.loc[index]
        comp_df = compare_df.loc[index]

        # Проверка на размер выборки.
        # Для пропорций проверяется (n*p) > 5 & (n*(1-p)) > 5. По факту, проверяется, чтобы количество значений, 
        # приходящихся на пересечение "строка-столбец" compare_df (датафрейма с подсчитанным распределением значений 
        # вспомогательного параметра (колонки) по значениям основного параметра (индексы)) было больше 5.
        
        # С учётом этого допущения:
        comp_val = 5 # Значение, с которым ведётся сравнение
        # Изучаемая строка подлежит оценке, если количество колонок с большим размером выборки > 50 %
        bool_ser = pd.Series(comp_df > comp_val)
        bool_count = bool_ser.value_counts(normalize=True)
        
        # Если вообще есть значение True
        if (True in bool_count.index):
            big_range = bool_count[True] > 0.5
        else: 
            big_range = False

        # Если изучаемая строка подлежит оценке
        if big_range:
            # Составляем датафрейм с индексами - значениями колонок compare_df (то есть 
            # значениями вспомогательного параметра) и колонками - "loc_count", "prop" и "sub_count"
            reslt_df = pd.DataFrame({'loc_count': comp_df, 'prop': resrch_df, 'sub_count': compare_df.sum()})

            # Составим комбинации пар значений колонок и пройдёмся по ним в цикле
            combinations_all = list(combinations(reslt_df.index,2))
            all_comb_count_lt.append(len(combinations_all))
            
            for comb in combinations_all:    
                p_value = get_prop_dif(reslt_df.loc[list(comb)]) # Определение p_value - для рассматриваемой пары
                p_val_list.append(p_value)

                # Если разница в выборочных пропорциях оказалась статистически значимой
                if p_value <= alpha: # Поправка Бонферони на множественную проверку гипотез была исключена, 
                #так как требуется получить наибольшее количество пар значений вспомогательной колонки, 
                #имеющих статистически значимое различие. Это допущение компенсируется тем, что отбор наиболее 
                #сильно связанной колонки идёт по максимальному расхождению
                    # Логарифмическая характеристика расхождения
                    diff = alpha/p_value
                    diff_list.append(diff)

                    #Вывод логов
                    if logs:
                        log_diff = round(math.log10(alpha/p_value), 3)
                        answer_dict['logs'].append(f"{column}/{index:5}\t{sub_column}/{list(comb)}\tОБНАРУЖЕНО статистически значимое различие\tlog10(alpha/p_value) = {log_diff}")          
                elif logs:
                    answer_dict['logs'].append(f"{column}/{index:5}\t{sub_column}/{list(comb)}\tстатистически значимое различие НЕ ОБНАРУЖЕНО\tp_value = {round(p_value*100, 2)}%")
        elif logs:
            answer_dict['logs'].append(f'{column}/{index:5}\tТестирование НЕВОЗМОЖНО, выборка мала.')
            # Вывод бинарного датафрейма
            answer_dict['logs'].append(bool_ser)

    if len(diff_list) > 0:
        # Имеется зависимость
        answer_dict['depends'] = True
        # Максимальное значение логарифма расхождения
        answer_dict['max_log_frac'] = round(math.log10(max(diff_list)), 3)
        # Логарифм среднего значения расхождения 
        answer_dict['log_mean_frac'] = round(math.log10(sum(diff_list)/len(diff_list)), 3)
    
        # Отношение числа статистически значимых различий в комбинациях пар колонок (значений вспомогательного параметра) к 
        # общему числу возможных комбинаций.
        if len(research_df.index) > 0:
            answer_dict['rel_amount'] = round(len(diff_list)/(max(all_comb_count_lt)*len(research_df.index)), 4) 

    return answer_dict

# Функция поиска связанных колонок
def find_depends_col(df, column, alpha=0.1, logs=False):
    # Получение списка колонок для анализа (c удалением изучаемой колонки)
    col_list = list(df.columns)
    col_list.remove(column)

    # Список всех оценённых колонок
    all_col = []
    # Анализ связи с каждой колонки
    for sub_column in col_list:
        all_col.append(get_col_depend(df, column, sub_column, alpha, logs))

    df = pd.DataFrame(all_col)
    # Вывод отсортированного датафрейма
    return df.sort_values(['log_mean_frac'], ascending=False)   

# Функция поиска простых делителей числа 
def dividers(val):
    ans_lt = []
    divider = 2 # Начинаем с первого простого числа
    # Делители числа ищем до тех пор, пока не дойдём до корня из val
    while divider**2 <= val: 
        if val % divider == 0: 
            ans_lt.append(divider)
            val = val // divider
        else:
            divider += 1
    # Если исследуемое число больше 1, оно тоже считается делителем числа
    if val > 1:
        ans_lt.append(val)
    return sorted(ans_lt, reverse=False)

# Функция поиска всех возможных делителей числа 
def all_dividers(val):
    ans_lt = []
    if val > 1:
        ans_lt.append(val)
        divider = val // 2 # Начинаем с половины значения
        # Делители числа ищем до тех пор, пока не дойдём до 2
        while divider >= 2: 
            if val % divider == 0: 
                ans_lt.append(divider)
            divider -= 1
    else:
        return 1

    return sorted(ans_lt, reverse=False)

# Функция превращения числовой колонки в категориальную
def change_num_to_count(df, column, bins_val): # num_div - порядковый номер делителя
    # Изучаемая серия
    series = df[column]

    if bins_val > 1:
        intreval_ser = series.value_counts(bins=bins_val)
        intervals = intreval_ser.index.values

        # Функция распределения значений на группы
        def div_groups(row):
            if pd.isna(row):
                return row
            # Создадим столбец с номерами групп, соответствующих разбиениям выше
            return intervals[intervals.contains(row)][0]

        return df[column].apply(div_groups)

    else:
        print('Малое число разбиений') 
        return 0

# Функция получения колонок со статистически значимым различием по goal_column 
def get_stat_dif(df, goal_column, column, large_val=30, alpha=0.01, top=0, eq_var=True): 
    # Если top > 0, берётся топ-список значений размерностью top. 
    # eq_var=True в случае равенства дисперсий обоих выборок
    answ = False
    p_val_list = []
    
    if top:
        cols = df[column].value_counts().index[:top] # возьмём топ- часто встречающихся значений
    else:
        cols = df[column].value_counts().index
    
    combinations_all = list(combinations(cols, 2)) # парные сочетания из значений
    for comb in combinations_all:
        ser_1 = df[df[column] == comb[0]][goal_column]
        ser_2 = df[df[column] == comb[1]][goal_column]

        if (len(ser_1) < large_val) or (len(ser_2) < large_val): # Если одна из выборок недостаточно 
            # велика, пропустить итерацию
            continue
        # Тест Стьюдента двухсторонней гипотезы
        p_val = ttest_ind(ser_1, ser_2, equal_var=eq_var).pvalue
        p_val_list.append(p_val)
        if p_val <= alpha/len(combinations_all): # Учёт поправки Бонферони на множественную проверку гипотез
            answ = True
            break
    if len(p_val_list):
        p_val = min(p_val_list)
    else:
        p_val = 1
    # Ответ в виде словаря из названия колонки, бинарной колонки наличия 
    # статистически значимого различия, уровня значимости, значения p_value
    return {'column': column,
            'differ': answ,
            'alpha': alpha,  
            'p_val': p_val}   