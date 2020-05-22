
# %%

# Importing required libraries
import pandas as pd
import numpy as np
import math

from itertools import combinations
from scipy.stats import ttest_ind
from scipy.stats import norm
from random import random

#Тестовый датафрейм
#df = pd.DataFrame({'foo': [10,4,8, None, 6, 16, 4, 4, None],
#                  'bar': ['T', 'T', 'A', None, 'T', 'A', 'A', 'T', 'A']})
#df.head()

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

    # Когда имеется дополнительный Series для "умного" заполнения пробелов основного .
    else:
        # Вычислим условную вероятность
            # Подсчёт количества значений главного столбца
        main_count = main_series.value_counts().sort_values(ascending=False).reset_index()
        main_count.columns=['main','main_count']

            # Объединение заданных колонок в один датафрейм
        serv_df = pd.DataFrame({'main': main_series, 'sub': sub_series})

            # Распределение значений дополнительной колонки, 
            # соответствующих паре: "основное-дополнительное"
        distr = pd.DataFrame(serv_df.groupby(['main'])['sub'].value_counts())
        distr.columns = ['sub_count']
        distr.reset_index(inplace=True)

            # Получение столбца пропорций
        distr = distr.merge(main_count, on='main', how='left')
        distr['cond_prop'] = distr.sub_count/distr.main_count

        # Добавление пропорций главного столбца, вычисленных ранее
        main_prop = main_prop.reset_index()
        main_prop.columns=['main','main_prop'] 
        distr = distr.merge(main_prop, on='main', how='left')

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

        # Получение строк, соответствующих пустым значениям главного столбца
        serv_nan = serv_df[serv_df['main'].isna()]

        # Получение списка для заполнения
        fill_list = []
        for i in range(len(serv_nan)):
            # Блок кода ниже справедлив в случае, когда соответствующее значение 'sub' не None:
            if pd.notna(serv_nan.iloc[i]['sub']):
                # Извлечение из созданного выше датафрейма строк, 
                # соответствующих текущему значению в дополнительной колонке
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
def get_col_depend(df, column, sub_column, full_column, alpha=0.1):
    """The function for determining statistically significant difference 
    between column and sub_column

    Arguments:
        df {pandas.DataFrame} -- DataFrame with columns for determining 
        statistically significant difference
        column {str} -- Main column
        sub_column {str} -- Additional column for finding level of dependence 
        on main column
        full_column {str} -- Fully filled column 

    Keyword Arguments:
        alpha {float} -- Level of significance (default: {0.1})

    Returns:
        list -- List of answers with the result text, the value of the maximum 
        difference between p_value and alpha and log.
    """
    #Список величин расхождений
    diff_list = []
    #Лист с ответами
    answer_list = ['','','']

    # Составим датафрейм с распределением долей вспомогательного столбца 
    # по параметрам изучаемого столбца
    # Подсчёт размера выборок 
    sub_count_df = pd.DataFrame(df.groupby([sub_column])[column].count())
    sub_count_df.columns=['sub_count']

    # Создадим сравнительную таблицу и приведём к удобному для работы с ней виду
    research_df = df.pivot_table(values=[full_column], # Полностью заполненная колонка
                                    index=[column],
                                    columns=[sub_column],
                                    aggfunc='count')[full_column]
    research_df = research_df.T
    research_df = research_df.merge(sub_count_df, how='left',on=sub_column)

    # Список проверяемых значений изучаемой колонки
    index_list = list(df[column].value_counts().sort_values(ascending=False).index)

    for index in index_list: 
        # Оставим значения, соответствующие только index:
        res_df = research_df[[index, 'sub_count']]

        # Переименуем первый столбец и дополним датафрейм пропорциями
        res_df.rename(columns={index: 'loc_count'}, inplace=True)
        res_df['prop'] = res_df['loc_count']/res_df['sub_count']
        res_df['inv_prop'] = 1 - res_df['prop']

        # Проверка на размер выборки
        comp_val = 5 # Значение, с которым ведётся сравнение
        
        bool_df = pd.DataFrame([res_df['sub_count']*res_df['prop'] > comp_val, 
                                res_df['sub_count']*res_df['inv_prop'] > comp_val])
        bool_ser_1 = res_df['sub_count']*res_df['prop'] > comp_val
        bool_ser_2 = res_df['sub_count']*res_df['inv_prop'] > comp_val
        # Выборка будет считаться большой, если количество параметров sub_column 
        # с малой величиной выборки будет меньше 50%
        big_range = (len(list(bool_ser_1[bool_ser_1 == False]))/len(bool_df.T) < 0.5) & \
                    (len(list(bool_ser_2[bool_ser_2 == False]))/len(bool_df.T) < 0.5)

        # Если размер выборки большой
        if big_range:
            combinations_all = list(combinations(res_df.index,2))
            
            for comb in combinations_all:
                p_value = get_prop_dif(res_df.loc[list(comb)])
                if p_value <= alpha/len(combinations_all): # Поправка Бонферони на множественную проверку гипотез
                    diff = str(round(math.log10(alpha/p_value),3))
                    answer_list.append(f'"{index}") Колонка "{sub_column}"" - имеет статистически значимую взаимосвязь с изучаемой колонкой "{column}". Относительное расхождение: {diff}.')
                    diff_list.append(diff)
                    break
                else:
                    answer_list.append(f'"{index}") Статистически значимой взаимосвязи между заданными колонками не обнаружено.')
        else:
            answer_list.append(f'Тестирование по "{index}" невозможно, выборка мала.')
            answer_list.append(bool_df)
    answer_list[0] = sub_column
    if len(diff_list) > 0:
        answer_list[2] = f'Обнаружена статистически значимая связь между "{sub_column}" и "{column}",\t log10(alpha/p_value) = {max(diff_list)}'
        answer_list[1] = max(diff_list)
    else:
        answer_list[2] = f'Не выявлена статистически значимая связь между "{sub_column}" и "{column}".'
        answer_list[1] = '0'
    return [answer_list[0], answer_list[1], answer_list[2], answer_list[3:]]


