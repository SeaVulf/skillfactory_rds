
# %%

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
def get_col_depend(df, column, sub_column, alpha=0.1):
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

    Returns:
        list -- List of answers with [Column Name, [Max difference, Relative amount of difference], Text of answer, [Test log]]
        difference between p_value and alpha and log.
    """
    
    #Список p_value
    p_val_list = []
    #Список величин расхождений
    diff_list = []

    #Лист с ответами
    # Наименование колонки, [Максимальное значение расхождения, Доля пар колонок с расхождением 
    # от общего числа значений вспомогательного параметра], Текстовый ответ, [Лог теста] 
    answer_list = ['',['', ''],''] 

    # Датафрейм с подсчитанным распределением значений вспомогательного параметра (колонки) 
    # по значениям основного параметра (индексы), в долях от количества значений в колонке
    research_df = pd.crosstab(df[column],df[sub_column],normalize='columns')
    
    # Датафрейм с подсчитанным распределением значений вспомогательного параметра (колонки) 
    # по значениям основного параметра (индексы) (нужен для оценки размера выборки)
    compare_df = pd.crosstab(df[column],df[sub_column])

    # Список проверяемых значений изучаемой колонки
    index_list = list(research_df.index)
    for index in index_list: 
        
        # Оставим значения, соответствующие только рассматриваемому index изучаемой колонки:
        resrch_df = research_df.loc[index]
        comp_df = compare_df.loc[index]

        # Проверка на размер выборки.
        # Для пропорций проверяется (n*p) & (n*(1-p)) > 5. По факту, проверяется, чтобы количество значений, 
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
            for comb in combinations_all:    
                p_value = get_prop_dif(reslt_df.loc[list(comb)]) # Определение p_value - для рассматриваемой пары
                p_val_list.append(p_value)

                # Если разница в выборочных пропорциях оказалась статистически значимой
                if p_value <= alpha/len(combinations_all): # Поправка Бонферони на множественную проверку гипотез
                    # Логарифмическая характеристика расхождения
                    diff = str(round(math.log10(alpha/p_value),3))
                    answer_list.append(f"{column}/{index:5}\t{sub_column}/{list(comb)}\tОБНАРУЖЕНО статистически значимое различие\tlog10(alpha/p_value) = {diff}")
                    diff_list.append(diff)
                else:
                    answer_list.append(f"{column}/{index:5}\t{sub_column}/{list(comb)}\tстатистически значимое различие НЕ ОБНАРУЖЕНО\tp_value = {round(p_value*100, 2)}%")
        else:
            answer_list.append(f'{column}/{index:5}\tТестирование НЕВОЗМОЖНО, выборка мала.')
            # Вывод бинарного датафрейма
            answer_list.append(bool_ser)

    answer_list[0] = sub_column # Значение рассматриваемой колонки
    # Если есть разница в выборочных пропорциях
    if len(diff_list) > 0:
        # Текстовое сообщение
        answer_list[2] = f"{column} ~ {sub_column:20}\tОБНАРУЖЕНА статистически значимая связь\t\tlog10(alpha/p_value) = {max(diff_list)}"
        answer_list[1][0] = max(diff_list)
        # Отношение числа статистически значимых различий в комбинациях пар колонок (значений вспомогательного параметра) к 
        # числу значений вспомогательного параметра.
        if len(research_df.columns) > 0:
            answer_list[1][1] = round(len(diff_list)/len(research_df.columns), 3) 
        else:
            answer_list[1][1] = '0'
    else:
        if len(p_val_list) == 0:
            p_val_list = [1]
        answer_list[2] = f"{column} ~ {sub_column:20}\tНЕ ОБНАРУЖЕНА статистически значимая связь\t\tmin(p_value) = {round(min(p_val_list)*100,2)}%"
        answer_list[1][0] = '0'
        answer_list[1][1] = '0'
    return [answer_list[0], answer_list[1], answer_list[2], answer_list[3:]]

# Функция ознакомления с составом КАТЕГОРИАЛЬНОЙ/РАНГОВОЙ колонки
def get_cat_compos(df, column, plot=True, rot=0):
    # Размер выборки
    stud_count_total = len(df)
    
    # Имеются ли None-элементы или пробельные пропуски
    nulls_list = find_nulls(df[column])
    print('\n', nulls_list[0])

    # Оценим процентное содержание пропусков
    nulls_sum = nulls_list[1] + nulls_list[2]
    if nulls_sum > 0:
        print(f'Доля пропусков от объёма выборки: {round(nulls_sum/stud_count_total*100, 1)}%')

    # Оценим состав данной колонки
    prop_df = pd.DataFrame({'total': df[column].value_counts(), 
                        'proportion': df[column].value_counts(normalize=True)})

    if plot:
        #Построим распределение
        df[column].value_counts(normalize=True).plot(kind='bar',
                                                        grid= True,
                                                        title=column + " proportion",
                                                        legend=False) 

        plt.xticks(rotation=rot)
    
    return [['Состав колонки ' + column, prop_df], nulls_list]

# Функция ознакомления с составом ЧИСЛОВОЙ колонки
def get_num_compos(df, column, plot=True):
    answer_list = [['',''],['',''],'']
    
    # Размер выборки
    stud_count_total = len(df)
    
    # Имеются ли None-элементы или пробельные пропуски
    nulls_list = find_nulls(df[column])
    print('\n', nulls_list[0])
    answer_list[2] = nulls_list

    # Оценим процентное содержание пропусков
    nulls_sum = nulls_list[1] + nulls_list[2]
    if nulls_sum > 0:
        print(f'Доля пропусков от объёма выборки: {round(nulls_sum/stud_count_total*100, 1)}%')

    # Статистическое описание колонки
    answer_list[0][0] = 'Статистические характеристики колонки ' + column
    answer_list[0][1] = pd.DataFrame(df[column].describe())

    # Определим границы выбросов, и выведем данные, соответствующие этому диапазону в гистограмму
    boards = get_boards(df[column])

    # Серия выбросов
    out_ser = df[(df[column] > boards[1])|
                            (df[column] < boards[0])][column]

    # Выведем значение и количество выбросов
    out_ct_df = pd.DataFrame({'outliers_count': out_ser.value_counts()})
    answer_list[1][1] = out_ct_df
    if len(out_ct_df) > 0:
        answer_list[1][0] = 'Значения выбросов и их количество'
    else:
        answer_list[1][0] = 'Выбросы не обнаружены'
    
    if plot:
        # Построение графиков в одном месте
        fig, axes = plt.subplots(1,2, figsize=(16,6))
        # Построение boxplot
        sns.boxplot(df[column],ax=axes[0])
        axes[0].set_title(column+' boxplot') 

        # Построение распределение признака
        
        # Датафрейм, содержащий значения в границах выбросов
        iqr_ser = df[df[column].between(boards[0], boards[1])][column]
        # Диапазон фактических значений датафрейма
        board_iqr = iqr_ser.value_counts().index.max() - iqr_ser.value_counts().index.min()
    
        axes[1].set_title(column + ' hist')
    
        # Построение гистограммы значений без выбросов
        axes[1].hist(iqr_ser, bins=board_iqr, label='IQR')

        if len(out_ct_df) > 0:
            axes[1].hist(out_ser, bins=len(out_ct_df), color='red', label='outliers') # Выбросы
            plt.legend()

    return answer_list

# Функция поиска связанных колонок
def find_depends_col(df, column, n_max=5): # n_max - количество колонок для вывода в датафрейме
    # Получение списка колонок для анализа (c удалением изучаемой колонки)
    col_list = list(df.columns)
    col_list.remove(column)

    # Вывод колонок, имеющих статистически значимую взаимосвязь
    all_col = [] # Вывод всех данных для дополнительной обработки
    col_depend = [] # Лист связанных колонок
    for sub_column in col_list:
        cur_res = get_col_depend(df, column, sub_column)
        all_col.append(cur_res)
        if float(cur_res[1][0]) > 0: # Если есть взаимосвязь
            col_depend.append(cur_res[2])
    print('\n'.join(col_depend))

    # Вывод колонок с наиболее сильной связью
    lt = [] # Список колонок, удовлетворяющих условию
    log_frac_lt = [] # Список значений log_frac

    # Получение списка значений
    for col in all_col:
        if float(col[1][0]) > 0:
            log_frac_lt.append(float(col[1][0]))
    log_frac_lt = sorted(log_frac_lt, reverse=True)[:n_max] # Сохранение топ-n значений

    # Вывод требуемых колонок
    for col in all_col:
        if float(col[1][0]) in log_frac_lt:
            lt.append([col[1][1], col[0], col[1][0]])
    df = pd.DataFrame(lt, columns=['rel_amount', 'col_name', 'log_frac'])
    print('\nКолонки с наиболее сильной взаимосвязью')
    
    return [df.sort_values(['log_frac'],ascending=False), all_col]    

# Функция оценки взаимосвязи между изучаемой и другими колонками
def find_col_relation(df, column, sub_column_list):
    answer_list = []
    for sub_column in sub_column_list:
        sub_answr_list = [['',''],'',['','']]
        # Рассмотрим распределение значений:
        sub_answr_list[0][0] = f'Распределение значений колонки {sub_column} по значениям изучаемой колонки {column}'
        sub_answr_list[0][1] = pd.crosstab(df[column], df[sub_column], normalize='columns')

        # Какое количество строк с пропусками в обеих колонках
        sub_answr_list[1] = 'Количество строк с пропусками в обеих колонках: ' + \
        str(len(df[(df[column].isna())&(df[sub_column].isna())]))

        # Оценим состав среди None в изучаемой колонки
        sub_answr_list[2][0] = f'Состав для None-значений изучаемой колонки {column} по значениям колонки {sub_column}'
        sub_answr_list[2][1] = pd.DataFrame(df[df[column].isna()][sub_column].value_counts(normalize=True))

        answer_list.append(sub_answr_list)

    return answer_list

# %%
