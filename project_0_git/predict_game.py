import numpy as np
import math

def game_core_v2(number):
    """The basic function of guessing the number by comparing it with the predicted

    Arguments:
        number {int} -- This is a number which is necessary to guess 
    """    
    count = 1
    predict = np.random.randint(1,101)
    while number != predict:
        count +=1
        if number > predict:
            predict +=1
        elif number < predict:
            predict -=1
    return(count)


def score_game_old(game_core):
    """Old function for starting the guessing game 1000 times and finding mean number of attempts.

    Arguments:
        game_core {object(function)} -- Function of guessing the number
    """    
    
    count_ls = []
    np.random.seed(1) #fixation of RANDOM SEED for experiment reproducibility 
    random_array = np.random.randint(1,101, size = (1000))
    for number in random_array:
        count_ls.append(game_core(number))
    score = int(np.mean(count_ls))
    print(f"Алгоритм автора угадывает число в среднем за {score} попыток")
    return(score)


def score_game(game_core):
    """My function for starting the guessing game 1000 times and finding mean number of attempts. 
    It gives the possibility to change range of the guessing number

    Arguments:
        game_core {object(function)} -- Function of guessing the number
    """    
    
    min_val=1
    max_val=101
    
    count_ls = []
    np.random.seed(1) #fixation of RANDOM SEED for experiment reproducibility 
    random_array = np.random.randint(min_val, max_val, size = (1000))
    
    for number in random_array:
        count_ls.append(game_core(number, min_val, max_val))
    score = int(np.mean(count_ls))
    print(f"Мой алгоритм угадывает число в среднем за {score} попыток")
    return(score)

def my_game_core(number, min_val, max_val):
    """My function of guessing the number which divide range on two

    Arguments:
        number {int} -- This is a number which is necessary to guess
        min_val {int} -- Minimum of guessing range
        max_val {int} -- Maximum of guessing range

    Returns:
        int -- Count of attempting
    """

    predict = max_val-min_val//2 #first number is mean of number's range
    count = 1

    while predict != number:
        if predict > number:
            max_val = predict            
            predict -= math.ceil((predict-min_val)/2) 
        
        elif predict < number:           
            min_val = predict           
            predict += math.ceil((max_val-predict)/2)
        else: 
            print('Невозможный вариант')
        count +=1
    return count
