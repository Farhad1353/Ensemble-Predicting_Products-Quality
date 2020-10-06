import sys
sys.path.append('..')
import numpy as np
import pandas as pd 
import sklearn.datasets
import matplotlib.pyplot as plt
from get_colors import colors

def plot_models(X, Y, title_name, colour = 'r'):
    plt.figure()
    plt.scatter(X, Y, c= colour)
    plt.title(title_name)
    plt.xlabel('Model number')
    plt.ylabel('Accuracy %')
    plt.show()

def hyper_2_param_array(param1_start, param1_end, param1_step, param2_start, param2_end, param2_step):
    '''
    hyper_2_param_array function creates a 2 dimension array 
    which the rows are representing each model to be trained and 
    the columns are representing each of the 2 HyperParameters 
    chosen for the training
    '''

    param1_total = int((param1_end + param1_step - param1_start-1)/param1_step)
    param2_total = int((param2_end + param2_step - param2_start-1)/param2_step)

   
    total_models = param1_total * param2_total
    param_array = np.ones((total_models, 2), dtype=int)
    param_array_idx = 0
    for param1_value in range(param1_start, param1_end, param1_step):
        for param2_value in range(param2_start, param2_end, param2_step):
            param_array[param_array_idx, 0] = int(param1_value)
            param_array[param_array_idx, 1] = int(param2_value)
            param_array_idx+=1         
    return param_array

def hyper_3_param_array(param1_start, param1_end, param1_step, \
                      param2_start, param2_end, param2_step, \
                      param3_start, param3_end, param3_step):
    '''
    hyper_3_param_array function creates a 2 dimension array 
    which the rows are representing each model to be trained and 
    the columns are representing each of the 3 HyperParameters 
    chosen for the training
    '''

    param1_total = int((param1_end + param1_step - param1_start-1)/param1_step)
    param2_total = int((param2_end + param2_step - param2_start-1)/param2_step)
    param3_total = int((param3_end + param3_step - param3_start-1)/param3_step)
   
    total_models = param1_total * param2_total * param3_total
    param_array = np.ones((total_models, 3), dtype=int)
    param_array_idx = 0
    for param1_value in range(param1_start, param1_end, param1_step):
        for param2_value in range(param2_start, param2_end, param2_step):
            for param3_value in range(param3_start, param3_end, param3_step):
                param_array[param_array_idx, 0] = int(param1_value)
                param_array[param_array_idx, 1] = int(param2_value)
                param_array[param_array_idx, 2] = int(param3_value)
                param_array_idx+=1         
    return param_array



def low_high_param(mid, step, param=3):
    '''
    low_high_param is a function which sets the lowest and
    highest figure that we want to tune our HyperParameters
    by getting the midpoint(which was the best HyperParameters)
    from previous stage.
    '''
    if param == 3:
        low = mid - step
        if mid <= step:
            low = mid
        high = mid + step + 1
    else:
        low = mid - (step * 2)
        high = mid + (step * 2) + 1
    return low, high

def show_features_impact(r, X_train, header):
    convert_coef = np.zeros((len(r),2))

    for i in range(len(r)):
        convert_coef[i][0] = round(r[i] * (np.max(X_train[:,i])-np.min(X_train[:,i])),1)
        header[i] = np.char.capitalize(header[i])
        if convert_coef[i][0] > 0 :
            convert_coef[i][1] = +1
        else:
            convert_coef[i][1] = -1
            convert_coef[i][0] *= -1
    header_feature = np.expand_dims(header[:-1], axis=1)
    convert_coef_big = np.append(header_feature,convert_coef, axis=1)
    convert_coef_big = convert_coef_big[convert_coef_big[:,1].argsort()][::-1]
    print("\n            Rank Highest impacting features ")
    print("          --------------------------------------")
    for i in range(len(r)):
        pos_neg = "positive"
        if convert_coef_big[i,2] == -1:
            pos_neg = "negative"
        print("%-20s" %convert_coef_big[i,0], " with weight of  ",convert_coef_big[i,1]," and ",pos_neg,"imapct")






    



