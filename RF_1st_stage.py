import sys                    # import libraries
sys.path.append('./Library')
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor # import sklearn libraries
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import accuracy_score

from utils import rgb2gray, get_regression_data, visualise_regression_data # import our libraries
from assigning_library import read_csv_files, split_train_validation, split_features_labels
from regression_library import plot_models, hyper_3param_array

#######################################################################################################################
## This is Part 1 of the project where we read the train and test csv files and splittig them to features and labels ##
#######################################################################################################################

path_train_file = "./Data/winequality-red-train.csv" # assigning the train.csv file path to a variable
path_test_file = "./Data/winequality-red-test.csv" # assigning the test.csv file path to a variable

my_train_File, my_test_File = read_csv_files(path_train_file, path_test_file) # reading the csv files using pandas
                                                                       
header, X_train, X_validation, Y_train, Y_validation\
 = split_train_validation(my_train_File, split_rate=0.25) # splitting the train.csv to train and validation arrays

X_test, Y_test = split_features_labels(my_test_File) # splitting the test.csv file to features and label arrays
                                                                                    
###################################################################################################################
############### This part is using RandomForest in Regression form(Part 2 of the Project) #########################
###################################################################################################################
best_regr_score_randomforest = 0 # variable which shows the best scoring RandomForest model
regr_score_randomforest = 0 # variable which shows the score for each of the Random Forest model

n_estimator_strarting_point = 100   # setting the range for n_estimators for different RandomForest models
n_estimator_ending_point = 500
n_estimator_step_size = 100  

max_features_starting_point = 2   # setting the range for max_features for different RandomForest models
max_features_ending_point = 10
max_features_step_size = 1    

max_depth_starting_point = 3    # setting the range for max_depth for different RandomForest models
max_depth_ending_point = 40
max_depth_step_size = 4

RF_hyper_param_array = \
    hyper_3param_array(n_estimator_strarting_point, n_estimator_ending_point, n_estimator_step_size, \
                       max_features_starting_point, max_features_ending_point, max_features_step_size, \
                       max_depth_starting_point, max_depth_ending_point, max_depth_step_size )

### training and testing(through validation sets) all the RandomForest models
for RF_idx in trange(len(RF_hyper_param_array)):
    # assigning an instance of a RandomForest
    randomforestregressor = RandomForestRegressor(n_estimators = int(RF_hyper_param_array[RF_idx,0]), \
                    max_features = RF_hyper_param_array[RF_idx,1], max_depth = RF_hyper_param_array[RF_idx,2])
    randomforestregressor.fit(X_train, Y_train)  # fitting the train arrays to each RandomForest model
    Y_prediction = np.around(randomforestregressor.predict(X_validation)) #predicting using the validation set  
    regr_score_randomforest = accuracy_score(Y_validation, Y_prediction) #accuaracy score for each model
    RF_hyper_param_array[RF_idx,3] = regr_score_randomforest * 100

max_index = np.argmax(RF_hyper_param_array[:,3])

os.system('cls')
print("\n               RandomForest performance ")
print("            --------------------------------")  # show the best model of RandomForest tested on validation set
print("Best Regressor Score for Random Forest : {:.2f}%".format(RF_hyper_param_array[max_index,3])) 
print("Best Estimator number : ", RF_hyper_param_array[max_index,0], "\nBest Features number : "\
      , RF_hyper_param_array[max_index,1], "\nbest_max_depth : ",RF_hyper_param_array[max_index,2])

randomforest_best_parameters = np.array([RF_hyper_param_array[max_index,0]\
                              , RF_hyper_param_array[max_index,1], RF_hyper_param_array[max_index,2]])

np.savetxt('./Data/RandomForest_reg.csv', randomforest_best_parameters, fmt="%d", delimiter=",")

input("Press Enter to continue...")

#plot_models(randomforest_models[:,0], randomforest_models[:,1], "RandomForest-Models", 'g') # Plot all the models

print("Best model has a score of ", round(RF_hyper_param_array[max_index,3],2), "%" )
