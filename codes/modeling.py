# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:11:35 2023

@author: gusta
"""


#Important to see if will be use a forecast horizon structure
#Needs a for loop to get horizons

#the Recursive method. It works by training a single model for one-step ahead forecasting.
# That is, predicting the next step. Then, the model is iterated with its previous forecasts 
#to get predictions for multiple steps

# Direct approach builds one model for each horizon. This approach avoids error propagation
# because there’s no need to iterate any model.
#But, there are some drawbacks as well. It requires more computational resources for the extra models. 
#Besides, it assumes that each horizon is independent. Often, this assumption will lead to poor results.

#DirectRecursive As the name implies, DirectRecursive attempts to merge the ideas of
# Direct and Recursive. A model is built for each horizon (following Direct). But, at each step, 
#the input data increases with the forecast of the previous model (following Recursive).
#This approach is known as chaining in the machine learning literature. 
#And scikit-learn provides an implementation for it with the RegressorChain class.

#Multi output The methods described so far are single-output approaches — They model one horizon at a time.
#This may be a limitation because they ignore the dependency among different horizons. 
#Capturing this dependency may be important for better multi-step ahead forecasts.
#This issue is solved by multi-output models. These fit a single model which learns all 
#the forecasting horizon jointly. Usually, learning algorithms need a single variable as the output. 
#This variable is known as the target variable. Yet, some algorithms can naturally take multiple
# output variables. In this case, we apply the method k-Nearest Neighbours. Other examples include 
#Ridge, Lasso, Neural Networks, or Random Forests (and the likes of it). The multi-output approach 
#has a variant which combines its ideas with Direct. This variant applied the Direct approach 
#in different subsets of the forecasting horizon. This method is described in reference.


from datetime import timedelta
import pandas as pd
import numpy as np
import itertools
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *
import json
import pickle
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm
import joblib
import matplotlib.pyplot as plt
from gluonts.mx import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.pandas import PandasDataset

import memory_aux
#Function to Create Hyperparameters Dictionary
def dict_hyperparameters():
    hyperparameters_dict = {
    
        'EN' : {
            'alpha': np.logspace(-4, 4, 100),    # Range of alpha values (regularization strength)
            'l1_ratio': np.linspace(0, 1, 100)
        },
        
        'RF' : {
        'n_estimators': np.arange(10, 200, 10),  # Vary the number of trees from 10 to 190
        'max_depth': [None] + list(np.arange(5, 31, 5)),  # Include None and depths from 5 to 30
        'min_samples_split': np.arange(2, 11),  # Vary min_samples_split from 2 to 10
        'min_samples_leaf': np.arange(1, 11),  # Vary min_samples_leaf from 1 to 10
        'max_features': ['sqrt', 'log2']  # Options for max_features
        },
        
        'XGB' : {
        'n_estimators': np.arange(100, 1000, 100),  # Vary the number of trees from 100 to 900
        'max_depth': np.arange(3, 11),  # Vary max_depth from 3 to 10
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],  # Options for learning_rate
        'subsample': [0.7, 0.8, 0.9, 1.0],  # Options for subsample
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # Options for colsample_bytree
        'gamma': [0, 1, 2, 3, 4],  # Options for gamma
        'reg_alpha': np.logspace(-4, 4, 100),  # Vary reg_alpha from 1e-4 to 1e4
        'reg_lambda': np.logspace(-4, 4, 100)  # Vary reg_lambda from 1e-4 to 1e4
        },
        
        'LGBM' : {
        'n_estimators': np.arange(100, 1000, 100),  # Vary the number of trees from 100 to 900
        'max_depth': np.arange(3, 11),  # Vary max_depth from 3 to 10
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],  # Options for learning_rate
        
        }
    }   
    print('Dictionary create')
    return  hyperparameters_dict
        
#Function to Define until when train and which date to predict
def ajust_data_horizon(df_key, date_column, round_date, horizon, data_freq):
    # print('Round_Date is when supposedly the model is being trained')
    # print('In other words, is the date where the data should be limit, because after that date \
    #       is considered future data')
    # print('So lets cut the dataset until the round Date')
    df_key = df_key[df_key[date_column] <= round_date]
    
    df_horizon = pd.DataFrame(  pd.date_range(start=round_date, periods=(horizon + 1),
                                              freq = data_freq), columns = [date_column])
    df_horizon = df_horizon.drop(df_horizon.index[0])
    
    return df_key, df_horizon

#Functio to create lagged features and decompose yearly seasonality, if enough data
def create_lags(df_key, df_horizon, key_column, date_column, target_column, lags, corr_limit, horizon):
    # print('Create Lags that make Senses and avoid any kind of leakage')
    #Major leakage problem is creating a exxtremely correlated column or bring future data to the past
    # print('Drop Columns that will not be used and order dataframe by Date ')
    df_key = df_key.drop([key_column], axis =1 ).sort_values(
        by = date_column, ascending = True)
    
    # print('Remove constant Feature')
    constant_columns = [col for col in df_key.columns if df_key[col].nunique() == 1]
    # print('Removing Cols:', constant_columns)
    df_key.drop(columns=constant_columns, inplace=True)
    # print('Creating column with Seasonality, Trend and Residual')
    try:
        result = sm.tsa.seasonal_decompose(df_key.set_index(date_column)[target_column], model='additive', period=52)  # 52 weeks in a year
    
        # Create new columns for seasonality, trend, and residuals
        df_key['Seasonality'] = result.seasonal.values
        df_key['Trend'] = result.trend.values
        df_key['Residual'] = result.resid.values
    except:
        print('Not Possible to get Seasonality, not enought data')
    
    #Drop cols from seasonal decompose that has 
    
    # print('Creating a copy to use latter as final dataframe with horizon to be predict aggregate')
    df_aux = df_key.copy()
    # print('The lags created take into acount the horizon that is predicting and the lags defined')
    # print(f'Creating {lags+horizon} lags to select best features and lagget features to fit our model')
    # print(f'The Correlation benchmark is {corr_limit} positive or negative')
    
    for col in [x for x in df_key.columns if date_column not in x]:
        for lag in range(horizon, lags+horizon):
            df_key[f'{col}_lag{lag}'] = df_key[f'{col}'].shift(lag)
    #This lag structure puts past data alongside with present data. For exemple,
    #takes the weekly sales from first week from Feb/2010 to the row of target week
    #sales at date second week Feb/2010
    # print(f'Drop first rows that now has NAN because lags made, losing first {lags} observations')
    df_key = df_key.iloc[lags+horizon-1:]
    #Esse drop ta errado
    corr = df_key.set_index(date_column).corr()[target_column]
    corr_filtered  = corr[(abs(corr) > corr_limit) & (abs(corr) < 0.9)] ##0.9 to limit to correlated columns
    features_corr_selected = list(corr_filtered.index)
    # print('Filtering final DataFrame with date, target columns plus most correlated columns with limit at 0.9')
    
    # print('Use copy made to append horizon to be predict and make features lags')
    df_aux = pd.concat([df_aux, df_horizon], axis=0, ignore_index=True)
    # print('dataframe Horizon has no data, will be only lagged features and blank value for target')
    for col in [x for x in df_aux.columns if date_column not in x]:
        for lag in range(1, lags+horizon):
            df_aux[f'{col}_lag{lag}'] = df_aux[f'{col}'].shift(lag)
    
    # print('Selecting only columns that were defined earlier')
    # print('In the row for the date to be predict, only lagged features can be displayed')
    features_corr_selected = [x for x in features_corr_selected if 'lag' in x]
    df_aux = df_aux[[date_column, target_column] + features_corr_selected]
    # df_aux = df_aux.dropna(subset=df_aux.columns.difference([target_column]), axis = 0)
    df_aux = df_aux.iloc[lags+horizon-1:]
    # print('Add constant')
    # print('Check if there is constant column in trainset')    
    constant_columns = [col for col in df_aux.columns if df_aux[col].nunique() == 1]
    # print('Removing Constant Cols:', constant_columns)
    df_aux.drop(columns=constant_columns, inplace=True)
    
    # print('Create Constant Column for Train and Test Sets')
    df_aux['constant'] = 1
    print('Final dataframe ready')
    # print('Another step could be calendar features')
    date_to_predict = df_aux[date_column].max()
    print(f'Date to Horizon {horizon} is {date_to_predict}')
    
    return df_aux, date_to_predict

#Function to Standardize data. It is more useful for linear models
def standard_scalling(X_train, X_test, y_train, y_test):
   
    # Store the original column names
    column_names_trainX = X_train.columns.tolist()
    index_trainX = X_train.index
    
    column_names_testX = X_test.columns.tolist()
    index_testX = X_test.index
    
    column_names_trainY = y_train.columns.tolist()
    index_trainY = y_train.index
    
    column_names_testY = y_test.columns.tolist()
    index_testY = y_test.index
    # Initialize the StandardScaler
    scalerX = StandardScaler()
    scalerY = StandardScaler()
    # Standardize the data
    scaled_trainX = scalerX.fit_transform(X_train)
    scaled_trainY = scalerY.fit_transform(y_train)
    
    scaled_testX = scalerX.transform(X_test)
    scaled_testY = scalerY.transform(y_test)
    # Create a new DataFrame with standardized values and original column names
    X_train = pd.DataFrame(data=scaled_trainX, columns=column_names_trainX, index = index_trainX)

    X_test = pd.DataFrame(data=scaled_testX, columns=column_names_testX, index = index_testX)
    
    y_train = pd.DataFrame(data=scaled_trainY, columns=column_names_trainY, index = index_trainY)
    
    y_test = pd.DataFrame(data=scaled_testY, columns=column_names_testY, index = index_testY)
    
    return X_train, X_test, y_train, y_test, scalerX, scalerY



def model_total_sales(df_agg_all, date_column, target_column, pred_column, round_date, horizon_range):
    
    train_data = df_agg_all[df_agg_all[date_column] <= round_date].set_index(date_column)
    test_data = df_agg_all[df_agg_all[date_column] > round_date]#.set_index(date_column)
    
  

    s =  52  # Seasonal orders (weekly seasonality)
    
    model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=s)
    results = model.fit()
    forecast = results.forecast(steps=horizon_range)
    forecast = forecast.reset_index().rename(columns = {'index':date_column, 0: pred_column})

    forecast = pd.merge(forecast, test_data, on=date_column, how='left')
    
    #Future Predictions ON FULL dataset
    model_future = ExponentialSmoothing(df_agg_all.set_index(date_column), seasonal='add', seasonal_periods=s)
    results_future = model_future.fit()
    forecast_future = results_future.forecast(steps=horizon_range)
    forecast_future = forecast_future.reset_index().rename(columns = {'index':date_column, 0: pred_column})
    
    return forecast, forecast_future


def future_forecast(df_aux, date_column, target_column, date_to_predict, round_date_pred, horizon, key_,
                        cv_iter, hyperparameters_dict, random_st, cv_option, models_path, hyper_path,
                        cv_score, cv_jobs, models, stantdardize_data):
   
    df_train = df_aux[df_aux[date_column] <= round_date_pred].set_index(date_column) #roud date
    df_test = df_aux[df_aux[date_column] == date_to_predict].set_index(date_column)
    
    X_train = df_train.loc[:, df_train.columns != target_column]
    y_train = df_train[[target_column]]
    
    X_test = df_test.loc[:, df_train.columns != target_column]
    y_test = df_test[[target_column]]
    
    #Standardize the data
    if stantdardize_data == True:
        
        X_train, X_test, y_train, y_test,scalerX, scalerY = standard_scalling(X_train, X_test, 
                                                                              y_train, y_test)
    
    #Two options for time series split. A Professor Hyndman approach and a scikit learn function
    

    if cv_option == 'Test_Size1_Gap_Horizon':
        tscv = TimeSeriesSplit(gap = (horizon - 1), max_train_size=None, n_splits=5, test_size = 1)
    else:
        tscv = TimeSeriesSplit( max_train_size=None, n_splits=5, test_size=(horizon)) 
   
    
    df_predictions = pd.DataFrame()
    
    df_predictions[date_column] = df_test.index
    df_predictions['Horizon'] = horizon
    df_predictions['Round_Date'] = pd.to_datetime( round_date_pred)
    df_predictions = df_predictions[['Round_Date', 'Horizon', date_column]]

    for model in models:
        print('Model', model)
        if model == 'LGBM':
            
            with open(hyper_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl', 'rb') as file:
                best_hyperparameters = pickle.load(file)
            
            new_model = lgb.LGBMRegressor(**best_hyperparameters)
            new_model.fit(X_train, y_train)
            
            y_pred = new_model.predict(X_test) 
            
            if stantdardize_data == True:
                y_pred = scalerY.inverse_transform(y_pred.reshape(-1, 1))
                
            df_predictions[f'Prediction_{model}'] = y_pred
            
        if model == 'XGB':
            
            with open(hyper_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl', 'rb') as file:
                best_hyperparameters = pickle.load(file)
            
            new_model = xgb.XGBRegressor(**best_hyperparameters)
            new_model.fit(X_train, y_train)
            
            y_pred = new_model.predict(X_test) 
            
            if stantdardize_data == True:
                y_pred = scalerY.inverse_transform(y_pred.reshape(-1, 1))
                
            df_predictions[f'Prediction_{model}'] = y_pred
            
        if model == 'RF':
            with open(hyper_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl', 'rb') as file:
                best_hyperparameters = pickle.load(file)
            
            new_model = RandomForestRegressor(**best_hyperparameters)
            new_model.fit(X_train, y_train)
            
            y_pred = new_model.predict(X_test) 
            
            if stantdardize_data == True:
                y_pred = scalerY.inverse_transform(y_pred.reshape(-1, 1))
                
            df_predictions[f'Prediction_{model}'] = y_pred
            
        if model == 'EN':
            with open(hyper_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl', 'rb') as file:
                best_hyperparameters = pickle.load(file)
            
            new_model = ElasticNet(**best_hyperparameters)
            new_model.fit(X_train, y_train)
            
            y_pred = new_model.predict(X_test) 
            
            if stantdardize_data == True:
                y_pred = scalerY.inverse_transform(y_pred.reshape(-1, 1))
                
            df_predictions[f'Prediction_{model}'] = y_pred
    
    return df_predictions

#Function to run the data through defined algorithms
def modeling_algorithms(df_aux, date_column, target_column, date_to_predict, round_date, horizon, key_,
                        cv_iter, hyperparameters_dict, random_st, cv_option, models_path, hyper_path,
                        cv_score, cv_jobs, models, stantdardize_data):

    df_train = df_aux[df_aux[date_column] <= round_date].set_index(date_column) #roud date
    df_test = df_aux[df_aux[date_column] == date_to_predict].set_index(date_column)
    
    X_train = df_train.loc[:, df_train.columns != target_column]
    y_train = df_train[[target_column]]
    
    X_test = df_test.loc[:, df_train.columns != target_column]
    y_test = df_test[[target_column]]
    
    #Standardize the data
    if stantdardize_data == True:
        
        X_train, X_test, y_train, y_test,scalerX, scalerY = standard_scalling(X_train, X_test, 
                                                                              y_train, y_test)
    
    #Two options for time series split. A Professor Hyndman approach and a scikit learn function
    

    if cv_option == 'Test_Size1_Gap_Horizon':
        tscv = TimeSeriesSplit(gap = (horizon - 1), max_train_size=None, n_splits=5, test_size = 1)
    else:
        tscv = TimeSeriesSplit( max_train_size=None, n_splits=5, test_size=(horizon)) 
   
    
    df_predictions = pd.DataFrame()
    
    df_predictions[date_column] = df_test.index
    df_predictions['Horizon'] = horizon
    df_predictions['Round_Date'] = pd.to_datetime( round_date)
    df_predictions = df_predictions[['Round_Date', 'Horizon', date_column]]

    for model in models:
        print('Model', model)
        if model == 'EN':
            elastic_net = ElasticNet(max_iter=10000, tol=0.001)
            
            
            # Create a RandomizedSearchCV object
            random_search = RandomizedSearchCV(
                elastic_net,          # Estimator
                param_distributions=hyperparameters_dict[model],  # Hyperparameter grid
                n_iter = cv_iter,           # Number of random combinations to try
                scoring= cv_score,  # Scoring metric (negated MSE for regression)
                cv = tscv, ###index_output,                 # Number of cross-validation folds
                random_state = random_st,      # Random state for reproducibility
                n_jobs = cv_jobs ,         # Use all available CPU cores for parallelization
                error_score='raise'          
            )
            
            start_time = time.time()

            
            # Fit the RandomizedSearchCV object to your data
            random_search.fit(X_train, y_train)  # Replace X_train and y_train with your data
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            n_iterations = random_search.n_splits_
            cv_results = random_search.cv_results_

            # Get the mean and standard deviation of the scores for each combination of hyperparameters
            # There will be scores equal the number of iterations defined
            
            
            mean_scores = cv_results['mean_test_score']
            score_mean = sum(mean_scores) / len(mean_scores)
            std_scores = cv_results['std_test_score']
            std_mean = sum(std_scores) / len(std_scores)
            # Get the best hyperparameters from the search
            best_params = random_search.best_params_
            
            # Create an ElasticNet regressor with the best hyperparameters
            best_elastic_net = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'])
        
            # Fit the regressor to your training data
            best_elastic_net.fit(X_train, y_train)
            joblib.dump(best_elastic_net,models_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl')
            with open(hyper_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl', 'wb') as file:
                pickle.dump(best_params, file)
            y_pred_en = best_elastic_net.predict(X_test)
            if stantdardize_data == True:
                y_pred_en = scalerY.inverse_transform(y_pred_en.reshape(-1, 1))
                
            df_predictions[f'Prediction_{model}'] = y_pred_en
            df_predictions[f'CV_Score_Mean_{model}'] = score_mean
            df_predictions[f'CV_StdDev_Mean_{model}'] = std_mean
            df_predictions[f'Tuning_Time_{model}'] = elapsed_time
            df_predictions[f'Iterations_Done_{model}'] = n_iterations 
        if model == 'XGB':

            xgb_reg = xgb.XGBRegressor()
            
            random_search = RandomizedSearchCV(
                xgb_reg,            
                param_distributions=hyperparameters_dict[model],  
                n_iter=cv_iter,        
                scoring = cv_score, 
                cv=tscv,              
                random_state=random_st,     
                n_jobs= cv_jobs,        
                error_score='raise'
            )
            
  
            start_time = time.time()

            
            # Fit the RandomizedSearchCV object to your data
            random_search.fit(X_train, y_train)  # Replace X_train and y_train with your data
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            n_iterations = random_search.n_splits_
            cv_results = random_search.cv_results_
            
            mean_scores = cv_results['mean_test_score']
            score_mean = sum(mean_scores) / len(mean_scores)
            std_scores = cv_results['std_test_score']
            std_mean = sum(std_scores) / len(std_scores)

            best_params = random_search.best_params_
            
            best_xgb_reg = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            gamma=best_params['gamma'],
            reg_alpha=best_params['reg_alpha'],
            reg_lambda=best_params['reg_lambda']
            )
            

            best_xgb_reg.fit(X_train, y_train)
            joblib.dump(best_xgb_reg,models_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl')
            with open(hyper_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl', 'wb') as file:
                pickle.dump(best_params, file)
            y_pred_xgb = best_xgb_reg.predict(X_test)
            if stantdardize_data == True:
                y_pred_xgb = scalerY.inverse_transform(y_pred_xgb.reshape(-1, 1))
            df_predictions[f'Prediction_{model}'] = y_pred_xgb
            df_predictions[f'CV_Score_Mean_{model}'] = score_mean
            df_predictions[f'CV_StdDev_Mean_{model}'] = std_mean
            df_predictions[f'Tuning_Time_{model}'] = elapsed_time
            df_predictions[f'Iterations_Done_{model}'] = n_iterations 
        if model == 'RF':
            
            rf_regressor = RandomForestRegressor()
            
            random_search = RandomizedSearchCV(
            rf_regressor,            
            param_distributions=hyperparameters_dict[model],  # Hyperparameter grid
            n_iter=cv_iter,         
            scoring = cv_score,  
            cv=tscv,                        
            n_jobs = cv_jobs,              
            random_state=random_st,    
            error_score='raise'
            )
            

            start_time = time.time()

            
            # Fit the RandomizedSearchCV object to your data
            random_search.fit(X_train,  np.ravel(y_train))  # Replace X_train and y_train with your data
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            n_iterations = random_search.n_splits_
            cv_results = random_search.cv_results_


            
            
            mean_scores = cv_results['mean_test_score'][~np.isnan( cv_results['mean_test_score'])]
            score_mean = sum(mean_scores) / len(mean_scores)
            std_scores = cv_results['std_test_score'][~np.isnan( cv_results['std_test_score'])]
            std_mean = sum(std_scores) / len(std_scores)
            # Get the best hyperparameters from the search
            best_params = random_search.best_params_
            
            # Create the best random forest regressor with the best hyperparameters
            best_rf_regressor = RandomForestRegressor(**best_params)
            
            # Fit the regressor to your training data
            best_rf_regressor.fit(X_train, np.ravel(y_train))
            joblib.dump(best_rf_regressor,models_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl')
            with open(hyper_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl', 'wb') as file:
                pickle.dump(best_params, file)
            # Make predictions on your test data
            y_pred_rf = best_rf_regressor.predict(X_test)
            if stantdardize_data == True:
                y_pred_rf = scalerY.inverse_transform(y_pred_rf.reshape(-1, 1))
            df_predictions[f'Prediction_{model}'] = y_pred_rf
            df_predictions[f'CV_Score_Mean_{model}'] = score_mean
            df_predictions[f'CV_StdDev_Mean_{model}'] = std_mean
            df_predictions[f'Tuning_Time_{model}'] = elapsed_time
            df_predictions[f'Iterations_Done_{model}'] = n_iterations 
        if model == 'LGBM':
            
            lgb_reg = lgb.LGBMRegressor() 
            
            random_search = RandomizedSearchCV(
            lgb_reg,            
            param_distributions=hyperparameters_dict[model],  
            n_iter=cv_iter,         
            scoring = cv_score,  
            cv=tscv,                        
            n_jobs = cv_jobs,              
            random_state=random_st,    
            error_score='raise'
            )
            

            start_time = time.time()

            
            # Fit the RandomizedSearchCV object to your data
            random_search.fit(X_train,  np.ravel(y_train))  # Replace X_train and y_train with your data

            end_time = time.time()
            elapsed_time = end_time - start_time
            n_iterations = random_search.n_splits_
            cv_results = random_search.cv_results_
            

            
            
            mean_scores = cv_results['mean_test_score'][~np.isnan( cv_results['mean_test_score'])]
            score_mean = sum(mean_scores) / len(mean_scores)
            std_scores = cv_results['std_test_score'][~np.isnan( cv_results['std_test_score'])]
            std_mean = sum(std_scores) / len(std_scores)
            # Get the best hyperparameters from the search
            best_params = random_search.best_params_
            
            # Create the best random forest regressor with the best hyperparameters
            best_lgbm_regressor = lgb.LGBMRegressor (**best_params)
            
            # Fit the regressor to your training data
            best_lgbm_regressor.fit(X_train, np.ravel(y_train))
            joblib.dump(best_lgbm_regressor,models_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl')
            with open(hyper_path + f'/{model}_model_Store_{key_}_Horizon_{horizon}.pkl', 'wb') as file:
                pickle.dump(best_params, file)
                
            # Make predictions on your test data
            y_pred_lgbm = best_lgbm_regressor.predict(X_test)
            if stantdardize_data == True:
                y_pred_lgbm = scalerY.inverse_transform(y_pred_lgbm.reshape(-1, 1))
            df_predictions[f'Prediction_{model}'] = y_pred_lgbm
            df_predictions[f'CV_Score_Mean_{model}'] = score_mean
            df_predictions[f'CV_StdDev_Mean_{model}'] = std_mean    
            df_predictions[f'Tuning_Time_{model}'] = elapsed_time
            df_predictions[f'Iterations_Done_{model}'] = n_iterations 
    return df_predictions

#Function to apply before forecast loop to filter if the key combination has proprer data to forecast.
def check_forecast_viable(df_key_, df_horizon, date_column, target_column, round_date, horizon):
    print('Measuring how far it is the last data point of the data key combination to the round date')
    frequency_points_distance = (pd.to_datetime( round_date) - df_key_[date_column].max()).days // 7
    total_point = len(df_key_)
    
    if df_key_.empty == True:
        print('Dataset not avaible to make predictions or averages, assume value zero')
        pred = 0
        
    elif frequency_points_distance > 8:
        print('Product probably discontinued, need more analysis but assume prediction zero ATM') #At The Moment
        pred = 0
        
    elif len(df_key_) < horizon:
        print(f'For Horizon {horizon} there is not enought data, assume average for forecast')
        pred = df_key_[target_column].mean()
        
    elif len(df_key_) <5:
        print('Check if it is possible to make forecast with dataset available for the dataset, do MA3')
        pred = df_key_.tail(3)[target_column].mean()
        
    elif len(pd.date_range(start=df_key_[date_column].min(), end=df_key_[date_column].max(), freq='W-FRI')) > total_point + 4:
        print('Date Range between start and end data is too big for dataset lenght. There are missing Data. Repeat last Value')
        pred = df_key_.tail(1)[target_column].mean()
        
    else:
        print('Apparentaly no problems with combinantion, set pred as string')
        pred = 'Go On'
    
    return pred

#Function define a pipeline where the steps for forecast key combination are organized and run
def run_forecast(df_complete, key_column, date_column, target_column, horizon_range, models_path, hyper_path, 
                 round_date,  data_freq, lags, corr_limit, cv_iter, hyperparameters_dict,
                 random_st,  cv_option, cv_score, cv_jobs, models, stantdardize_data,
                 action):
    
    start_time = time.time()
    list_keys_problem = []
    df_forecast = pd.DataFrame()
    for key_ in df_complete[key_column].unique():#[:10]:
        start_time_key = time.time()
        df_key = df_complete[df_complete[key_column] == key_]
        df_forecast_key = pd.DataFrame()
        for horizon in range(1,horizon_range + 1):
            print('key',key_ ,'horizon', horizon)
            if action == 'Train':
                df_key_, df_horizon = ajust_data_horizon(df_key, date_column, round_date, horizon, data_freq)
                df_aux, date_to_predict = create_lags(df_key_, df_horizon, key_column, date_column, target_column, lags, corr_limit, horizon)
            
                df_predictions = modeling_algorithms(df_aux, date_column, target_column, date_to_predict, 
                                                     round_date, horizon, key_,
                                        cv_iter, hyperparameters_dict, random_st, cv_option,  models_path,hyper_path,
                                        cv_score, cv_jobs, models, stantdardize_data)
            
            if action == 'Predict':
                df_key_, df_horizon = ajust_data_horizon(df_key, date_column, round_date, horizon, data_freq)
                df_aux, date_to_predict = create_lags(df_key_, df_horizon, key_column, date_column, target_column, lags, corr_limit, horizon)
            
                df_predictions = future_forecast(df_aux, date_column, target_column, date_to_predict, round_date, horizon, key_,
                                        cv_iter, hyperparameters_dict, random_st, cv_option, models_path,hyper_path,
                                        cv_score, cv_jobs, models, stantdardize_data)
                
            df_forecast_key = pd.concat([df_forecast_key, df_predictions], axis=0)

            df_forecast_key[key_column] = key_
        df_forecast = pd.concat([df_forecast, df_forecast_key], axis=0)
        end_time_key = time.time()
        elapsed_time_key = end_time_key - start_time_key
        print(f"Elapsed time For Key {key_}: {elapsed_time_key:.2f} seconds")
    df_forecast_final = pd.merge(df_forecast, df_complete[[date_column, key_column, target_column]],
                                  on = [date_column, key_column], how = 'left')   
    
    #Rearrange Columns
    predictions_columns = [x for x in df_forecast_final if 'Pred' in x]
    cv_score_columns = [x for x in df_forecast_final if 'CV_Score' in x]
    cv_std_columns = [x for x in df_forecast_final if 'CV_Std' in x]
    Tunning_Time_columns = [x for x in df_forecast_final if 'Time' in x]
    Iterations_columns = [x for x in df_forecast_final if 'Iterations' in x]
    df_forecast_final = df_forecast_final[['Round_Date', 'Horizon', date_column, key_column, target_column] +
                                           predictions_columns + 
                                           cv_score_columns +
                                           cv_std_columns + 
                                           Tunning_Time_columns +
                                           Iterations_columns]
    
    end_time = time.time()


    elapsed_time = end_time - start_time
    print(f"Elapsed time For Store Forecast: {elapsed_time:.2f} seconds")
    print('Forecast Done')

    return df_forecast_final



def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def deepAR_MeltDF(train, valid, target_column, key_column, date_column, data_freq, prediction_length,
                  layers, cells, drop_rate, epochs, num_workers, lower_bound, higher_bound):
    
    start_time = time.time()
    print('Set data format')
    train_ds = PandasDataset.from_long_dataframe(train, target=target_column, item_id=key_column, 
                                           timestamp=date_column, freq=data_freq)
    print('Set Estimator')
    estimator = DeepAREstimator(freq=data_freq, prediction_length=prediction_length,
                                num_cells = cells, 
                                num_layers=layers, dropout_rate = drop_rate,
                                trainer=Trainer(#ctx = mx.context.gpu(),
                                                epochs=epochs, #learning_rate = learning_rate,
                                                #callbacks=[scheduler]
                                                ))
    print('Train Estimator')
    predictor = estimator.train(train_ds, num_workers=num_workers)
    print('Predicting Data not seen yet but using past data used on training')
    pred = list(predictor.predict(train_ds))
    
    all_preds = list()
    for item in pred:
        key = item.item_id
        p = item.samples.mean(axis=0)
        p_lower = np.percentile(item.samples, lower_bound, axis=0)
        p_higher = np.percentile(item.samples, higher_bound, axis=0)
        dates = pd.date_range(start=item.start_date.to_timestamp(), periods=len(p), freq = data_freq)
        family_pred = pd.DataFrame({date_column: dates, key_column: key, 'pred': p, 
                                    f'p_{lower_bound}': p_lower, f'p_{higher_bound}': p_higher})
        all_preds += [family_pred]
    all_preds = pd.concat(all_preds, ignore_index=True)
    
    all_preds = all_preds.set_index([date_column, key_column])
    all_preds[all_preds < 0] = 0
    all_preds = all_preds.reset_index()
    
    all_preds = all_preds.merge(valid, on=[date_column, key_column], how='left')
    all_preds['metric_Acc'] = 1 - (abs(all_preds[target_column] - all_preds['pred']
                                       )/all_preds[target_column])
    wmape_metric = wmape(all_preds[target_column], all_preds['pred'])
    
    print('Now predict using the valid set, that is the latest avaible data, to predict future')
    print('Predicting Data not realized yet')
    
    print('Set format')
    valid_ds = PandasDataset.from_long_dataframe(valid, target=target_column, item_id=key_column, 
                                           timestamp=date_column, freq=data_freq)
    pred_future = list(predictor.predict(valid_ds))
    
    all_preds_future = list()
    for item in pred_future:
        key = item.item_id
        p = item.samples.mean(axis=0)
        p_lower = np.percentile(item.samples, lower_bound, axis=0)
        p_higher = np.percentile(item.samples, higher_bound, axis=0)
        dates = pd.date_range(start=item.start_date.to_timestamp(), periods=len(p), freq = data_freq)
        family_pred = pd.DataFrame({date_column: dates, key_column: key, 'pred': p, 
                                    f'p_{lower_bound}': p_lower, f'p_{higher_bound}': p_higher})
        all_preds_future += [family_pred]
    all_preds_future = pd.concat(all_preds_future, ignore_index=True)
    
    all_preds_future = all_preds_future.set_index([date_column, key_column])
    all_preds_future[all_preds_future < 0] = 0
    all_preds_future = all_preds_future.reset_index()
    
    all_preds = memory_aux.reduce_mem_usage(all_preds)
    all_preds_future = memory_aux.reduce_mem_usage(all_preds_future)
    end_time = time.time()
    
    # all_preds = all_preds.rename(columns = {'pred':'Predictions'})

    elapsed_time = end_time - start_time
    print(f"Elapsed time For all Forecast: {elapsed_time:.2f} seconds")
    return all_preds, wmape_metric, all_preds_future


def group_lower_level_forecast(all_preds, all_preds_future,
                               date_column, key_column, higher_level, target_column):
    
    all_preds[higher_level] = all_preds[key_column].str.split('_').str[0]
    pred_cols = [x for x in all_preds.columns if 'pred' in x or 'p_' in x]
    all_preds_agg = all_preds.groupby([date_column, higher_level]).sum()
    all_preds_agg = all_preds_agg[pred_cols + [target_column]]
    all_preds_agg.reset_index(inplace = True)
    
    all_preds_future[higher_level] = all_preds_future[key_column].str.split('_').str[0]
    pred_cols_fut = [x for x in all_preds_future.columns if 'pred' in x or 'p_' in x]
    all_preds_future_agg = all_preds_future.groupby([date_column, higher_level]).sum()
    all_preds_future_agg = all_preds_future_agg[pred_cols_fut]
    all_preds_future_agg.reset_index(inplace = True)
    
    return all_preds_agg, all_preds_future_agg

def get_all_forecast(df_lower_preds, df_final, df_forecast_total, date_column, id_column, pred_column):
    #Get only date, id column and prediction
    #also create sum matrix
    ####CREATE FOrecast sum total
    df_final[id_column] = 'Store_' + df_final['Store'].astype(str)    
    df_lower_preds[id_column] = 'Store_Dept_' + df_lower_preds['key_Store_Dept'].astype(str) 
    df_forecast_total[id_column] = 'Total' 
    df_forecast_total = df_forecast_total.reset_index()
    
    df_lower_preds = df_lower_preds[[date_column, id_column, pred_column]]
    df_final = df_final[[date_column, id_column, pred_column]]
    df_forecast_total = df_forecast_total[[date_column, id_column, pred_column]]
    #Ajust key names

    df_all = pd.concat([df_final, df_lower_preds, df_forecast_total], axis=0)
    
    
    return 'done'

def forecast_reconciliation():
    #For the package we need make a dataframe with all time series stack and with
    #a column id to identify the time series key and which hierarchy level it is.
    #So it will forecast all of it
    
    
    return 'done'


if __name__ == '__main__':
    
    
    print('teste')