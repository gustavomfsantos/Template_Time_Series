# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 17:45:45 2023

@author: gusta
"""
print('Import Packages')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import os

print('Paths')
general_path = r'C:\Users\gusta\Desktop\Personal_Projects'
project_path = general_path + r'\Template_Time_Series'
data_path = project_path + r'\data'
codes_path = project_path + r'\codes'
final_path = project_path + r'\final_files'
models_path = project_path + r'\models_saved'
hyper_path = project_path + r'\hyperparameters_saved'


print('Import modules')
import data_prep
import explore_data
import modeling
import evaluation

print('Setting Configurations')
print('Dataset configurations')
target_column = 'Weekly_Sales'
date_column = 'Date'
higher_level = 'Store'
lower_level = 'Dept'
key_column = 'key_Store_Dept'
id_column = 'ID_column'
pred_column = 'Predictions'
data_freq = 'W-FRI'
data_seasonality = 52 #Weeks in a Year

print('Explore dataset configurations')
select_n = 1 #Number for highest volumes display on plots
corr_limit = 0.3 #Correlation minimum Absolute Value to consider for features and target and lag target
lags = 8 #lags for correlation analysis.

print('Prediction Configuration')
horizon_range = 8
prediction_length = horizon_range
context_length = prediction_length
round_date = pd.to_datetime('2012-08-31' ) #date to assume that is the last point of avaible data
round_date_pred = pd.to_datetime('2012-10-26' ) 

print('DeepAR configuration')
learning_rate = 0.001
patience = 20
cells = 32 
layers = 2
epochs = 30
drop_rate = 0.001
n_samples = 200
num_workers = 2
lower_bound = 20
higher_bound = 80

print('Machine learning algorithms and Cross validation configuration')
cv_iter = 50
random_st = 42
cv_option = 'Test_Size1_Gap_Horizon' #'Test_Size1_Gap_Horizon' ## 'Test_Size1_Gap_Horizon' or 'Alternative to other option
cv_score = 'neg_mean_squared_error'
cv_jobs = -1
stantdardize_data = False
models = [ 'LGBM'] # , 'XGB', 'RF', 'LGBM'

def explore_dataset():
    
    df_complete = data_prep.import_dataset(data_path, target_column, date_column, higher_level, lower_level,
                       key_column)

    print('Explore Target Data Agregate and get outliers dates')
    outliers_dates_agg = explore_data.check_all_data(df_complete, key_column, date_column, target_column)
    
    explore_data.run_histograms_plots_heatmaps(df_complete, key_column, date_column, target_column,
                                            select_n, lags, corr_limit, higher_level, lower_level)

    print('Explore Data Done, Output is general outlier')
    
    return outliers_dates_agg


def forecast_all_levels():
    
    print('Lower Level Forecast - DeepAR')
    df_complete = data_prep.import_dataset(data_path, target_column, date_column, higher_level, lower_level,
                       key_column)
    
    df, train, valid = data_prep.data_prep_deepAR(df_complete, round_date, date_column, key_column, target_column)    

    df_lower_preds, wmape_metric, df_lower_preds_future = modeling.deepAR_MeltDF(train, valid, target_column, 
                                    key_column, date_column, data_freq, prediction_length,
                      layers, cells, drop_rate, epochs, num_workers, lower_bound, higher_bound)  
    
    df_lower_preds = data_prep.change_pred_name(df_lower_preds)
    df_lower_preds_future = data_prep.change_pred_name(df_lower_preds_future)
    

    df_lower_preds.to_csv(os.path.join(final_path, 'Forecast_results_Lower_Level.csv'),decimal = ',', sep = ';', index = False)
    df_lower_preds_future.to_csv(os.path.join(final_path, 'Forecast_Future_Lower_Level.csv')) 
    
    
    print('Mid Level Forecast - LGBM')
    df_group_store =  data_prep.groupby_store(df_complete, higher_level, lower_level, 
                                        date_column, target_column, key_column)
        
    hyperparameters_dict = modeling.dict_hyperparameters()

    df_mid_level_pred = modeling.run_forecast(df_group_store, higher_level, date_column, 
                                                        target_column, horizon_range, models_path, hyper_path,
                      round_date, data_freq, lags, corr_limit, cv_iter, hyperparameters_dict,
                      random_st, cv_option, cv_score, cv_jobs, models, stantdardize_data, 'Train')
    
    df_mid_level_pred_fut = modeling.run_forecast(df_group_store, higher_level, date_column, 
                                                        target_column, horizon_range, models_path, hyper_path,
                      round_date_pred, data_freq, lags, corr_limit, cv_iter, hyperparameters_dict,
                      random_st, cv_option, cv_score, cv_jobs, models, stantdardize_data, 'Predict')
    
    df_mid_level_pred = data_prep.change_pred_name(df_mid_level_pred)
    df_mid_level_pred_fut = data_prep.change_pred_name(df_mid_level_pred_fut)
    
    df_mid_level_pred.to_csv(os.path.join(final_path, 'Forecast_results_Mid_Level.csv'),decimal = ',', sep = ';', index = False)
    df_mid_level_pred_fut.to_csv(os.path.join(final_path, 'Forecast_Future_Mid_Level.csv'),decimal = ',', sep = ';', index = False)

    print('Top Level Forecast - Exp Smothing')
    df_agg_all = data_prep.group_all(df_group_store, date_column, target_column)
   
    df_forecast_total, df_forecast_total_future = modeling.model_total_sales(df_agg_all, date_column, target_column, 
                                                   pred_column, data_seasonality, round_date, horizon_range) 
    
    df_forecast_total = data_prep.change_pred_name(df_forecast_total)
    df_forecast_total_future = data_prep.change_pred_name(df_forecast_total_future)
    
    df_forecast_total.to_csv(os.path.join(final_path, 'Forecast_results_Total.csv'),decimal = ',', sep = ';', index = False)
    df_forecast_total_future.to_csv(os.path.join(final_path, 'Forecast_Future_Total.csv'),decimal = ',', sep = ';', index = False)
    df_all_levels = modeling.set_data_togheter(df_complete, df_group_store, df_agg_all, date_column, higher_level, key_column, pred_column, id_column,
                          target_column)
    ###DO the reconciliantion
    df_reconcile, tags, df_reconcile_fut = modeling.get_all_forecast(df_lower_preds, df_lower_preds_future, df_mid_level_pred, df_mid_level_pred_fut, \
                         df_forecast_total, df_forecast_total_future, date_column, id_column, pred_column, key_column, higher_level,
                         lower_level, target_column, df_all_levels)
    df_reconcile.to_csv(os.path.join(final_path, 'Forecast_results_Reconcile.csv'),decimal = ',', sep = ';')   
    df_reconcile_fut.to_csv(os.path.join(final_path, 'Forecast_Future_Reconcile.csv'),decimal = ',', sep = ';') 
    
    df_reconcile_scores = evaluation.evaluate_reconcile(df_reconcile, target_column, tags)
    
    df_reconcile_scores.to_csv(os.path.join(final_path, 'Accuracy_Scores_Reconcile.csv'),decimal = ',', sep = ';') 
    
    return 'Forecast Done and Saved!'

# Reconciliation is the process of combining forecasts from different levels of the hierarchy in a
#  way that ensures consistency between the forecasts.




if __name__ == '__main__':
    
    
    print('Teste')
    
    
    # explore_dataset()
    
    forecast_all_levels()
    


    
    