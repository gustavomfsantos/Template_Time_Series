# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 17:45:45 2023

@author: gusta
"""


import pandas as pd
import numpy as np
import os
#get paths
general_path = r'C:\Users\gusta\Desktop\Personal_Projects'
project_path = general_path + r'\Template_Time_Series'
data_path = project_path + r'\data'
codes_path = project_path + r'\codes'
final_path = project_path + r'\final_files'
# get aux codes

os.chdir(codes_path)
# import memory_aux
import data_prep
import explore_data


# configs
target_column = 'Weekly_Sales'
date_column = 'Date'
key_column = 'key_Store_Dept'
data_freq = 'W-FRI'
select_n = 1 #Number for highest volumes display on plots
lags = 4 #lags for correlation analysis.
corr_limit = 0.3 #Correlation minimum Absolute Value to consider for features and target and lag target



if __name__ == '__main__':
    df_complete = data_prep.import_dataset(data_path)

    print('Explore Target Data Agregate and get outliers dates')
    outliers_dates_agg = explore_data.check_all_data(df_complete, key_column, date_column, target_column)
    
    print('Explore Target Data Agregate and Diff/Pct Change from target')
    explore_data.check_diff_pct(df_complete, key_column, date_column, target_column, lags,
                       corr_limit)
    
    print(f'Explore Target Data for {select_n} Stores with Highest Volumes')
    explore_data.check_high_vol_store(df_complete, key_column, date_column, target_column, select_n)
    
    print(f'Explore Target Data for {select_n} Keys Combination with Highest Volumes')
    explore_data.check_high_vol_key(df_complete, key_column, date_column, target_column, select_n)

    print('Get histogram for predictors features for Store')
    explore_data.features_by_store(df_complete, target_column, date_column, key_column, select_n)
    
    print('Explore Correlations between target and lagged target and also features inside key combination')
    explore_data.get_corr_with_features_and_lags_key(df_complete, key_column, date_column, target_column,
                                            select_n, lags, corr_limit)
    
    print('Explore Correlations between target and lagged target and also features for Store')
    explore_data.get_corr_with_features_and_lags_store(df_complete, key_column, date_column, target_column,
                                            select_n, lags, corr_limit)
    
    ##After results from this Exploratory Analysis is done, see if it fit any transformation
    # like normalization or standart scalling on the data.
    
    
    
    
    #Do regression for minor hierarchy level - Key Combination
    
    
    
    
    
    
    
    
    