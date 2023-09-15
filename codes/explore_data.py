# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:12:57 2023

@author: gusta
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
 
from sklearn.preprocessing import StandardScaler

def detect_outliers_zscore(data, threshold=2):
    z_scores = (data - np.mean(data)) / np.std(data)
    outliers = np.abs(z_scores) > threshold
    return outliers

def check_diff_pct(df_complete, key_column, date_column, target_column, lags,
                   corr_limit):
    
    df = df_complete.groupby([date_column]).sum()[target_column].reset_index()
    df[f'{target_column}_diff'] = df[f'{target_column}'].diff()
    df[f'{target_column}_pct'] = df[f'{target_column}'].pct_change()
    
    # df.set_index(date_column, inplace = True)
    
    scaler = StandardScaler()
    scaled_data = pd.DataFrame( scaler.fit_transform(df.drop([date_column], axis = 1)))
    scaled_data.index = df[date_column]
    scaled_data.columns = df.drop([date_column], axis = 1).columns
    scaled_data.reset_index(inplace = True)
    print('Apply normalization')
    
    plt.figure(figsize=(10, 6))
    plt.plot(scaled_data[date_column], scaled_data[target_column], label='Original Data')
    plt.plot(scaled_data[date_column], scaled_data[f'{target_column}_diff'], label='Diff Variation')
    plt.plot(scaled_data[date_column], scaled_data[f'{target_column}_pct'], label='Pct Variation')
    plt.title('Aggregate Sales')
    plt.xlabel(date_column)
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    #Create Lags for Diff and Pct Columns in order to see if past data of this 
    #columns can affect the target in present date. 
    
    #But again, we lag the target 
    # to past instead to lag features ahead. The values will be 
    num_lags = lags
    for lag in range(1, num_lags+1):
        scaled_data[f'{target_column}_diff_{lag}'] = scaled_data[f'{target_column}_diff'].shift(lag)
        scaled_data[f'{target_column}_pct_{lag}'] = scaled_data[f'{target_column}_pct'].shift(lag)
        
        
    correlation_matrix = scaled_data.drop([date_column], axis = 1).corr()
    
    # Plot correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    plt.title('Correlation Heatmap For Target and Diff and Pct')
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.colorbar()
    plt.show()
    
    correlation_matrix_target = correlation_matrix[target_column]
    correlation_matrix_target = correlation_matrix_target[correlation_matrix_target.index != target_column]
    
    
    selected_indices = correlation_matrix_target[abs(correlation_matrix_target)>corr_limit]
    list_title = list(selected_indices.index)
    correlation_target_lag = correlation_matrix_target.loc[selected_indices.index].T
    if correlation_target_lag.empty:
        print(f'No Correlations over {corr_limit}')
    else:
        plt.figure(figsize=(8, 5))
        plt.plot( correlation_target_lag, marker='o')
        plt.title(f'Correlation of {list_title} with Target')
        # plt.xlabel('Lag', rotation = 45)
        plt.ylabel('Correlation')
        plt.xticks(range(0, len(correlation_target_lag.index)), rotation = 45)
        plt.grid()
        plt.show()
    
    return 'Plots with Diff and Pct from target Done'



def check_all_data(df_complete, key_column, date_column, target_column):

    df = df_complete.groupby([date_column]).sum()[target_column].reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df[date_column], df[target_column])
    plt.title('Time Series Plot Aggregate Sales')
    plt.xlabel(date_column)
    plt.ylabel(target_column)
    plt.show()
    
    # Check for stationarity
    result = adfuller(df[target_column])
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    
    # Plot rolling mean and std deviation
    rolling_mean = df[target_column].rolling(window=10).mean()
    rolling_std = df[target_column].rolling(window=10).std()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df[date_column], df[target_column], label='Original Data')
    plt.plot(df[date_column], rolling_mean, label='Rolling Mean')
    plt.plot(df[date_column], rolling_std, label='Rolling Std Deviation')
    plt.title('Rolling Mean and Standard Deviation Aggregate Sales')
    plt.xlabel(date_column)
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[target_column], bins=20, edgecolor='black')
    plt.title('Histogram of Data Aggregate Sales')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
    # Autocorrelation and Lag Analysis
    plt.figure(figsize=(12, 6))
    plot_acf(df[target_column], lags=30, ax=plt.subplot(121))
    plt.title('Autocorrelation Aggregate Sales')
    plot_pacf(df[target_column], lags=30, ax=plt.subplot(122))
    plt.title('Partial Autocorrelation Aggregate Sales')
    plt.tight_layout()
    plt.show()
    
    # Detect outliers
    outliers = detect_outliers_zscore(df[target_column])
    print('Outliers Dates')
    print((df.loc[outliers, date_column]))
    outliers_dates = (df.loc[outliers, date_column])
    plt.figure(figsize=(10, 6))
    plt.plot(df[date_column], df[target_column], label=target_column)
    plt.scatter(df.loc[outliers, date_column], df.loc[outliers, target_column], color='red', label='Outliers')
    plt.title('Outliers Detection Aggregate Sales')
    plt.xlabel(date_column)
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    print('Plots for All Data Agregate Done')
    return outliers_dates

    
def check_high_vol_store(df_complete, key_column, date_column, target_column, select_n, higher_level):
        
    print('Getting higher Volumes by Store')
    df_store_vol = df_complete.groupby([higher_level]).sum([target_column]).sort_values(
        by = target_column, ascending = False)
    list_higher_vols = list( df_store_vol.head(select_n).index)
    
    df_store_agg = df_complete.groupby([higher_level, date_column]).sum()[target_column].reset_index()
    print(f'Plotting for the {select_n} highest volumes')
    for store in list_higher_vols:
        print(higher_level,store)
        df = df_store_agg[df_store_agg[higher_level] == store]
        plt.figure(figsize=(10, 6))
        plt.plot(df[date_column], df[target_column])
        plt.title(f'Time Series Plot {store}')
        plt.xlabel(date_column)
        plt.ylabel(target_column)
        plt.show()
        
        # Check for stationarity
        result = adfuller(df[target_column])
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        
        # Plot rolling mean and std deviation
        rolling_mean = df[target_column].rolling(window=10).mean()
        rolling_std = df[target_column].rolling(window=10).std()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df[date_column], df[target_column], label='Original Data')
        plt.plot(df[date_column], rolling_mean, label='Rolling Mean')
        plt.plot(df[date_column], rolling_std, label='Rolling Std Deviation')
        plt.title(f'Rolling Mean and Standard Deviation {store}')
        plt.xlabel(date_column)
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        
        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df[target_column], bins=20, edgecolor='black')
        plt.title(f'Histogram of Data {store}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()
        
        # Autocorrelation and Lag Analysis
        plt.figure(figsize=(12, 6))
        plot_acf(df[target_column], lags=30, ax=plt.subplot(121))
        plt.title(f'Autocorrelation {store}')
        plot_pacf(df[target_column], lags=30, ax=plt.subplot(122))
        plt.title(f'Partial Autocorrelation {store}')
        plt.tight_layout()
        plt.show()
        
        outliers = detect_outliers_zscore(df[target_column])
        print('Outliers Dates')
        print((df.loc[outliers, date_column]))
        plt.figure(figsize=(10, 6))
        plt.plot(df[date_column], df[target_column], label=target_column)
        plt.scatter(df.loc[outliers, date_column], df.loc[outliers, target_column], color='red', label='Outliers')
        plt.title(f'Outliers Detection {store}')
        plt.xlabel(date_column)
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    return 'Plots for Store Aggregation Done'



   
def check_high_vol_key(df_complete, key_column, date_column, target_column, select_n):
    
    print('Getting higher Volumes by Key Combination')
    df_store_vol = df_complete.groupby([key_column]).sum([target_column]).sort_values(
        by = target_column, ascending = False)
    list_higher_vols = list( df_store_vol.head(select_n).index)
    
    df_aux = df_complete.groupby([key_column, date_column]).sum()[target_column].reset_index()
    print(f'Plotting for the {select_n} keys highest volumes')
    for key in list_higher_vols:
        print('Key',key)
        df = df_aux[df_aux[key_column] == key]
        plt.figure(figsize=(10, 6))
        plt.plot(df[date_column], df[target_column])
        plt.title(f'Time Series Plot {key}')
        plt.xlabel(date_column)
        plt.ylabel(target_column)
        plt.show()
        
        # Check for stationarity
        result = adfuller(df[target_column])
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        
        # Plot rolling mean and std deviation
        rolling_mean = df[target_column].rolling(window=10).mean()
        rolling_std = df[target_column].rolling(window=10).std()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df[date_column], df[target_column], label='Original Data')
        plt.plot(df[date_column], rolling_mean, label='Rolling Mean')
        plt.plot(df[date_column], rolling_std, label='Rolling Std Deviation')
        plt.title(f'Rolling Mean and Standard Deviation {key}')
        plt.xlabel(date_column)
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        
        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df[target_column], bins=20, edgecolor='black')
        plt.title(f'Histogram of Data {key}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()
        
        # Autocorrelation and Lag Analysis
        plt.figure(figsize=(12, 6))
        plot_acf(df[target_column], lags=30, ax=plt.subplot(121))
        plt.title(f'Autocorrelation {key}')
        plot_pacf(df[target_column], lags=30, ax=plt.subplot(122))
        plt.title(f'Partial Autocorrelation {key}')
        plt.tight_layout()
        plt.show()
        
        outliers = detect_outliers_zscore(df[target_column])
        print('Outliers Dates')
        print((df.loc[outliers, date_column]))
        plt.figure(figsize=(10, 6))
        plt.plot(df[date_column], df[target_column], label=target_column)
        plt.scatter(df.loc[outliers, date_column], df.loc[outliers, target_column], color='red', label='Outliers')
        plt.title(f'Outliers Detection {key}')
        plt.xlabel(date_column)
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    return 'Plots for Key combination Done'


def features_by_store(df_complete, target_column, date_column, key_column, select_n, higher_level,
                      lower_level):
    
     print('Getting higher Volumes by Key Combination')
     df_store_vol = df_complete.groupby([higher_level]).sum([target_column]).sort_values(
         by = target_column, ascending = False)
     list_higher_vols = list( df_store_vol.head(select_n).index)
     
     aggregation_dict = {target_column: 'sum'}
     for column in df_complete.columns:
         if column != target_column and column != date_column and column != higher_level \
             and column != key_column and column != lower_level:
             aggregation_dict[column] = 'mean'
     
     df_group_store = df_complete.groupby([higher_level, date_column]).agg(aggregation_dict).reset_index()
     for store in list_higher_vols:
         df = df_group_store[df_group_store[higher_level] == store]
         # df.set_index(date_column, inplace = True)
         df.drop([higher_level], axis = 1, inplace = True)
         #Size is constant
         cols_to_analyse = [x for x in df.columns if date_column not in x and 'Size' not in x]
         for col in cols_to_analyse:
             print(col)
     
             # Plot rolling mean and std deviation
             # rolling_mean = df[col].rolling(window=10).mean()
             # rolling_std = df[col].rolling(window=10).std()
             
             # plt.figure(figsize=(10, 6))
             # plt.plot(df[date_column], df[col], label='Original Data')
             # plt.plot(df[date_column], rolling_mean, label='Rolling Mean')
             # plt.plot(df[date_column], rolling_std, label='Rolling Std Deviation')
             # plt.title(f'Rolling Mean and Standard Deviation {col} for store {store}')
             # plt.xlabel(date_column)
             # plt.ylabel('Value')
             # plt.legend()
             # plt.show()
             
             # Histogram
             plt.figure(figsize=(10, 6))
             plt.hist(df[col], bins=20, edgecolor='black')
             plt.title(f'Histogram of Data {col} for store {store}')
             plt.xlabel('Value')
             plt.ylabel('Frequency')
             plt.show()
             
             # Autocorrelation and Lag Analysis
             # plt.figure(figsize=(12, 6))
             # plot_acf(df[col], lags=30, ax=plt.subplot(121))
             # plt.title(f'Autocorrelation {col} for store {store}')
             # plot_pacf(df[col], lags=30, ax=plt.subplot(122))
             # plt.title(f'Partial Autocorrelation {col} for store {store}')
             # plt.tight_layout()
             # plt.show()
             
             # outliers = detect_outliers_zscore(df[target_column])
             # print('Outliers Dates')
             # print((df.loc[outliers, date_column]))
             # plt.figure(figsize=(10, 6))
             # plt.plot(df[date_column], df[col], label=col)
             # plt.scatter(df.loc[outliers, date_column], df.loc[outliers, col], color='red', label='Outliers')
             # plt.title(f'Outliers Detection {col} for store {store}')
             # plt.xlabel(date_column)
             # plt.ylabel('Value')
             # plt.legend()
             # plt.show()
     return 'Plots for features Done'
 
    
def get_corr_with_features_and_lags_key(df_complete, key_column, date_column, target_column,
                                        select_n, lags, corr_limit, higher_level, lower_level):
    print('Getting higher Volumes by Key Combination')
    df_key_vol = df_complete.groupby([key_column]).sum([target_column]).sort_values(
        by = target_column, ascending = False)
    list_higher_vols = list( df_key_vol.head(select_n).index)
    
    print(f'Getting Correlation for the {select_n} highest volumes')
    for key in list_higher_vols:
        df = df_complete[df_complete[key_column] == key]
        df.set_index(date_column, inplace = True)
        df.drop([key_column, higher_level, lower_level], axis = 1, inplace = True)
        num_lags = lags
        #The shift on target have to be negative in order to create a row
        #with past feature and future target in the same line. The goal is to see
        #how historic and past data from features can affect the target and if it possible to use 
        #the lagged features to help predict the target. When modeling, instead of
        #use negative lag on target, will be use a positive lag on the features.
        #The negative shift on target is only for analysis propuse.
        for lag in range(1, num_lags+1):
            df[f'{target_column}_{lag}'] = df[f'{target_column}'].shift(-lag)
        
        # Calculate correlations
        correlation_matrix = df.corr()
        
        # Plot correlation matrix heatmap
        plt.figure(figsize=(10, 8))
        plt.title(f'Correlation Heatmap {key}')
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.colorbar()
        plt.show()
        
        # Select Features that have 0.5 or more of correlation with target or lagged target
        correlation_target_lag = correlation_matrix[[x for x in correlation_matrix.columns if target_column in x]]
        # filtered_df = correlation_target_lag[abs(correlation_target_lag) > 0.3]
        
        selected_indices = correlation_target_lag.index[
            correlation_target_lag.apply(lambda row: (abs(row) > corr_limit).all(), axis=1)]
        list_title = list(selected_indices)
        correlation_target_lag = correlation_target_lag.loc[selected_indices].T
        if correlation_target_lag.empty:
            print(f'No Correlations over {corr_limit}')
        else:
            plt.figure(figsize=(8, 5))
            plt.plot( correlation_target_lag, marker='o')
            plt.title(f'Correlation of {list_title} and Target and Lagged Target Columns {key}')
            # plt.xlabel('Lag', rotation = 45)
            plt.ylabel('Correlation')
            plt.xticks(range(0, len(correlation_target_lag.index)+1), rotation = 45)
            plt.grid()
            plt.show()
        
        
        # Select correlations of target and lagged target columns
        target_correlations = correlation_matrix[f'{target_column}'][[f'{target_column}_{lag}' for lag in range(1, num_lags+1)]]
        
        # Plot correlation of target and lagged target columns
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, num_lags+1), target_correlations, marker='o')
        plt.title(f'Correlation of Target and Lagged Target Columns {key}')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.xticks(range(1, num_lags+1))
        plt.grid()
        plt.show()

    return 'Correlation for key combinations done'


def get_corr_with_features_and_lags_store(df_complete, key_column, date_column, target_column,
                                        select_n, lags, corr_limit, higher_level, lower_level):
    print('Getting higher Volumes by Key Combination')
    df_store_vol = df_complete.groupby([higher_level]).sum([target_column]).sort_values(
        by = target_column, ascending = False)
    list_higher_vols = list( df_store_vol.head(select_n).index)
    
    aggregation_dict = {target_column: 'sum'}
    for column in df_complete.columns:
        if column != target_column and column != date_column and column != higher_level \
            and column != key_column and column != lower_level:
            aggregation_dict[column] = 'mean'
    
    df_group_store = df_complete.groupby([higher_level, date_column]).agg(aggregation_dict).reset_index()
    print(f'Getting Correlation for the {select_n} highest volumes for Stores')
    for store in list_higher_vols:
        df = df_group_store[df_group_store[higher_level] == store]
        df.set_index(date_column, inplace = True)
        df.drop([higher_level], axis = 1, inplace = True)
        num_lags = lags
        #The shift on target have to be negative in order to create a row
        #with past feature and future target in the same line. The goal is to see
        #how historic and past data from features can affect the target and if it possible to use 
        #the lagged features to help predict the target. When modeling, instead of
        #use negative lag on target, will be use a positive lag on the features.
        #The negative shift on target is only for analysis propuse.
        for lag in range(1, num_lags+1):
            df[f'{target_column}_{lag}'] = df[f'{target_column}'].shift(-lag)
        
        # Calculate correlations
        correlation_matrix = df.corr()
        
        # Plot correlation matrix heatmap
        plt.figure(figsize=(10, 8))
        plt.title(f'Correlation Heatmap {store}')
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.colorbar()
        plt.show()
        
        # Select Features that have 0.5 or more of correlation with target or lagged target
        correlation_target_lag = correlation_matrix[[x for x in correlation_matrix.columns if target_column in x]]
        # filtered_df = correlation_target_lag[abs(correlation_target_lag) > 0.3]
        
        selected_indices = correlation_target_lag.index[
            correlation_target_lag.apply(lambda row: (abs(row) > corr_limit).all(), axis=1)]
        list_title = list(selected_indices)
        correlation_target_lag = correlation_target_lag.loc[selected_indices].T
        if correlation_target_lag.empty:
            print(f'No Correlations over {corr_limit}')
        else:
            plt.figure(figsize=(8, 5))
            plt.plot( correlation_target_lag, marker='o')
            plt.title(f'Correlation of {list_title} and Target and Lagged Target Columns {store}')
            # plt.xlabel('Lag', rotation = 45)
            plt.ylabel('Correlation')
            plt.xticks(range(0, len(correlation_target_lag.index)+1), rotation = 45)
            plt.grid()
            plt.show()
        
        # Select correlations of target and lagged target columns
        target_correlations = correlation_matrix[f'{target_column}'][[f'{target_column}_{lag}' for lag in range(1, num_lags+1)]]
        
        # Plot correlation of target and lagged target columns
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, num_lags+1), target_correlations, marker='o')
        plt.title(f'Correlation of Target and Lagged Target Columns {store}')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.xticks(range(1, num_lags+1))
        plt.grid()
        plt.show()

    return 'Correlation for Store combinations done'

def run_histograms_plots_heatmaps(df_complete, key_column, date_column, target_column,
                                        select_n, lags, corr_limit, higher_level, lower_level):
    
    print('Explore Target Data Agregate and Diff/Pct Change from target')
    check_diff_pct(df_complete, key_column, date_column, target_column, lags,
                        corr_limit)
    
    print(f'Explore Target Data for {select_n} Stores with Highest Volumes')
    check_high_vol_store(df_complete, key_column, date_column, target_column, select_n, 
    higher_level)

    
    print(f'Explore Target Data for {select_n} Keys Combination with Highest Volumes')
    check_high_vol_key(df_complete, key_column, date_column, target_column, select_n)

    print('Get histogram for predictors features for Store')
    features_by_store(df_complete, target_column, date_column, key_column, select_n, 
    higher_level, lower_level)
    
    print('Explore Correlations between target and lagged target and also features inside key combination')
    get_corr_with_features_and_lags_key(df_complete, key_column, date_column, target_column,
                                            select_n, lags, corr_limit, higher_level, lower_level)
    
    print('Explore Correlations between target and lagged target and also features for Store')
    get_corr_with_features_and_lags_store(df_complete, key_column, date_column, target_column,
                                            select_n, lags, corr_limit, higher_level, lower_level)
    
    return "Plots, Histograms and Heatmaps Done!"



if __name__ == '__main__':
        
    

    print('teste')
    