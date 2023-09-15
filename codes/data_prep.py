# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 17:48:19 2023

@author: gusta
"""


#Get libs
import pandas as pd
import numpy as np
import os
from datetime import timedelta

import memory_aux
from gluonts.dataset.pandas import PandasDataset

import matplotlib.pyplot as plt
from gluonts.mx import DeepAREstimator
from gluonts.mx.trainer import Trainer



def data_prep_deepAR(df_complete, round_date, date_column, key_column, target_column):
    

    data2 = df_complete[[date_column, key_column, target_column]]
    
    data2 = pd.pivot_table(data = data2, columns = key_column,
                                    index = date_column, values = target_column)#.dropna(thresh=25, axis=1)
    data2 = data2.fillna(0)
    
    data2 = pd.melt(data2.reset_index(), id_vars=date_column, var_name=key_column, value_name=target_column)

    
    train = data2.loc[data2['Date'] <= round_date]
    valid = data2.loc[(data2['Date'] > round_date) & (data2['Date'] < '2030-01-01')]    
    data2 = memory_aux.reduce_mem_usage(data2)
    train = memory_aux.reduce_mem_usage(train)
    valid = memory_aux.reduce_mem_usage(valid)
    
    return data2, train, valid

def change_pred_name(df):
    
    pred_col = [x for x in df.columns if 'pred' in x or 'Pred' in x]
    print(pred_col)
    df = df.rename(columns = {pred_col[0]: 'Predictions'})
    
    return df

def import_dataset(data_path, target_column, date_column, higher_level, lower_level,
                   key_column):
    print('Files')
    print(os.listdir(data_path))

    features_df = pd.read_csv(os.path.join(data_path, 'Features data set.csv'))
    sales_df = pd.read_csv(os.path.join(data_path, 'sales data-set.csv'))
    stores_df = pd.read_csv(os.path.join(data_path, 'stores data-set.csv'))
    print('Data loaded')
    print('Reduce memory use')
    features_df = memory_aux.reduce_mem_usage(features_df)
    sales_df = memory_aux.reduce_mem_usage(sales_df)
    stores_df = memory_aux.reduce_mem_usage(stores_df)
    print('Memory use Reduced ')

    stores_df['Size_Type'] = np.where(stores_df['Type'] == 'A', 3,
                                      np.where(stores_df['Type'] == 'B', 2 ,1))
    stores_df = stores_df.drop(['Type'], axis = 1)
    print('Size Store ordinal feature created')
    

    features_df = pd.merge(features_df, stores_df, how = 'left', on = [higher_level])
    features_df[date_column] = pd.to_datetime(features_df[date_column], format='%d/%m/%Y')
    print('Store features included on Sales Dataset')
    # Check Date
    
    features_df[[ 'MarkDown1', 'MarkDown2',
           'MarkDown3', 'MarkDown4', 'MarkDown5']] = features_df[[ 'MarkDown1', 
                                                                   'MarkDown2',
           'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)
    print('Ajusted NAN for markdown')                                                  

    features_df['IsHoliday'] =  np.where(features_df['IsHoliday'] == True, 1, 0) 
    
   

    print('Most Recent data not avaible for CPI and Unployment')
    
    sales_df = sales_df.drop(['IsHoliday'], axis  = 1)
    sales_df[date_column] = pd.to_datetime(sales_df[date_column], format='%d/%m/%Y')
    print('Ajusted date format for sales data and droped feature duplicated')

    df_sales_final = pd.merge(sales_df, features_df, how = 'left', on = [date_column,
                                                                      higher_level])
    del stores_df, sales_df,features_df
    print('Final merge realized and dataframes deleted')
    
    print('Create key for Store-Department')
    df_sales_final[key_column] = df_sales_final[higher_level].astype(str) + '_' +  df_sales_final[
        lower_level].astype(str) 
    
    df_sales_final.sort_values([higher_level, lower_level, date_column], inplace = True)
    # df_sales_final.drop(['Store', 'Dept'], axis = 1, inplace = True)
    df_sales_final.reset_index(drop = False, inplace = True)    
    print('Values sorted by store, dept and date')
    df_sales_final[target_column][df_sales_final[target_column] < 0] = 0

    df_sales_final = memory_aux.reduce_mem_usage(df_sales_final)

    df_sales_final = df_sales_final[[date_column, key_column, higher_level, lower_level, target_column, 'Temperature',
                         'Fuel_Price', 'MarkDown1', 'MarkDown2',
           'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment',
           'IsHoliday', 'Size', 'Size_Type']]
    df_sales_final[['Size', 'Size_Type', 'IsHoliday']] = df_sales_final[['Size', 'Size_Type', 'IsHoliday']].astype(float)
    print('Final Dataset is ready!')
    
    return df_sales_final
    
def groupby_store(df_complete, higher_level, lower_level, date_column, target_column, key_column):
    
    aggregation_dict = {target_column: 'sum'}
    for column in df_complete.columns:
        if column != target_column and column != date_column and column != higher_level \
            and column != key_column and column != lower_level:
            aggregation_dict[column] = 'mean'
    
    df_group_store = df_complete.groupby([higher_level, date_column]).agg(aggregation_dict).reset_index()
    
    
    return df_group_store

def group_all(df_group_store, date_column, target_column):
    
    df_agg_all = df_group_store.groupby([date_column]).sum()[target_column].reset_index()
    
    return df_agg_all

def count_obs_for_key(df_sales_final):
    
    print('Check Data Avaible for each store-dept')
    print('Get combination with 5 or less observations')
    df_few_obs = pd.DataFrame(columns = ['Store_Dept', 'Number_obs'])
    for store_id in df_sales_final['key_Store_Dept'].unique():
        print(store_id, len(df_sales_final[df_sales_final['key_Store_Dept'] == store_id]))
        # if len(df_sales_final[df_sales_final['key_Store_Dept'] == store_id]) <6:
        list_append = [store_id, len(df_sales_final[df_sales_final['key_Store_Dept'] == store_id])]
        df_few_obs.loc[len(df_few_obs)] = list_append
    del list_append, store_id
    print(len(df_few_obs), 'Combinations with 5 or less observations')
    
    return df_few_obs
  
   


def define_date_index_split(df, split_ratio):
    end_date = df.index.max() 
    start_date = df.index.min() 
    date_interval = (end_date - start_date).days

    # Calculate the 90% point of the interval
    ninety_percent_point = start_date + timedelta(days=int(date_interval * split_ratio))
    
    print("Date interval:", date_interval, "days")
    print("90% Point:", ninety_percent_point.strftime("%Y-%m-%d"))
    split_date = pd.Timestamp(ninety_percent_point)  # Replace "yyyy-mm-dd" with your desired split date
    
    return split_date



def clean_plots_files(folder_path):
    files = os.listdir(folder_path)
    if len(files)>0:
        for file in files:
            os.remove(os.path.join(folder_path, f'{file}'))
    
    return 'Folder cleaned'
    
if __name__ == '__main__':
    
    print('teste')
    
    
    
   
    

    