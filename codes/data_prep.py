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


def import_dataset(data_path):
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
    

    features_df = pd.merge(features_df, stores_df, how = 'left', on = ['Store'])
    features_df['Date'] = pd.to_datetime(features_df['Date'], format='%d/%m/%Y')
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
    sales_df['Date'] = pd.to_datetime(sales_df['Date'], format='%d/%m/%Y')
    print('Ajusted date format for sales data and droped feature duplicated')

    df_sales_final = pd.merge(sales_df, features_df, how = 'left', on = ['Date',
                                                                      'Store'])
    del stores_df, sales_df,features_df
    print('Final merge realized and dataframes deleted')
    
    print('Create key for Store-Department')
    df_sales_final['key_Store_Dept'] = df_sales_final['Store'].astype(str) + '_' +  df_sales_final[
        'Dept'].astype(str) 
    
    df_sales_final.sort_values(['Store', 'Dept', 'Date'], inplace = True)
    # df_sales_final.drop(['Store', 'Dept'], axis = 1, inplace = True)
    df_sales_final.reset_index(drop = False, inplace = True)    
    print('Values sorted by store, dept and date')

    df_sales_final = memory_aux.reduce_mem_usage(df_sales_final)

    df_sales_final = df_sales_final[['Date', 'key_Store_Dept', 'Store', 'Dept', 'Weekly_Sales', 'Temperature',
                         'Fuel_Price', 'MarkDown1', 'MarkDown2',
           'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment',
           'IsHoliday', 'Size', 'Size_Type']]
    df_sales_final[['Size', 'Size_Type', 'IsHoliday']] = df_sales_final[['Size', 'Size_Type', 'IsHoliday']].astype(float)
    print('Final Dataset is ready!')
    
    return df_sales_final
    


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
    
    
    
   
    

    