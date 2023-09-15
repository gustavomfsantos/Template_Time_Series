# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 11:40:13 2023

@author: gusta
"""
import os
import pandas as pd
import numpy as np


def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

#Evaluate on real values, not using  CV results
def evaluate_higher_level_forecast(final_path, higher_level, models, target_column):
    df_accuracy = pd.DataFrame()
    final_files = os.listdir(final_path)
    higher_forecast_files = [x for x in final_files if higher_level and 'CV' in x]
    
    for file in higher_forecast_files:
        df_acc1 = pd.DataFrame()
        print(file)
        df = pd.read_csv( os.path.join(final_path, file), decimal = '.', sep = ',',
                                      index_col = 0)
        df = df.dropna(subset = (target_column), axis = 0)

        
        #General Acc for all models in each file
        for model in models:
            df[f'Acc_ind_{model}'] = 1 - ( abs( df[target_column] - df[f'Prediction_{model}'])/df[target_column] )
            
            df_acc2 = pd.DataFrame()
            
            # print(file, model, 1 - (wmape(df[target_column], df[f'Prediction_{model}'])))
            # print('Time spent', df[f'Tuning_Time_{model}'].sum())
            df_acc2['File'] = [file]
            df_acc2['Model'] = [model]
            df_acc2['Acc_wheighted'] = 1 - (wmape(df[target_column], df[f'Prediction_{model}']))
            df_acc2['Acc_ind_mean'] = df[f'Acc_ind_{model}'].mean()
            df_acc2['Time_spent'] = df[f'Tuning_Time_{model}'].sum()
            
            df_acc1 = pd.concat([df_acc1, df_acc2], ignore_index=True)
            
        df_accuracy =  pd.concat([df_accuracy, df_acc1], ignore_index=True)
        
    acc_cols = [x for x in df_accuracy.columns if 'Acc' in x]   
    for col in acc_cols:
        df_accuracy = df_accuracy[df_accuracy[col] > 0.90]
        
    df_accuracy.sort_values(by = 'Time_spent', ascending = True, inplace = True)    
    
    model_option = df_accuracy['Model'].head(1).iloc[0]
    file_option = df_accuracy['File'].head(1).iloc[0]
    
    return df_accuracy, model_option, file_option
            #FAZER media das acc individuais
#Go with no standard scalling
def evaluation(final_path, higher_level, models, target_column,
                                   date_column):
    
    df_accuracy, model_option, file_option = evaluate_higher_level_forecast(final_path, 
                                                    higher_level, models, target_column)
    # df_accuracy = df_accuracy.iloc[0]
    
    df_final = pd.read_csv( os.path.join(final_path, file_option), decimal = '.', sep = ',',
                                  index_col = 0)
    df_final = df_final.dropna(subset = (target_column), axis = 0)
    
    model_cols = [x for x in df_final.columns if model_option and 'Pred' in x]
    
    
    
    df_final = df_final[['Round_Date', 'Horizon', date_column, higher_level, target_column] +
                        model_cols]
    
    df_final.rename(columns = {model_cols:'Predictions'})
    
    return df_final












# def scores_each_key(df, models, date_column, higher_level, target_column):
    
#     #Each row is a horizon for store
#     cv_scores_cols = [x for x in df.columns if 'CV' and 'Score' in x]
#     cv_std_cols = [x for x in df.columns if 'CV' and 'Std' in x]
    
#     df[cv_scores_cols] = df[cv_scores_cols].abs()
#     #Create Coeficient Variation for each model and create a score mean discounted by Coef Var.
#     #The point is that score is MSE and lower value is better
#     #The Coef is Score divide by standard deviation. However, both score and std are better when smaller
#     #A division does not make sense. smaller multiply by smaller makes
#     #So we need to find a metric that combine the two values in order that smaller indicates better, for both.
#     #Since both values were gotten by normalized data, they are normalized
#     for model in models:
        
#         df[f'Coef_Variation_{model}'] = df[f'CV_Score_Mean_{model}']/df[f'CV_StdDev_Mean_{model}']
#         df[f'Score_Weighted_by_StdDev_{model}'] = df[f'CV_Score_Mean_{model}']/df[f'Coef_Variation_{model}'] 
        
#     coef_var_cols = [x for x in df.columns if 'Score_Weighted' in x] 
    
#     df['Score_Weighted_Sum'] = df[coef_var_cols].sum(axis=1)
    
#     for model in models:
        
#         df[f'Score_Weighted_Porportion_{model}'] = df[f'Score_Weighted_by_StdDev_{model}']/df['Score_Weighted_Sum']
    
#     score_weighted_cols = [x for x in df.columns if 'Score_Weighted_Porportion' in x]
#     prediction_cols = [x for x in df.columns if 'Prediction' in x]

#     df_aux = df[['Round_Date', 'Horizon', date_column, higher_level, target_column]
#                 + prediction_cols + score_weighted_cols]
    

#     return 'Done'

if __name__ == '__main__':
    
    
    print('teste')
