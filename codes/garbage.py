# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:18:42 2023

@author: gusta
"""

class expanding_window(object):
    '''	
    Parameters 
    ----------
    
    Note that if you define a horizon that is too far, then subsequently the split will ignore horizon length 
    such that there is validation data left. This similar to Prof Rob hyndman's TsCv 
    
    
    initial: int
        initial train length 
    horizon: int 
        forecast horizon (forecast length). Default = 1
    period: int 
        length of train data to add each iteration 
    '''
    

    def __init__(self,initial= 1,horizon = 1,period = 1):
        self.initial = initial
        self.horizon = horizon 
        self.period = period 


    def split(self,data):
        '''
        Parameters 
        ----------
        
        Data: Training data 
        
        Returns 
        -------
        train_index ,test_index: 
            index for train and valid set similar to sklearn model selection
        '''
        self.data = data
        self.counter = 0 # for us to iterate and track later 


        data_length = data.shape[0] # rows 
        data_index = list(np.arange(data_length))
         
        output_train = []
        output_test = []
        # append initial 
        output_train.append(list(np.arange(self.initial)))
        progress = [x for x in data_index if x not in list(np.arange(self.initial)) ] # indexes left to append to train 
        output_test.append([x for x in data_index if x not in output_train[self.counter]][self.horizon-1:self.horizon] ) #Or [:self.horizon] to get 1 to horizons and not only the horizon
        # clip initial indexes from progress since that is what we are left 
         
        while len(progress) != 0:
            temp = progress[:self.period]
            to_add = output_train[self.counter] + temp
            # update the train index 
            output_train.append(to_add)
            # increment counter 
            self.counter +=1 
            # then we update the test index 
            
            to_add_test = [x for x in data_index if x not in output_train[self.counter] ][self.horizon-1:self.horizon]  #Or [:self.horizon] to get 1 to horizons and not only the horizon
            output_test.append(to_add_test)

            # update progress 
            progress = [x for x in data_index if x not in output_train[self.counter]]	
            
        # clip the last element of output_train and output_test
        output_train = output_train[:-1]
        output_test = output_test[:-1]
        
        # mimic sklearn output 
        index_output = [(train,test) for train,test in zip(output_train,output_test)]
        
        return index_output
    
    #initial: int
    #     initial train length 
    # horizon: int 
    #     forecast horizon (forecast length). Default = 1
    # period: int 
    #     length of train data to add each iteration 
    # tscv = expanding_window(initial = len(df_train)/1.2, horizon = horizon, period = horizon)
    # index_output = tscv.split(df_train)
    # #Transform index_output in int
    # index_output = [[tuple(int(num) for num in inner) for inner in outer] for outer in index_output]
    # #Dropping list with empty tupples
    # index_output = [lst for lst in index_output if not any(not tpl for tpl in lst)]