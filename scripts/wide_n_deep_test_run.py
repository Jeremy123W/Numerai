#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:40:47 2017

@author: jeremy
"""
import pandas as pd
import datetime as dt
import wide_and_deep_model
import numpy as np

    
    
    
    
if __name__ == "__main__":
    start_time = dt.datetime.now()
    print("Start time: ",start_time)
  

    train = pd.read_csv('../numerai_training_data.csv')
    test = pd.read_csv('../numerai_tournament_data.csv')

    features=list(train.columns)
    test['target']=np.nan
    features.remove('target')
    

    
    print(features)

    print("Building model.. ",dt.datetime.now()-start_time)
    preds = wide_and_deep_model.train_and_eval(train,test,"wide_n_deep",'target')

    
    
    print("Creating submission file...")
    out_df = test.t_id.copy()
    out_df=out_df.to_frame()
    out_df['probability']=preds
    out_df.to_csv('submission_file.csv', index=False)  
    print("Submission file created: ") 
    
    print(dt.datetime.now()-start_time)
    
    
    
    