#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:40:47 2017

@author: jeremy
"""
import pandas as pd
import datetime as dt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
import wide_and_deep_model
import random


    
    
    
if __name__ == "__main__":
    start_time = dt.datetime.now()
    print("Start time: ",start_time)
  

    train = pd.read_csv('../numerai_training_data.csv')
    print(train.isnull().values.any())
    features=list(train.columns)
    
    features.remove('target')
    
    #rem_list=['feature19','feature12','feature21','feature24']
    
    #for each in rem_list:
    #    features.remove(each)
    #print(train['target'])
 
    train, test = train_test_split(train, test_size=.1, random_state=random.seed(2017))

    
    print(features)

    print("Building model.. ",dt.datetime.now()-start_time)
    preds = wide_and_deep_model.train_and_eval(train,test,"wide_n_deep",'target')
    score = log_loss(test['target'].values, preds)

    print('logloss: {:.6f}'.format(score))
    

    
    
    