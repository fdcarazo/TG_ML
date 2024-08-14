#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## Train-Test split of dataset-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date (Fr): Mon Mar 18 21:52:48 CET 2024-.
## last_modify (Fr): -.
##
## ======================================================================= INI79

## print(dir()); input(1)

## 1- IMPORT MODULUS ---.
## Import the required packages/libraries/modules-.
## 1-1- GENERAL MODULES -.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class test_val():
    def __init__(self,df,feat_var,targ_var):
        self.df=df
        self.feat_var=feat_var
        self.targ_var=targ_var

    def train_val(self,rand,val_frac,shuff=True):
        ## train_test_split(df.iloc[:, 0:(len(tv)+3+xp)], ## include IRUN and W11, W12, W13 in X_*-.
        X_train,X_val,y_train,y_val=train_test_split(self.df.loc[:,self.feat_var],
                                                     self.df.loc[:,self.targ_var],
                                                     test_size=val_frac,
                                                     shuffle=shuff,
                                                     random_state=rand
                                                     )
        return X_train,X_val,y_train,y_val

    def scaler(self,scaler,X_train,X_val,y_train,y_val):
        scaler=eval(scaler)
        X_train=scaler.fit_transform(X_train)
        X_val=scaler.transform(X_val)
        y_train=scaler.fit_transform(y_train)
        y_val=scaler.transform(y_val)

        ## print(np.mean(X_train));print(np.mean(X_val)),print(np.mean(y_train));print(np.mean(y_val))
        ## print(np.std(X_train));print(np.std(X_val)),print(np.std(y_train));print(np.std(y_val))
        
        return X_train,X_val,y_train,y_val

