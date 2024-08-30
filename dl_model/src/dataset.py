#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# class to convert PandasDataFrame to dataset, apply StandardScaler(), split
# in train-test sets, and create Pytorch Dataset and DataLoader-.
#
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# start date (Fr): Tue Mar 19 12:19:01 CET 2024-.
# last modify (Ar): mié 21 ago 2024 11:52:38 -03-.
##
# ====================================================================== INI79

# - ======================================================================INI79
# 1- include packages, modules, variables, etc.-.
import pandas as pd
import pickle as pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# 2don't have conflict between float datatype of torch and bnnNN
# if torch.is_tensor(xx) else torch.tensor(xx,dtype=float) (float64) and float
# of NN (float32)-.
torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)
# - ======================================================================END79


# dataset class-.
class DataObject():
    ''' constructor '''
    def __init__(self, df, feat_var, targ_var):
        # self.inp= torch.stack((torch.linspace(-2,2,100), torch.linspace(-1,1,100)))
        self.df= df  # pandaDataFrame-.
        self.feat_var= feat_var  # list-.
        self.targ_var= targ_var  # list-.

    # apply standardScaler
    def scalerStandarize(self, sca:str, sca_targ: bool, dir_save:str):
        # sacalers used for dataframe, features and targets-.
        scaler_df, scaler_feat, scaler_targ= eval(sca), eval(sca), eval(sca)
        
        # sacale/standarize all features, targets and a whole dataframe-.
        df_scal= scaler_df.fit_transform(self.df[self.feat_var+self.targ_var])
        feat_scal= scaler_feat.fit(self.df[self.feat_var])
        targ_scal= scaler_targ.fit(self.df[self.targ_var])        
        # built pandasDataframe using dataframe scaled/standarized-.
        df_scal= pd.DataFrame(df_scal, columns=self.feat_var+self.targ_var)

        # built pandasDataframe using only features scaled/standarized-.
        feat_scaled = scaler_feat.fit_transform(self.df[self.feat_var])
        arr_feat_Scaled = np.concatenate((feat_scaled,
                                          self.df[self.targ_var].to_numpy(dtype=float)),
                                         axis=1)
        '''
        print(np.mean(arr_feat_Scaled, axis=0),
              np.std(arr_feat_Scaled, axis=0),
              np.shape(arr_feat_Scaled), sep='\n'); input(88)
        '''
        
        df_feat_scal = pd.DataFrame(arr_feat_Scaled, columns=self.feat_var+self.targ_var)
        # save scalers-.
        feat_scal_name= dir_save+'/'+'featScaler.pkl'
        fsac= open(feat_scal_name, 'wb')
        targ_scal_name= dir_save+'/'+'targScaler.pkl'
        tsca= open(targ_scal_name, 'wb')
        all_scal_name= dir_save+'/'+'allScaler.pkl'
        asca= open(all_scal_name, 'wb')

        pickle.dump(scaler_feat, fsac)
        pickle.dump(scaler_targ, tsca)
        pickle.dump(scaler_df, asca)
        fsac.close(); tsca.close(); asca.close() 
        
        return df_scal, df_feat_scal, feat_scal, targ_scal

    # split in train-test-.
    def train_test(self, df, df_scaled, rand:float, ts=0.1, shuff=True):
        
        X_train, X_val, y_train, y_val=train_test_split(df_scaled[self.feat_var],
                                                        df_scaled[self.targ_var],
                                                        ## df[self.targ_var],
                                                        test_size= ts,
                                                        shuffle= shuff,
                                                        random_state= rand)
        # print(type(X_train),type(X_val),type(y_train),type(y_val), sep='');input(99)
        # it isn't necessary, they are numpy.array-.
        return X_train.to_numpy(dtype=float), X_val.to_numpy(dtype=float),\
            y_train.to_numpy(dtype=float), y_val.to_numpy(dtype=float)

    
# create Pytorch Dataset and Pytorch DataLoader-.
class DatasetTorch(Dataset):
    def __init__(self, x, y):
        self.x= x if torch.is_tensor(x) else torch.tensor(x, dtype=float)
        self.y= y if torch.is_tensor(y) else torch.tensor(y, dtype=float)
        # print(self.x.dtype); print(self.y.dtype), input(44) # torch.float64

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
# create Pytorch DataLoader-.
class DataloaderTorch(DataLoader):
    def __init__(self, torch_dataset, batch_size, shuff):
        self.dataloader= DataLoader(torch_dataset,
                                    batch_size= batch_size,
                                    shuffle= shuff)

# main (to load as module)-.
if __name__ == '__main__':
    pass
    
