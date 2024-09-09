#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# Neural Network for Optuna-.
#
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# start date (Arg): lun 02 sep 2024 14:32:33 -03-.
# last modify (Arg): lun 02 
##
# ====================================================================== INI79


# ====================================================================== INI79
# include packages/modules/variables, etc.-.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pickle

import torchbnn as bnn

from .weight_init import weights_init

from .ffnn import MLP

# 2don't have problems between float datatype of torch and bnnNN
# if torch.is_tensor(xx) else torch.tensor(xx,dtype=float) (float64) and float
# of NN (float32)-.
torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)
# - ======================================================================END79


class nn_hyper_param_optim():
    def __init__(self, features, targets):
        self.features = features  # list-.
        self.targets = targets  # list-.

    # https://medium.com/pytorch/using-optuna-to-optimize-pytorch-hyperparameters-990607385e36
    def define_model(self, trial, input_size, output_size):
        '''
        optimize the number of layers, hidden units and dropout ratio
        in each layer
        '''
    
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        
        in_features = len(self.features)
        for i in range(n_layers):
            out_features = trial.suggest_int('n_units_l{}'.format(i), 32, 64
                                             )
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_uniform('dropout_l{}'.format(i), 0.0, 0.2)
            layers.append(nn.Dropout(p))
            
            in_features = out_features
        
        layers.append(nn.Linear(in_features, output_size))
        # layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    # objective method for Optuna-.
    def objective(self, trial, input_size, output_size, train_dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # model creation-.
        model = self.define_model(trial, input_size, output_size).to(device)
        model.apply(weights_init)  # weights initialization -.
    
        #  optimizers generation.
        optimizer_name = trial.suggest_categorical('optimizer',
                                                   ['Adam', 'RMSprop', 'SGD'])
        # optimizer_name= trial.suggest_categorical('optimizer', ['Adam'])
        lr = trial.suggest_loguniform('lr', 1e-3, 1e-1)
        # lr= 1.0e-2
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
    
        # first define a dictionary to save train and validations loss values-.
        # loss_stats= {'train_loss_d': [],
        #             'val_loss_d': []
        #             }
        loss_stats = {'train_loss_d': []}
    
        # train and validate final model (to have points to plot loss)-.
        num_epochs = 10
        # dist.init_process_group(backend='gloo')
        for epoch in range(num_epochs):        
            train_epoch_loss = 0.0
            model.train()
        
            for batch_features, batch_labels in train_dataloader:
                # send batch-datas to device-.
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                # forward propagation and loss calculation-.
                optimizer.zero_grad()
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # backward propagation and weights and bias updates-.
                loss.backward()
                optimizer.step()
                
                train_epoch_loss += loss.item()
                
            # calculate and save the mean loss for each epoch-.
            loss_stats['train_loss_d'].append(train_epoch_loss/len(train_dataloader))
        
            # if the time limit is achieved or the process is interrumpted, runing is stoped-.
            # if trial.should_prune(): break
            
        '''
        import matplolib.pyplot as plt
        fig. axes = plt.figure(figsize=(12,6))
        plt.scatter(loss_stats['train_loss_d'])
        plt.show()
        '''
        print(loss_stats['train_loss_d']); input(2222222)

        return np.mean(train_epoch_loss)  # loss_stats#['train_loss_d'].item()

# - ======================================================================END79

if __name__ == 'main':
    pass
