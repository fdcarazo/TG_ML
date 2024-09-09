#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# hyperparamenter optimization of NN-.
#
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# start date (Arg): dom 01 sep 2024 08:41:53 -03-.
# last modify (Arg): lun 02 sep 2024 18:30:22 -03-.
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

from .nn_for_optuna import RegressionModel

# 2don't have problems between float datatype of torch and bnnNN
# if torch.is_tensor(xx) else torch.tensor(xx,dtype=float) (float64) and float
# of NN (float32)-.
torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)
# - ======================================================================END79


class nn_hyper_param_optim():
    def __init__(self, train_loader, val_loader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        
    # define training function-.
    def train(self, model, criterion, optimizer, dataloader):
        model.train()
        total_loss = 0
        for i_batch, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    # define evaluation function-.
    def evaluate(self, model, criterion, dataloader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for i_batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), \
                    targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    
    # objective method for Optuna-.
    def objective(self, trial, input_size, output_size, train_dataloader,
                  val_dataloader, n_epochs):

        # hyperparameters to optimize-.
        input_size = input_size  # example input size-.
        hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
        hidden_units = trial.suggest_int('hidden_units', 512, 1024)
        dropout_prob = trial.suggest_float('dropout_prob', 0.0, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        batch_size = trial.suggest_int('batch_size', 500, 1000)


        # initialize model, criterion, and optimizer-.
        model = RegressionModel(input_size, hidden_layers, hidden_units,
                                dropout_prob)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # dictionary to save train and validation looses as lists-.
        loss_dir = {'train_loss': [], 'val_loss': []}
        # train and evaluate-.
        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):  # fixed number of epochs-.
            train_loss = self.train(model, criterion, optimizer,
                                    self.train_dataloader)
            val_loss = self.evaluate(model, criterion, self.val_dataloader)

            loss_dir['train_loss'].append(train_loss)
            loss_dir['val_loss'].append(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
    
        # save the losses-.
        trial.set_user_attr('all_loss', loss_dir)
        trial.set_user_attr('train_loss', train_losses)
        trial.set_user_attr('val_loss', val_losses)
        
        return val_losses[-1]
# - ======================================================================END79


if __name__ == 'main':
    pass
