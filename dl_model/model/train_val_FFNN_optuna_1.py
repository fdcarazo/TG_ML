#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# SCRIPT: 2train and test (with validation DS) best_model obtained with
#         Optuna -.
##
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# start date (Arg): mart 03 sep 2024 14:37:24 -03-.-.
# last modify (Arg): mié 04 sep 2024 07:51:24 -03-.
##
# ====================================================================== INI79

# ====================================================================== INI79
# include packages/modules/variables, etc.-.
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm as tqdm
import time as time

from model.nn_for_optuna import RegressionModel
# - ======================================================================END79


# - ======================================================================INI79
# to train and test models-.
class TrainPredict():
    # some params are needed only to inherit MLP class (for constructor)-.
    def __init__(self, model, train_loader, val_loader, device): 
        ''' class constructor
        Args:
        =====
        Returns:
        =======
        '''
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def train(self, epochs=1, val=False):
        ''' FeedForwardNeuralNetwork -FFNN- trainer
        Args:
        =====
        Returns:
        =======
        '''
        
        #
        optimizer = self.model.optimizer
        criterion = self.model.loss
        # optimizer = optim.Adam(self.model.parameters(), lr=1.0e-1)
        # criterion = nn.MSELoss()
        
        
        # dictionary to save train and validation looses as lists-.
        loss_dir = {'train_loss': [], 'val_loss': []}
        
        train_time = 0.0
        print(len(self.train_loader))
        
        # n_train = len(self.train_loader)
        # train
        # -> 2apply dropout/normalization (only in train) and 2Calc and Save the
        #    grad of params W.R.T. loss function which T.B.U. to update model 
        #    params (weights and biases)-.
        self.model.train()

        epochs = tqdm.tqdm(range(epochs), desc='Epochs')
        start_time = time.time()
        
        for iepoch in epochs:
            # for iepoch in range(epochs):
            # with tqdm.tqdm(total=n_train,position=0) as pbar_train:
            #    pbar_train.set_description('Epoch -- {0} -- / '+'({1})'+ ' - train- {2}'.
            #                               format(epoch+1,'epoch','\n')
            #                               )
            #    pbar_train.set_postfix(avg_loss='0.0')
            loss_dir['train_loss'].append(self.fit_regression_ffnn(self.train_loader,
                                                                   # pbar_train,
                                                                   optimizer,
                                                                   criterion,
                                                                   True))
            # val loop-.
            if val:
                n_val = len(self.val_loader)
                # -> if exists (layers), don't applied dropout, batchNormalization,
                # and don't registers gradients among other things-.
                self.model.eval() 
                # with tqdm(total=n_val, position=0) as pbar_val:
                # pbar_val.set_description('Epoch -- {0} -- / '+'({1})'+ ' - val-.'.
                #                          format(epoch+1,'epoch')
                #                          )
                # pbar_val.set_postfix(avg_loss='0.0')
                loss_dir['val_loss'].append(self.fit_regression_ffnn(self.val_loader,
                                                                     ## pbar_val,
                                                                     optimizer,
                                                                     criterion,
                                                                     False))
                print('Epoch -- {0} -- / '+'({1})'+ ' - val_loss = {2}'.
                      format(iepoch+1,'epoch',loss_dir['val_loss'][-1]))

        train_time = time.time()- start_time # per epoch-.
        return train_time, loss_dir

    
    def fit_regression_ffnn(self, dataloader, optimizer, criterion, train=True):
        ''' train and/or validation using batches
        Args:
        =====
        Returns:
        =======
        '''
        
        running_loss = 0.0
        self.model.to(self.device)
        for idl, (X,Y) in enumerate(dataloader, 0):
            # X,Y = map(lambda x: x.to('cpu'),data)  # cpu or gpu-.
            X, Y = X.to(self.device), Y.to(self.device)  # map(lambda x: x.to('cpu'), data) # cpu or gpu-.
            # print(X.get_device(), Y.get_device(), sep='\n'); input(11)
            # if train: self.model.optimizer.zero_grad() # re-start gradient-.
            if train: optimizer.zero_grad() # re-start gradient-.
            pred = self.model(X)  # forward pass -.
            # loss = self.model.loss(pred, Y)  # evaluate prediction-.
            loss = criterion(pred, Y)  # evaluate prediction-.
            
            if train:
                # apply loss back propropagation-.
                loss.backward() # backpropagation-.
                # self.model.optimizer.step()  # update parameters (weights and biases))-.
                optimizer.step()  # update parameters (weights and biases))-.
            running_loss += loss.item()

            # pbar.set_postfix(avg_loss='{:.4f}'.format(avg_loss))
            # pbar.update(Y.shape[0]) # ?-.

        avg_loss = running_loss/len(X)
        return avg_loss

    
    def fit_regression_bnn(self, dataloader,train=True):
        ''' train and/or validation using batches '''
        
        running_loss=0.0
        
        # for idl, data in enumerate(dataloader, 0):
        for idl, (X, Y) in enumerate(dataloader, 0):
            # X,Y = map(lambda x: x.to('cpu'),data)  # cpu or gpu-.
            X,Y = X.to(self.device), Y.to(self.device)  # map(lambda x: x.to('cpu'),data) # cpu or gpu-.
            # print(X.get_device(), Y.get_device(), sep='\n'); input(11)
            if train: self.model.optimizer.zero_grad()  # re-start gradient-.
            pred = self.model(X)  # forward pass -.
            loss = self.model.loss(pred, Y)  # evaluate prediction-.
            kl = self.model.kl_loss(self.model)  # calc. loss (as KL -between distributions-)-.
            cost = loss + self.model.kl_weight*kl  # calc total loss (MSE_loss + KL_loss)-.
            
            if train:
                # apply loss back propropagation-.
                cost.backward() # backpropagation-.
                self.model.optimizer.step() # update parameters (weights and biases))-.
                
            running_loss += cost.item()
            
            # pbar.set_postfix(avg_loss='{:.4f}'.format(avg_loss))
            # pbar.update(Y.shape[0]) # ?-.
            
        avg_loss = running_loss/len(X)
        return avg_loss
# - ======================================================================END79
