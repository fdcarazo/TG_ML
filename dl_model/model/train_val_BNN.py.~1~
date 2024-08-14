#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: 2train and validate any FFNN-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
## start date (Fr): Mon Mar  4 12:39:33 CET 2024-.
## last modify (Fr): -.
##
## ====================================================================== INI79

## ====================================================================== INI79
## 1- include packages, modules, variables, etc.-.
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pickle
import tqdm as tqdm
import time as time

from . ffnn_bnn import BayesianNet
## - ======================================================================END79
## - ======================================================================INI79
## 2- To train and test models-.
class TrainPredict(BayesianNet):
    def __init__(self,model,train_dl,val_dl,inp,out,loss,optimizer,kl_loss,
                 device,w_d,lr,kl_weight
                 ):
        ''' class constructor '''
        super().__init__(inp,out,loss,optimizer,kl_loss,device,w_d,lr,kl_weight)
        
        self.model=model
        self.train_dl=train_dl
        self.val_dl=val_dl
        self.device=device
        
    def train(self,epochs=1, val=False):
        ''' Bayesian NN trainer '''
        loss_dir={'train_loss': [],
                  'val_loss': []
                  }
        train_time=0.0
        n_train=len(self.train_dl)
        ## train
        ## -> 2apply dropout/normalization (only in train) and 2Calc and Save the
        ##    grad of params W.R.T. loss function which T.B.U. to update model 
        ##    params (weights and biases)-.
        self.model.train()

        epochs=tqdm.tqdm(range(epochs), desc='Epochs')
        for iepoch in epochs:
            ##########for iepoch in range(epochs):
            start_time=time.time()
            ##########with tqdm.tqdm(total=n_train,position=0) as pbar_train:
            ##########   pbar_train.set_description('Epoch -- {0} -- / '+'({1})'+ ' - train- {2}'.
            ##########                              format(epoch+1,'epoch','\n')
            ##########                               )
            ##########    ## pbar_train.set_postfix(avg_loss='0.0')
            loss_dir['train_loss'].append(self.fit_regression(self.train_dl,
                                                              ## pbar_train,
                                                              True
                                                              )
                                          )        
            ## val loop-.
            if val:
                n_val=len(self.val_dl)
                ## -> if exists (layers), don't applied dropout, batchNormalization,
                ## and don't registers gradients among other things-.
                self.model.eval() 
                ## with tqdm(total=n_val, position=0) as pbar_val:
                ## pbar_val.set_description('Epoch -- {0} -- / '+'({1})'+ ' - val-.'.
                ##                             format(epoch+1,'epoch')
                ##                             )
                ## pbar_val.set_postfix(avg_loss='0.0')
                loss_dir['val_loss'].append(self.fit_regression(self.val_dl,
                                                                ## pbar_val,
                                                                False
                                                                )
                                            )
        train_time=time.time()- start_time # per epoch-.
        return train_time,loss_dir

    ########## def fit_regression(self,dataloader,pbar,train=True):
    def fit_regression(self,dataloader,train=True):
        ''' train and/or validation using batches '''
        running_loss=0.0
        ## for idl, data in enumerate(dataloader,0):
        for idl, (X,Y) in enumerate(dataloader,0):
            ## X,Y=map(lambda x: x.to('cpu'),data) # cpu or gpu-.
            X,Y=X.to(self.device),Y.to(self.device) ## map(lambda x: x.to('cpu'),data) # cpu or gpu-.
            ## print(X.get_self.device(), Y.get_self.device(), sep='\n'); input(11)
            if train: self.model.optimizer.zero_grad() # re-start gradient-.
            pred=self.model(X) # forward pass -.
            loss=self.model.loss(pred,Y) # evaluate prediction-.
            kl=self.model.kl_loss(self.model) # calc. loss (as KL -between distributions-)-.
            cost=loss+self.model.kl_weight*kl # calc total loss (MSE_loss + KL_loss)-.            
            if train:
                ## apply loss back propropagation-.
                cost.backward() # backpropagation-.
                self.model.optimizer.step() # update parameters (weights and biases))-.
            running_loss+=cost.item()
        avg_loss=running_loss/len(X)
            ########## pbar.set_postfix(avg_loss='{:.4f}'.format(avg_loss))
            ## pbar.update(Y.shape[0]) # ?-.
        return avg_loss
## - ======================================================================END79
