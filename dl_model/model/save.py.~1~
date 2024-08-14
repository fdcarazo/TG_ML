#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: 2save DL model and loss values-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
## start date (Fr): Mon Mar  4 16:25:40 CET 2024-.
## last modify (Fr): -.
##
## ====================================================================== INI79
import pickle as pickle
from torch import save as torch_save

class SaveDLModelLoss():
    def __init__(self,dir_save:str):
        self.dir_save=dir_save
        
    def save_model(self,model):
        ## save the model-.                                                                                                      
        filemod=self.dir_save+'/'+'dlModelWithoutHyperOpt_pkl'+ '.pkl'
        with open(filemod, 'wb') as fdl: pickle.dump(model, fdl)
        fdl.close()
        filemod=self.dir_save+'/'+'dlModelWithoutHyperOpt_torch'+ '.pkl'
        with open(filemod, 'wb') as fdl: torch_save(model, fdl)
        fdl.close()
        filemod=self.dir_save+ '/'+ 'dlModelWithoutHyperOpt_sd'+ '.pt' ## state_dict                                   
        with open(filemod, 'wb') as fdl: torch_save(model.state_dict(), filemod)
        fdl.close()

    def save_loss(self,loss:dict):
        ## print a lossFile-.                                                                                                    
        loss_file=self.dir_save+'/'+'loss.p'
        ## with open(loss_file, 'wb') as fl: file_loss= open(fl)                                                                 
        file_loss=open(loss_file,'wb')
        ## dictionary2print-.                                                                                                    
        pickle.dump(loss,file_loss)
        file_loss.close()
