#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# SCRIPT: 2save DL model and loss values-.
#
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# start date (Fr): Mon Mar  4 16:25:40 CET 2024-.
# last modify (Fr): -.
# last modify (Arg): vie 06 sep 2024 17:50:44 -03 -.
##
# ====================================================================== INI79


# import modules/package/libraries
import pickle as pickle
from torch import save as torch_save


# class definition-.
class SaveDLModelLoss():
    def __init__(self, dir_save:str, case:str):
        self.dir_save = dir_save
        self.case = case
        
    def save_model(self, model)-> None:
        ''' method to save DL  (pytorch) model '''
        
        model_name = 'dlModelWithoutHyperOpt'+ self.case
        
        # save the entire NN's model as a pickle file-.
        filemod = self.dir_save+ '/'+ model_name+ '_pkl'+ '.pkl'
        with open(filemod, 'wb') as fdl: pickle.dump(model, fdl)
        fdl.close()

        # save the entire NN's model as a pytorch file-.
        filemod = self.dir_save+'/'+ model_name+ '_torch'+ '.pt'
        with open(filemod, 'wb') as fdl: torch_save(model, fdl)
        fdl.close()

        # save the NN's model as a dictionary object-.
        # save as dictionary state object (only parameters of the models,
        # in this case, when I try to open the model, I should to use the 
        # NN class also)-.
        filemod = self.dir_save+ '/'+ model_name+ '_sd'+ '.pt'  # state_dict                                   
        with open(filemod, 'wb') as fdl: torch_save(model.state_dict(), filemod)
        fdl.close()

        return None

    
    def save_loss(self, loss:dict)-> None:
        '''  method to train and validation loss '''

        loss_file = self.dir_save+ '/'+ 'loss' + self.case + '.p'
        # with open(loss_file, 'wb') as fl: file_loss= open(fl)

        file_loss = open(loss_file, 'wb')
        # dictionary2print-.
        
        pickle.dump(loss, file_loss)
        file_loss.close()

        return None
