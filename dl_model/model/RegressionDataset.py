#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# class to define de torch.Dataset object-.
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# start_date (Fr): Sun Mar 17 23:33:25 CET 2024-.
# last_modify (Fr): -.
# last_modify (Arg): jue 08 ago 2024 08:39:07 -03-.
##
# ======================================================================= INI79

# print(dir()); input(1)

# import packages/libraries/modules-.
import torch
from torch.utils.data import Dataset

# - =======================================================================INI79
# define the class of dataset-.
class RegressionDataset(Dataset):
    def __init__(self, features, targets):
        # self.features= features.clone().detach().requires_grad_(True)  # is it necessary GRAD? 
        # self.targets= targets.clone().detach().requires_grad_(True)  # is it necessary GRAD?
        self.features = torch.FloatTensor(features)  # torch.FloatTensor == float32-.
        self.targets = torch.FloatTensor(targets)  # torch.FloatTensor == float32-.

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # x= self.features[index].clone().detach().requires_grad_(True)
        # y= self.targets[index].clone().detach().requires_grad_(True)
        return self.features[idx], self.targets[idx]
# - =======================================================================END79
