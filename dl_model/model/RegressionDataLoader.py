#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# class to define torch.dataloader object-.
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
#
# start_date (Fr): Tue Mar 19 09:54:11 CET 2024-.
# last_modify (Fr): -.
# last_modify (Arg): jue 08 ago 2024 08:41:36 -03-.
##
# ======================================================================= INI79

# print(dir()); input(1)

# import packages/libraries/modules-.
from torch.utils.data import DataLoader
import numpy as np

# - =======================================================================INI79
# Definir la clase del conjunto de datos
class RegressionDataLoader(DataLoader):
    def __init__(self, dataset, b_s:int, shuff:True):
        self.dataloader = DataLoader(dataset, batch_size=b_s,
                                     num_workers=2,
                                     persistent_workers=True,
                                     shuffle=shuff)
# - =======================================================================END79
