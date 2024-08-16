#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# SCRIPT: used in main & process_plot scripts-.
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# last_modify: Sun Mar 17 22:47:21 CET 2024-.
##

# ======================================================================= INI79
# 1- include packages, modules, variables, etc.-.
import pandas as pd

# ======================================================================= END79


# ======================================================================= INI79
def load_ds(file_ext: str, file_name: str, sheet_name: str,
            vars_names: list):  # load ds-.
    '''
    procedure to load dataset-.
    '''
    if file_ext == 'xls' or file_ext == 'xlsx':
        df = pd.read_excel(file_name,
                           sheet_name=sheet_name,
                           header=None,
                           names=vars_names)
    elif file_ext == 'csv':
        df = pd.read_csv(file_name)
        
    return df
