#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# SCRIPT: used in main & process_plot scripts-.
# @author: Fernando Diego Carazo (@buenaluna) -.
# start_date (Arg): mar 23 abr 2024 13:29:11 -03-.
# last_modify (Arg): mar 06 ago 2024 10:59:26 -03-.
##
# ======================================================================= INI79
# 1- include packages, modules, variables, etc.-.
import pandas as pd
# ======================================================================= END79

# ======================================================================= INI79
def load_ds(file_name: str, vars_names: list):  # load ds-.
    df = pd.read_csv(file_name, low_memory=False, usecols=vars_names)
    return df
