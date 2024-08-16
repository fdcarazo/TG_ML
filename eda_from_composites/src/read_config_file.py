#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# class to read config file-.
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# start_date (Fr): Sun Mar 17 23:33:25 CET 2024-.-. 
# last_modify (Fr): -.
# last_modify (Arg): vie 16 ago 2024 08:54:45 -03-.
##
# ======================================================================= INI79

# print(dir()); input(1)

# Import packages/libraries/modules-.
from os.path import dirname as drn, realpath as rp
from typing import Dict
import yaml

# from ..utils.gen_tools import get_args  as ga
# from ..utils import gen_tools as gt

# main class-.
# 2BeMod: set attributes as private and use getter and setter methods,
#         also delete object after it is used-.
class Config():
    '''
    A class to load config file-.
    ...
    Attributes (only an example, 2BeCompleted-.)
    ----------
    name : str
        first name of the person
    Methods  (only an example, 2BeCompleted-.)
    -------
    info(additional=""):
        Prints the person's name and age.
    '''
    
    def __init__(self, cfg: Dict):
        ''' constructor '''
        
        self.config= cfg # in

        # out-.
        self.save = self.config['gen_options']['save']  # 2save or not the figures-.
        self.dir_save = self.config['gen_options']['dir_save']  # 2set path save figures-.
        self.dir_results = self.config['gen_options']['dir_results']  # 2set path save figures-.
        self.dir_logs = self.config['gen_options']['dir_logs']  # 2set path save figures-.
        self.sheet_name = self.config['gen_options']['sheet_name']
        self.vars_names = self.config['gen_options']['vars_names']
        self.feat_names = self.config['gen_options']['feat_names']
        self.targ_names = self.config['gen_options']['targ_names']

        # dataset block-.
        self.ds_path=self.config['dataset']['path']
        self.ds_file=self.config['dataset']['name']

        
if __name__=='__main__':
    config_file_path='/my_github/TG_ML/eda_from_composites/config_file.yaml'
    with open(config_file_path, 'r') as f: config = yaml.safe_load(f)
    cfg_obj = Config(config)
    print(cfg_obj.__dir__(), dir(cfg_obj), sep='\n'*2)
    print('{}{}'.format('\n'*3,cfg_obj.__dict__))
    print(cfg_obj.sca_targ, cfg_obj.mfn, sep='\n')
else:
    print('{0} imported as Module'.format(__file__.split('/')[-1]))

