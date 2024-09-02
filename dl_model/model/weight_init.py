#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
##
# to initializae neurons weight
#
# start_date (Arg): dom 01 sep 2024 12:52:23 -03-.
# last_modify  (Arg): -.
##
# - =======================================================================INI79
# 1- import modulus-.
# general modules-.
# torch framework-.
from torch.nn.init import xavier_normal_ as normal_weights

# - =======================================================================END79


# - =======================================================================INI79
# auxiliaries classes and functions-.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('linear') != -1:
        normal_weights(m_weight, gain=1.0)

# - =======================================================================END79
