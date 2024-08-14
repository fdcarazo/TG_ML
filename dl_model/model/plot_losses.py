#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# class to plot training and validation loss-.
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# start_date (Fr): Sun May 17 22:33:25 CET 2024-.
# last_modify (Fr): -.
# last_modify (Arg): jue 08 ago 2024 08:45:45 -03-.
##
# ======================================================================= INI79

# print(dir()); input(1)

# import packages/libraries/modules-.
import matplotlib.pyplot as plt
import os


class PlotLosses():
    '''
    class to plot plot training and validation loss
    '''
    def __init__(self):
        '''
        empty constructor
        '''
        pass

    def plot_loss(self, label_str: str, val: list, dir_save: str)-> None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        plt.plot(range(len(val['train_loss'])), val['train_loss'], 'r*',
                 label='train')
        plt.plot(range(len(val['val_loss'])), val['val_loss'], 'b<',
                 label='validation')
        plt.yscale('log')
        
        plt.grid(color='b', linestyle='--', linewidth='0.5')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.legend(loc='best')
        ax.set_title('loss using Feed Forward Neural Network (FFNN)')

        plt.show()
        fig.savefig(os.path.join(dir_save, 'trainAndValLoss.png'),
                    format='png', dpi=100)

        return None
