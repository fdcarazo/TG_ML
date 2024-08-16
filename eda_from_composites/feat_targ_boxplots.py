#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns

from exploratory import Data_Analysis

# set global font settings-.
plt.rcParams.update({
    'font.size': 10,             # set font size-.
    'font.family': 'serif',      # set font family-.
    'font.weight': 'normal',     # set font weight-.
    'axes.labelsize': 8,         # set xlabel and ylabel font size-.
    'axes.titlesize': 8,         # set xlabel and ylabel font size-.
    'axes.labelweight': 'bold',  # set xlabel and ylabel font weight-.
})

# set global font settings using rc parameters-.
sns.set(rc={
    'font.size': 8,          # set font size-.
    'font.family': 'serif',  # set font family-.
    'font.weight': 'normal'  # set font weight-.
})


class Feat_Targ_Boxplots(Data_Analysis):
    def __init__(self, df, features, targets):
        super().__init__(df, features, targets)

    def plot_boxplots(self):
        ''' 
        features and targets boxplots-.
        '''
        
        # general options (None in this case)-.
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(211)
        
        # exploratory plots-.
        print('Features and Targets Boxplots')
        
        # features boxplots-.
        self.df[self.features].boxplot(ax=ax)
        ax.set_xlabel('Features', fontsize=10)
        ax = fig.add_subplot(212)
        # targets boxplots-.
        self.df[self.targets].boxplot(ax=ax)
        ax.set_xlabel('Targets', fontsize=10)
        
        # plt.title('Features and Targets Boxplot')
        plt.legend()
        plt.show()
