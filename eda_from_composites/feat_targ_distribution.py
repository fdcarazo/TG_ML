#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


class Feat_Targ_Distributions(Data_Analysis):
    def __init__(self, df, features, targets):
        super().__init__(df, features, targets)

    def plot_feat_target_dist(self, n_bins, class_name) -> None:
        ''' 
        unimodal features and targets values distribution-.
        '''
        # general options-.
        colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'black']
        symbols = ['<', 'o', '>', '^', 'p', 's', '*']
        vars_to_plot = [self.features, self.targets]  # [list, list]

        # exploratory plots-.
        print('From {0} class ploting features and targets values distribution.'.
              format(class_name))
        
        # join samples visualization-.
        for idx, list_to_plot in enumerate(vars_to_plot):  # 0: features, 1- targets-.
            nrows = len(list_to_plot)
            locals()['fig{0}'.format(str(idx))] = plt.figure(figsize=(12, 12))
            for i_plot, var_to_plot in enumerate(list_to_plot):
                i_row = 2*i_plot + 1
                ax = locals()['fig{0}'.format(str(idx))].add_subplot(2*len(list_to_plot),
                                                                     1,
                                                                     i_row)
                plt.scatter(self.df[var_to_plot], np.full(len(self.df[var_to_plot]), 1),
                            label=var_to_plot, color=colors[i_plot],
                            marker=symbols[i_plot])
                ax.legend()
                ax.set_ylabel(None)
                ax.set_xlabel(None)
                ax.set_yticks([1])
                ax = locals()['fig{0}'.format(str(idx))].add_subplot(2*len(list_to_plot),
                                                                     1,
                                                                     i_row+1)
                sns.histplot(self.df[var_to_plot],
                             ax=ax,
                             bins=n_bins,
                             label=var_to_plot,
                             color=colors[i_plot],
                             alpha=0.5)
                ax.legend()
                ax.set_ylabel('count', fontsize=8)
                ax.set_xlabel(None)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return None
