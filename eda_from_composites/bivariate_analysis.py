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
    'axes.labelsize': 2,         # set xlabel and ylabel font size-.
    'axes.titlesize': 2,         # set xlabel and ylabel font size-.
    'axes.labelweight': 'bold',  # set xlabel and ylabel font weight-.
})

# set global font settings using rc parameters-.
sns.set(rc={
    'font.size': 8,           # set font size-.
    'font.family': 'serif',   # set font family-.
    'font.weight': 'normal',  # set font weight-.
    'axes.labelsize': 8
})


class Bivariate_Analysis_Plots(Data_Analysis):
    def __init__(self, df, features, targets):
        super().__init__(df, features, targets)

    def bivariate_plot(self, frac: int, class_name: str) -> None:
        '''
        bivariate plots using pairplot-.
        '''

        # exploratory plots-.
        print('From {0} class ploting features and targets values'
              ' distribution.'.format(class_name)
              )

        # join pairplots-.
    
        # sns.set_context('paper', rc={'axes.labelsize': 1})
        # sns.set_context('talk', font_scale=0.5)
        # sns.set_context('paper', rc={'axes.labelsize': 8})
        
        sns.pairplot(self.df[self.features+self.targets].sample(frac=frac),
                     corner=True,
                     plot_kws=dict(alpha=0.2, edgecolor='none'),
                     # kind='reg',
                     height=1,
                     aspect=1,
                     diag_kind='hist',
                     diag_kws=dict(multiple='stack', edgecolor='blue'))
        plt.tight_layout()
        plt.show()

        return None

    def feature_target_plot(self) -> None:
        '''
        features and targets bivariate plots-.
        ** NOTE: this procedure/function could be replaced by the previous one
        passing vars to plot as argument (features in one case and target in
        other)** -.
        '''
        
        # general options-.
        colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'black']
        symbols = ['<', 'o', '>', '^', 'p', 's', '*']
        vars_to_plot = [self.features, self.targets]  # [list, list]
        
        # exploratory plots-.
        print('Bivariate general plots')
        
        # FEATURES AND TARGETS bivariate plots-.
        nrows = len(self.features)
        fig = plt.figure(figsize=(12, 10))
        for i_plot, var_to_plot_x in enumerate(self.features):
            i_row = i_plot * (len(self.features)+1)
            for i_var in range(i_plot+1, len(self.features)+1):
                ij_plot = i_var + i_row
                # print(i_plot, i_var, var_to_plot_x, sep='\n')
                ax = fig.add_subplot(len(self.features)+1,
                                     len(self.features)+1,
                                     ij_plot)
                # if i_row <= len(self.features):
                plt.scatter(self.df[var_to_plot_x],  #.sample(frac=1),
                            self.df[self.features[i_var-1]],  # .sample(frac=1),
                            marker='o',  #, label=self.features[i_var-1],
                            color=colors[i_plot])
                ax.set_xlabel(var_to_plot_x)
                ax.set_ylabel(self.features[i_var-1])
                # plt.scatter(self.df[var_to_plot], np.full(len(self.df[var_to_plot]), 1),
                #             label=var_to_plot, color=colors[i_plot],
                #             marker=symbols[i_plot])
                ax.legend()
        '''
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        # sns.jointplot(data=self.df, x=self.df[self.features[0]], y=self.features[1])
        # sns.jointplot(data=self.df, x=self.df[self.features[1]], y=self.targets[4])
        plt.scatter(self.df[self.features[1]].sample(frac=1),
                    self.df[self.features[3]].sample(frac=1))
        '''
        plt.tight_layout()
        plt.show()
        return None
