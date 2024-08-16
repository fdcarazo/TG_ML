#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# script to do a general EDA analysis in DataSet used to train and validate
# ANN's  model to predict ELASTIC EFFECTIVE properties in fiber's composites-.
#
# synthetic DATASET obtained using PMM's Barberos model-.
# provided by Dr. Ing. Dario Barulich - CONICET - UTN - FRC -.
#
# @author: Fernando Diego Carazo (@buenaluna) -.
#
# start_date (Arg): dom jun 16 10:22:29 -03 2024-.
# last_modify (Arg): -.
#
# ======================================================================= END79

# ======================================================================= INI79

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

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


class Data_Analysis():
    def __init__(self, df, features, targets):
        self.df = df  # .sample(frac=1)
        self.features = features
        self.targets = targets

    def info_and_descriptive_statistics(self) -> None:
        '''
        basic info (column data types and non-missing values)-.
        &
        decriptive statistics (statistics for each column)-.
        '''
        # basic info == column data types and non-missing values-.
        print('{0}Data Types and Missing Values {0}{1}'.
              format('\n', self.df.info()))
        # decriptive statistics-.
        print('{0}DataFrame description (columns){0}{1}'.
              format('\n', self.df.describe()))
        return None

    # Exploratory Data Analysis (EDA)-.
    def univ_hist_plots(self, opt: str) -> None:
        '''
        Features and Targets Univariate distribution plots-.
        '''

        # general options-.
        colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'black']
        symbols = ['<', 'o', '>', '^', 'p', 's', '*']
        vars_to_plot = [self.features, self.targets]  # [list, list]

        # Univariate distribution plots of Features and Targets-.
        print('Univariate distribution plots of Features and Targets.')

        # join samples visualization-.
        for idx, list_to_plot in enumerate(vars_to_plot):  # 0: features, 1- targets-.
            nrows = len(list_to_plot)
            plt.style.use('fivethirtyeight')
            plt.style.use('fivethirtyeight')
            locals()['fig{0}'.format(str(idx))] = plt.figure(figsize=(13, 5))
            for i_plot, var_to_plot in enumerate(list_to_plot):
                i_col = i_plot + 1
                n_bins = 10
                ax = locals()['fig{0}'.format(str(idx))].add_subplot(1,
                                                                     len(list_to_plot),
                                                                     i_col)
                # histplot-.
                if opt == 'distribution':
                    sns.histplot(self.df[var_to_plot],
                                 ax=ax,
                                 bins=n_bins,
                                 label=var_to_plot,
                                 color=colors[i_plot],
                                 alpha=0.5)
                    ax.set_xlabel(var_to_plot, fontsize=10)
                    ax.set_ylabel('count', fontsize=10)
                elif opt == 'var_stat':
                    self.df[var_to_plot].describe().\
                        drop(labels=['count']).plot(kind='barh',
                                                    label=var_to_plot,
                                                    color=colors[i_plot],
                                                    alpha=0.5)
                    ax.set_xlabel(var_to_plot, fontsize=10)
                    # ax.set_ylabel('count', fontsize=8)
                ax.legend()
            plt.legend(fontsize=8)
            plt.tight_layout()
            
            plt.show()
            
            plt.close(locals()['fig{0}'.format(str(idx))])
            
        return None
    
    def features_bivariate(self, class_name) -> None:
        ''' 
        features relations (features bivariate plots)-.
        '''
        # general options-.
        colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'black']

        # exploratory plots-.
        print('From {0} class ploting features values relations '.
              format(class_name))
            
        # FEATURES samples visualization-.
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
        print('estoy aca')
                
    def plot_corr(self, opt: int) -> int:
        '''
        plot correltion between variables (features + targets)-.
        '''
        
        corr = self.df.corr()
        if opt == 1:
            corr.style.background_gradient(cmap='coolwarm')
            # corr = dataframe.corr()
            sns.heatmap(corr,
                        xticklabels=corr.columns.values,
                        yticklabels=corr.columns.values,
                        annot=True)
            plt.show()
        elif opt == 2:
            size = 10
            fig, ax = plt.subplots(figsize=(size, size))
            ax.matshow(corr)
            plt.xticks(range(len(corr.columns)), corr.columns)
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.grid()
            plt.show()
        elif opt == 3:
            print(2222222222)
            g = sns.clustermap(self.df[self.features+self.targets].corr(),
                               method='complete',
                               cmap='RdBu',
                               annot=True,
                               annot_kws={'size': 8})
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=60)
            plt.show()
        return 0
    
    def transform_df(self, class_name) -> pd.DataFrame:
        '''
        transform numerical features applying log and sqrt. To check if the
        linear correlations between features and the score increases-.
        '''
        
        # transform numerical features  to calculate the correlation
        # with targets ()-.
        
        if 'Unnamed: 0' in self.features:
            features_num = self.df.drop(columns=['Unnamed: 0']+self.targets). \
                select_dtypes('number')  # 'float'
        else:
            print('entre')
            print(self.df.columns)
            print(self.targets)
            input(1111111)
            features_num = self.df.drop(self.targets, axis='columns').\
                select_dtypes('number')  # 'float'
            print('sali')
            
        print(' Print from {0}.{1}Numerical Features are {1}{2}'.
              format(class_name, '\n', features_num))
        
        # features_num_log = pd.DataFrame(columns=features_num.columns)
        # features_num_sqrt = pd.DataFrame(columns=features_num.columns)
        features_num_log, features_num_sqrt = pd.DataFrame(), pd.DataFrame()

        # appy log and sqrt transformation at each column-.
        for col in features_num:
            features_num_log[col+'_log'] = np.log(features_num[col])
            features_num_sqrt[col+'_sqrt'] = np.sqrt(features_num[col])
        # print(features_num_log, features_num_sqrt, sep='\n')

        # correlation using transformed dataFrame-.
        corr_sqrt = pd.concat([features_num_sqrt, self.df[self.targets]],
                              axis=1).corr()
        corr_log = pd.concat([features_num_log, self.df[self.targets]],
                             axis=1).corr()
        
        return corr_sqrt, corr_log
    
    def correlation(self, class_name):
        '''
        calculate and plot correlation matrix (features + targets)-.
        '''

        # get transformed dataFrames-.
        sqrt_df, log_df = self.transform_df(class_name)
        
        # correlation matrix (using features without transform)-.
        corr = self.df[self.features + self.targets].corr()
        print('correlation matrix')
        print(corr)

        # calculate and print correlation between features and each target-.
        for targ in self.targets:
            print(self.df[self.features+self.targets].corr()[targ].
                  sort_values())

        # correlation using transformed dataFrame-.
        corr_sqrt, corr_log = sqrt_df.corr(), log_df.corr()

        # plot correlations matrix using heatmap-.
        if 'Unnamed: 0' in self.features:
            mask = np.triu(np.ones_like(self.df.drop(columns=['Unnamed: 0']).corr(),
                                        dtype=np.bool))
        else:
            mask = np.triu(np.ones_like(self.df.corr(),dtype=np.bool))

        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', square=True,
                    vmin=-1.0, vmax=1.0, mask=mask)
        plt.title('correlation matrix using features in original scale')
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(corr_log, annot=True, cmap='coolwarm', square=True,
                    vmin=-1.0, vmax=1.0, mask=mask)
        plt.title('correlation matrix using features in log scale')
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(corr_sqrt, annot=True, cmap='coolwarm', square=True,
                    vmin=-1.0, vmax=1.0, mask=mask)
        plt.title('correlation matrix using features in sqrt scale')
        plt.show()

        # FEATURES CORRELATED with EACH TARGET-.
        # https://github.com/PacktPublishing/Building-Statistical-Models-in-Python
        # source: https://github.com/PacktPublishing/
        # Building-Statistical-Models-in-Python/blob/main/chapter_7/
        # 2_feature_selection.ipynb
        # Must Be Completed (MBC) with 'sqrt_df' and 'log_df' to check if trend
        # and values change-.
        for target in self.targets:
            fig = plt.figure(figsize=(4, 10))
            heatmap = sns.heatmap(
                self.df.corr()[[target]].
                sort_values(by=target, ascending=False).drop(self.targets),
                vmin=-1,
                vmax=1,
                annot=True,
                cmap='coolwarm')
            heatmap.set_title(
                f'Features Correlated with {target}',
                fontdict={'fontsize': 8},
                pad=12)
            # fig = heatmap.get_figure()
            # fig.savefig('correlation_list.png', dpi=300)

        
    def classical_inferential_analysis(self, class_name):
        '''
        classical statistical inferentials analysis using statsmodels-.
        '''
        
        print('Classical statistical inferentials analysis from {0}'.
              format(class_name))
        for target in self.targets:
            print(f'Classical regression analysis of {target}:')
            formula = f"{target} ~ {' + '.join(self.features)}"
            model = smf.ols(formula, data=self.df).fit()
            print(model.summary())
    
    def bayesian_inferential_analysis(self, class_name):
        '''
        bayesian statistical inferentials analysis using statsmodels-.
        '''
        print('Bayessian statistical inferentials analysis from {0}'.
              format(class_name))
        print(sm.__version__)
        for target in self.targets:
            print(f'Bayesian regression analysis of {target}:')
            formula = f"{target} ~ {' + '.join(self.features)}"
            # from statsmodels.formula.api import BayesMixedGLM
            # model = BayesMixedGLM(formula, data=self.df).fit()
            model = smf.bayes_mixed_gl(formula, data=self.df).fit()
            print(model.summary())

    def logistic_inferential_analysis(self, class_name):
        '''
        logistic statistical inferentials analysis using statsmodels-.
        '''
        print('Logistic statistical inferentials analysis from {0}'.
              format(class_name))
        print(sm.__version__)
        for target in self.targets:
            print(f'Logistic regression analysis of {target}:')
            formula = f"{target} ~ {' + '.join(self.features)}"
            model = smf.logit(formula, data=self.df).fit()
            print(model.summary())
