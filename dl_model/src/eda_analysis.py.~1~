#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## EDA analysis -Exploratory Data Analysss-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date (Fr): Mon Mar 18 20:58:20 CET 2024-.
## last_modify (Fr): -.
##
## ======================================================================= INI79

## print(dir()); input(1)

## 1- IMPORT MODULUS ---.
## Import the required packages/libraries/modules-.
## 1-1- GENERAL MODULES -.
import seaborn as sns
import matplotlib.pyplot as plt

## main class-.
## 2BeMod: set attributes as private and use getter and setter methods,
##         also delete object after it is used-.
class Eda():
    '''
        A class to load PandasDataFrame-.
    ...
    Attributes (only an example, 2BeCompleted-.)
    ----------
    df : PandasDataFrame-.
        pandas data frame
    Methods  (only an example, 2BeCompleted-.)
    -------
    info(additional=""):
        Prints the person's name and age.
    '''
    
    def __init__(self,df):
        ''' constructor '''
        self.df=df # in

    def plot_corr(self,opt:int)->int:
        corr=self.df.corr()
        if opt==1:
            corr.style.background_gradient(cmap='coolwarm')
            ## corr = dataframe.corr()
            sns.heatmap(corr, 
                        xticklabels=corr.columns.values,
                        yticklabels=corr.columns.values,
                        annot=True)
            plt.show()
        elif opt==2:
            size=10
            fig,ax=plt.subplots(figsize=(size,size))
            ax.matshow(corr)
            plt.xticks(range(len(corr.columns)),corr.columns)
            plt.yticks(range(len(corr.columns)),corr.columns)
            plt.grid()
            plt.show()
        elif opt==3:
            g = sns.clustermap(self.df.corr(), 
                               method = 'complete', 
                               cmap   = 'RdBu', 
                               annot  = True, 
                               annot_kws = {'size': 8})
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=60);
            plt.show()
        return 0

    def describe(self):
        print(self.df.describe().transpose())

    ## add graphics of dsitributions, correlation, etc.
