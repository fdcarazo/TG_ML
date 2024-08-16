#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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


class Machine_Learning_Regression(Data_Analysis):
    def __init__(self, df, features, targets):
        super().__init__(df, features, targets)
        # self.features_log = features_log  # transf. feat. as log(features)-.
        # self.features_sqrt = features_sqrt # transf. feat. as sqrt(features)-.
        
    def get_edited_df(self, class_name: str) -> pd.DataFrame:
        '''
        get a DF edited, this inlude:
        1- categorical features: generation of dummies columns-.
        2- numerical features: add transformed features using log and sqrt-.
        '''

        # generation a new edited dataset-.
        print('From {0} generation a new dataset (1- encoding categorical '
              'features and transform numerical features using log and sqrt).'.
              format(class_name)
              )
        df_copied = self.df.copy(deep=True)

        # get transformed dataFrames-.
        features_num = df_copied[self.features].select_dtypes(include=['number'])  # float/int, etc.
        features_num_names = features_num.columns.to_list()
        log_df, sqrt_df = pd.DataFrame(), pd.DataFrame()
        
        # apply log and sqrt transformation at each column-.
        for col in features_num_names:
            log_df[str(col)+'_log'] = np.log(features_num[str(col)])
            sqrt_df[str(col)+'_sqrt'] = np.sqrt(features_num[str(col)])

        # get numerical columns/features-.
        # df.select_dtypes(exclude=["number","bool_","object_"])
        # df.select_dtypes(include=["number","bool_","object_"])
        # num_df_subset = df_copied.select_dtypes(include=['number'])   # float/int, etc.
        cat_df_subset = df_copied.select_dtypes(include=['object_'])  # categorical.

        if cat_df_subset.empty:
            list_features = [self.df[self.features], log_df, sqrt_df,
                             self.df[self.targets]]
        else:
            # categorical subset with OneHotEncode-.
            cat_df_subset_with_one_hot = pd.get_dummies(cat_df_subset)
            print(cat_df_subset_with_one_hot)
            list_features = [self.df[self.features], log_df, sqrt_df,
                             cat_df_subset_with_one_hot, self.df[self.targets]]
        print(list_features)
        print(log_df)
        print(sqrt_df)
        print(self.df[self.features])
        print(self.df[self.targets])
        input(3333)
        
        # join dataframes to built a GeneralDataFrame-.
        df_edited = pd.concat(list_features, axis=1)

        return df_edited

    # remove collinear Features-.
    def remove_collinear_features(self, x, threshold):
        '''
        Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
        Inputs: 
        threshold: any features with correlations greater than this value are removed
        
        Output:
        dataframe that contains only the non-highly-collinear features

        adapted from (source): machine-learning-project-walkthrough
                               /Machine Learning Project Part 1.ipynb
        '''
    
        # don't want to remove correlations between targets Energy Star Score-.
        y = self.df[self.targets]
        # x = self.df.drop(columns=[self.targets])
        x = self.df[self.features]
    
        # calculate the correlation matrix-.
        corr_matrix = x.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # iterate through the correlation matrix and compare correlations-.
        for i in iters:
            for j in range(i):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)
            
                # if correlation exceeds the threshold-.
                if val >= threshold:
                    # print the correlated features and the correlation value-.
                    # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                    drop_cols.append(col.values[0])

        # drop one of each pair of correlated columns-.
        drops = set(drop_cols)
        x = x.drop(columns=drops)
    
        # Add the score back in to the data
        x = pd.concat([x, y], axis=1)
               
        return x
