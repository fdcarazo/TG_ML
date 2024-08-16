#!/usr/env/python3
#-*- coding: utf-8 -*-

import pandas as pd
from exploratory import Data_Analysis


class Outliers(Data_Analysis):
    def __init__(self, df, features, targets):
        super().__init__(df, features, targets)

    def detect_remove_outliers(self) -> pd.DataFrame:
        '''
        to find outliers-.
        '''

        df_without_outliers = pd.DataFrame(columns=self.features+self.targets)

        for col in list(df_without_outliers.columns):
            firsth_quartile = self.df[col].describe()['25%']
            third_quartile = self.df[col].describe()['75%']
            iqr = firsth_quartile - third_quartile  # interquartile range-.
            index = self.df[(self.df[col] > (firsth_quartile-3*iqr)) &
                            (self.df[col] < (third_quartile+3*iqr))]
            if len(index) == 0:
                pass
            else:
                df_without_outliers[col] = self.df[col][index]
            
        return df_without_outliers
