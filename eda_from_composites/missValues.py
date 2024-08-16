#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
import pandas as pd
from exploratory import Data_Analysis


class Missing_Values(Data_Analysis):
    def __init__(self, df, features, targets):
        super().__init__(df, features, targets)

    def missing_values_table(self):
        '''
        function to calculate missing values by column
        source: https://stackoverflow.com/questions/26266362/
        how-do-i-count-the-nan-values-in-a-column-in-pandas-dataframe/39734251#39734251
        '''
        
        # total missing values-.
        mis_val = self.df.isnull().sum()
        
        # percentage of missing values-.
        mis_val_percent = 100 * mis_val / len(self.df)
        
        # make a table with the results-.
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # rename the columns-.
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})
        
        # sort the table by percentage of missing descending-.
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
                '% of Total Values', ascending=False).round(1)
        
        # print some summary information-.
        print('Your selected dataframe has {0} columns. {1}There are ' \
              '{2} columns that have missing values.'. \
              format(str(self.df.shape[1]),
                     '\n',
                     str(mis_val_table_ren_columns.shape[0])))
        
        # return the dataframe with missing information-.
        return mis_val_table_ren_columns

    def columns_porc_missing(self, perc: int, missing_df) -> pd.DataFrame:
        '''
        get the columns with > perc% missing-.
        '''

        # missing_df = self.missing_values_table(self.df)
        missing_columns = list(missing_df[missing_df['% of Total Values'] > perc].index)
        print('We will remove {0} columns.'.format(len(missing_columns)))
        df_droped = self.df.drop(columns=missing_columns, axis=1)
        
        return df_droped

    
