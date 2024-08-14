#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# script to load, analyze, plot, debug and curate datasets-.
# dataset: TG's curves of stalk and marc Pyrolysis-.
# Provided by Dra.- Ing. Ana Fernando - IIQ - CONICET - FI - UNSJ-.
#
# start_date: mar jul  5 13:11:15 -03 2022 -.
# last_modify: lun 12 ago 2024 14:12:43 -03 -.
##

# import modules/packages/libraries-.
import pandas as pd
import glob
import os
from aux_proc import check_reg_exp, plot_dp_vals
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from scipy import interpolate

# warning configuration-.
# - ==========================================================================79
import warnings
warnings.filterwarnings('ignore')


# clean terminal (XTERM in my GNU-Linux)-.
import subprocess as sp
sp.run(['clear'])

# folder/diretory in which are the datasets-.
root_path = '/home/fdcarazo/my_github/pyrolysis/trabajoPirolisis_IIQ/'+ \
    'tga_yield/ds2script1/'
ds_path = '/home/fdcarazo/ds/iiq_unsj/2022/'

# 1- =================================================================
# load files and built PandasDataFrames-.
# load files-.
files_list = list()  # empty list-.
files_list = glob.glob(os.path.join(str(ds_path), '*.csv'),
                       recursive='True')  # strObject-.
print('{0}{1} *.csv files was/were finded. Their names are: {0}'.
      format('\n', len(files_list)))
for i in files_list: print(i)

reg_exp = 'stalkmarc'  # string leave out in search-.

# built PandasDataFrames-.
pd_list_name = list()  # list with PandasDataFrames names-.
i = 0
for files_names in files_list:
    if check_reg_exp(reg_exp, files_names):  # don't load unlabeled data-.
        # print('Hi !!!!!')
        pass
    else:
        # print(check_reg_exp(reg_exp, files_names))
        # print(files_names)
        i += 1
        df_name = 'df_{0}'.format(i)
        # index_col= None or index_col=False don't work-. why?-.
        exec('df_' + str(i) + "= pd.read_csv(files_names, index_col=0)")
        # modify label columnas-.
        pd_list_name.append(df_name)

        print(pd_list_name)

# dataframes check-.
for idx, i_pd_name in enumerate(pd_list_name): print(eval(i_pd_name))
# - ==========================================================================79

# - ==========================================================================79
ndatas = 3  # numbers of rows to show with sample method-.
nl = '\n'  # newline character. Will be used with f-'s prints statements-.

DS_name = 'Biochar (mg) prediction of stalk and marc pyrolysis. ' + \
    'DataFrame provided by Dra. Ana Fernandez, IIQ-UNSJ-CONICET'
# 
print(f'Load, analyze, plot, debug and curation of '+
      f'del DS{nl}{DS_name}{nl}')

# - ==========================================================================79
# INI @ ANALYSIS_1 ----
# ''' @FDC-comment sampling in the preset order of the 3 (three) records-.
print('{0}ANALYSIS_1 {1}'.format('\n'*2, '*'*100))
print('{0}{1}Visualization in the order of DataSet (DS){0}'.
      format('\n', '\t /// << --- >> /// '))
for i in np.arange(len(pd_list_name)):
    print('{0}Visualization of {1} registers in DS {0}{2}'.
          format('\n', ndatas, eval(pd_list_name[i])[:ndatas]))
# @ ANALYSIS_1: -.
# END @ ANALYSIS_1 ----
# - ==========================================================================79

# - ==========================================================================79
# INI @ ANALYSIS_2 ----
# ''' @FDC-general info of all PandasDataFrame-.
print('{0}ANALYSIS_2 {1}'.format('\n'*2, '*'*100))
for i_pd_name in pd_list_name: print(eval(i_pd_name).info())

'''
if True:
    plt.plot(df_1.TGA, df_1.Temp, 'ro')
    plt.xlabel('TGA (mg)')
    plt.ylabel('T (K)')
    plt.legend(['train', 'test'])
'''

''' ANOTATION
==============
We see that there are columns/rows with values ​​that, although they are
not detected as null, are NULLs (strings) - columns Temp, TGA and
DrTGA for Stalk and Temp, TGA and DrTGA for Marc-. We visualize them.
Although the data loss is insignificant, we see in a visual encoding
the quantity and its distribution in the DS-.
'''
num_samples = 5000
if True:
    figchar = (1, 4, 12, 6)
    for i in range(len(pd_list_name)):
        print(pd_list_name[i])
        # plot_dp_vals(figchar, pd_list_name[i], eval(pd_list_name[i]), i)
        plot_dp_vals(figchar, eval(pd_list_name[i]).iloc[:, :], i,
                     num_samples)
    plt.show()

''' ANOTATION
==============
We see a random loss of little data, we proceed to delete them to
clean the dataset.
'''

# df_1[filter].astype(object).where(df_1.notna(), None)
# for i_pd_name in pd_list_name: print(eval(i_pd_name).info())

# porcentage of null registers in each PDF-.
for x in pd_list_name:
    print('Porcentage of null PDF {0} registers = {1:0.2f}{2}'.
          format(x,
                 (eval(x).isnull().sum()).sum()/np.prod(eval(x).shape)*100, ' %.'))
# print total number of registers equal to Null values-.
# eval(x).isna().any(axis=1) => create a boolen mask if any row value is equal zo NaN or Null-.
for x in pd_list_name: print(eval(x).loc[eval(x).isna().any(axis=1)].sum())  # it isn't working-.
for x in pd_list_name: print(eval(x).shape)

# delete Null and NaNs values-.
df_1 = df_1.dropna().sample(frac=1).reset_index(drop=True)
df_2 = df_2.dropna().sample(frac=1).reset_index(drop=True)

# df_stalk.sample(frac=1).reset_index(drop=True)
# df_marc.sample(frac=1).reset_index(drop=True)
for x in pd_list_name: print(eval(x).shape)

# check if there is/are som Null od NaN value-.
print('{0}'.format('\n'*2))
for x in pd_list_name: print(eval(x).loc[eval(x).isna().any(axis=1)])

# save PDF without Nulls/NaNs values (in csv file format)-.
df_1.to_csv(root_path+'stalk_depurado.csv')
df_2.to_csv(root_path+'marc_depurado.csv')

for x in pd_list_name: print(eval(x).shape)
# @ ANALYSIS_2: -.
# END @ ANALYSIS_2 ----
# - ==========================================================================79

# - ==========================================================================79
# INI @ ANALYSIS_3 ----
# @FDC-comment for each variable (feaeture or target) print min, max and mean values-.
li = ['Temp', 'TGA', 'DrTGA']  # variables-.
for i_var_name in (li):
    print('{0}'.format(df_1.groupby('vel_cal').
                       agg({str(i_var_name): ['max', 'min', 'mean']}).round(2)))
for i_var_name in (li):
    print('{0}'.format(df_2.groupby('vel_cal').
                       agg({str(i_var_name): ['max', 'min', 'mean']}).round(2)))
# print('{0}'.format('\n'*4)); input(567)
# @ ANALYSIS_3: -.
# END @ ANALYSIS_3 ----
# - ==========================================================================79

# - ==========================================================================79
# INI @ ANALYSIS_4 ----
# @FDC- Analyze the distribution, outliers and correlation of the variables
# 4-1- attributes of the two PDfs
for x in pd_list_name:
    sns.pairplot(eval(x),
                 hue='vel_cal',
                 markers=['o', 's', '*'],
                 corner=True)
# plt.show()

# 4-2- correlation-.
for x in pd_list_name: sns.pairplot(eval(x), hue='vel_cal')
# plt.show()

# 4-3- heatmap
''''
function to plot correlation between variables of each PDF-.
'''
fig_char = (1, 2, 12, 6)
nr, nc, fig_width, fig_height = fig_char
fig, ax = plt.subplots(nrows=nr, ncols=nc,
                       figsize=(fig_width, fig_height),
                       sharey=True
                       )
pdf_list = [df_1, df_2]
for i, x in enumerate(pdf_list):
    # ir, icol = rel_col_matr_list[icol]
    # corr_0
    corr_0 = x.corr()
    mask_0 = np.triu(np.ones_like(corr_0, dtype=bool))
    sns.heatmap(corr_0, mask=mask_0, cmap='RdYlGn',
                linewidths=0.30, annot=True, annot_kws={"size": 8},
                ax=ax[i-1]
                )
    ax[i-1].set_title(f'data_frame {i}')
    fig.suptitle(f'Correlation plot between variables of ' +
                 f'DataFrame {i}')

# plt.show()

metrics = df_1.drop(axis=1, labels=['vel_cal']).columns  # remove col 'vel_cal'
print(len(metrics))

for x in pdf_list:
    fig, axes = plt.subplots(nrows=1,
                             ncols=len(metrics),
                             figsize=(18, 9),
                             squeeze=False
                             )
    for i, column in enumerate(metrics):
        sns.boxplot(ax=axes[0, i],
                    data=x,
                    y=column,
                    x='vel_cal'
                    )
        sns.stripplot(ax=axes[0, i],
                      data=x,
                      y=column,
                      x='vel_cal'
                      )
        axes[0, i].set_title(column)

# for ax in axes:
# ax.tick_params(axis='y')

plt.show()
# @ ANALYSIS_4: -.
# END @ ANALYSIS_4 ----
# - ==========================================================================79
