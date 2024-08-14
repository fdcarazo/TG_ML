#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# script to visualize and assemble csv files to process the data provided by
# Prof. Dra.- Anabel Fernandez - IIQ - CONICET - FI - UNSJ-.
#
# start_date (Arg): lun 04 jul 2022 09:50:29 -03 -.
# last_modify (Arg): mar 13 ago 2024 20:31:54 -03 -.
##


# import modules/packages/libraries-.
import pandas as pd
import subprocess as sp
import numpy as np
import os

import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.interpolate import interp1d  # , Rbf-.

from my_deriv import calcDeriv

# wraning configurations-.
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# clean terminal (XTERM in my GNU-Linux)-.
sp.run(['clear'])

# path (folder + file) with dataset (excel file provided by Dras. AF & RR)-.
# root_path = '/home/fcarazo/diploDatos2022/trabajoPirolisis_IIQ/'
root_path = os.getcwd()+'/'
ds_path = '/home/fcarazo/ds/iiq_unsj/2022/'
fileName = 'DatosParaRNA_Orig.xlsx'

# 1- before starting, check that the directory and the file exist,
#    (pending task)-.

# header[0,1]  ==> list with row positions to be combined into a multiple
#                  index-.
heat_rate = ['5 Kmin', '10 Kmin', '15 Kmin']  # heating_rate-.
df_names, df_list = list(), list()

for i, i_heat_rate in enumerate(heat_rate):
    # print(i_heat_rate, i)
    df_name = ''.join(['df', i_heat_rate.replace(' ', '')])
    df_names.append(df_name)
    df = pd.read_excel(ds_path + fileName, header=[0, 1],
                       sheet_name=i_heat_rate)
    # delete the first MultiIndex row:: 'Stalk' y 'Marc'-.
    df = df.droplevel(level=0, axis=1)
    # delete the second column "Index" (Excel indexes)-.
    # df = df.drop(['index'], axis=1)  # it doesn't work-.
    # df = df.reset_index(drop=True, inplace=True)

    if i_heat_rate.replace(' ', '') == '5Kmin':
        df.loc[:, 'vel_cal'] = 5
    elif i_heat_rate.replace(' ', '') == '10Kmin':
        df.loc[:, 'vel_cal'] = 10
    elif i_heat_rate.replace(' ', '') == '15Kmin':
        df.loc[:, 'vel_cal'] = 15

    df_list.append(df)

    '''
    print('Content of DataFrame {0}{1}'.format('\n', df_name))
    # dfName.index.name = ''.join(['df', iVelEnfr.replace(' ', '')])
    print('DataFrame name {0}{1}'.format(df_list[i].index.name, '\n'))
    '''
    
# eliminate row 0 -zero, i.e. first- with the units of each PandasDataFrame-.
df_list = list(map(lambda x: x.drop(labels=0, axis=0).reset_index(), df_list))
# eliminate column "index" -excel indexes- of each PandsDataFrame-.
df_list = list(map(lambda x: x.drop(['index'], axis=1), df_list))

'''
print(df_list)
print('{0}{1}'.format('\n', len(df_list)))
print([print(df_list[i].columns) for i in range(len(df_list))])
input(11)
'''

# INI -- BLOCK_1 =============================================================79

# print(df_list[0].columns)
# print(df_list[0].iloc[:, 0], df_list[0].columns, sep= '\n'); input(11)

x, x_der, y, y_der = list(), list(), list(), list()
for idx in range(len(df_list)):  # for each heating rate calculate DTG-.
    # print(len(df_list), df_list[idx].columns, df_list[idx].vel_cal.unique())

    # df = df_list[0].drop_duplicates(subset=['Temp']).sample(n=1000)
    # df = df_list[idx].iloc[:, 0:4].sample(n=len(df_list[idx]))  # columns 0, 1, and 2 @ STALK-.
    df = df_list[idx].iloc[:, 3:6].sample(n=len(df_list[idx]))  # columns 3, 4, and 5 @ MARC-.
    # print(df)
    # print('shape: {0}'.format(df.shape))
    # df = df.drop_duplicates(subset=['TGA'])
    # df = df.drop_duplicates(subset=['Temp'])
    # print('idx = {0} -- Shape: {1}'.format(idx, df.shape))

    df.dropna(inplace=True)  # delete a row with NaN/s in each column/s-.

    '''
    print('shape: {0}'.format(df.shape))
    input(11)
    '''
    
    x2 = df.iloc[:, 0].to_numpy().astype(float)  # column 0 ==> T (K)
    tga = df.iloc[:, 1].to_numpy().astype(float)  # column 1 ==> TGA (mg)

    unique, indices = np.unique(x2, return_index=True)
    # x2= x2[indices]
    # print(np.shape(x2), np.shape(np.unique(x2)))
    # print(x2[indices], tga[indices])
    # input(99)

    fun = interpolate.interp1d(x=x2[indices], y=tga[indices], kind='cubic')
    # print(np.shape(x2[indices]), np.shape(np.unique(x2[indices]))); input(88)
    # x2 = df_list[0].iloc[:, 0]  # np.linspace(start=0, stop=4, num=1000)
    x3 = np.linspace(np.min(x2[indices]), np.max(x2[indices]), 100)
    y3 = fun(x3)  # x2[indices]
    # print(y3); input(66)

    tck = interpolate.splrep(x3, y3, s=0, k=3)
    x4 = np.linspace(np.min(x2[indices]), np.max(x2[indices]), 5000)
    x4_der = np.linspace(np.min(x2[indices]), np.max(x2[indices]), 5000)
    ynew_4 = interpolate.splev(x4, tck, der=0)
    ynew_4_der = interpolate.splev(x4_der, tck, der=1)
    # print(x4); input(44)
    # print(ynew_4); input(33)

    x.append(x4)
    x_der.append(x4_der)
    y.append(ynew_4)
    y_der.append(ynew_4_der)

# tck_new, u = interpolate.splprep([x3, y3], s=0, per=True)
# xnew_4_1, ynew_4_1 = interpolate.splev(x4, tck_new, der=0)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax1 = ax.twinx()

# custom_cycler = (cycler(ls=['-', '--', ':']))
# plt.rc('axes', prop_cycle=custom_cycler)

l_c = ['b', 'r', 'k']
l_c_der = ['g', 'm', 'c']
l_s = ['--', '-.', '-']
l_l = ['5', '10', '15']

for idx in range(len(x)):
    # ax.plot(x2, tga, label='true', c= 'b', marker='o')
    ax.plot(x[idx], y[idx], label='{0}'.format(l_l[idx]), c=l_c[idx],
            ls=l_s[idx])
    ax1.plot(x_der[idx], -1.0*y_der[idx], label='{0}'.format(l_l[idx]),
             c=l_c_der[idx], ls=l_s[idx])
    # ax.plot(xnew_4_1, ynew_4_1, label='splrep_spline_interp_splev', c= 'm')
    # ax2= ax.twinx()
    # ax2.scatter(x44, ynew_4_der, label='der(spline_interp_splev)/dT', c= 'b', marker='o')
    # ax.plot(x4, ynew_6, label='d(spline_interp_splev)/dx', c= 'g', marker= '+')
    # ax.plot(xnew_5, ynew_5, label='spline_interp_splev', c= 'm', marker= '*')

# ax.set_prop_cycle(custom_cycler)
# ax1.set_prop_cycle(custom_cycler)

# ax.tick_params(labelcolor='red', color='red', grid_color='red')
# ax.tick_params(left=True, labelcolor='green', color='green', grid_color='green')

# ax.legend()
ax.set_xlabel(r'$T\ (K)$')
ax.set_ylabel(r'$TGA\ (mg)$')
ax1.set_ylabel(r'$\frac{d(TGA)}{dT}\ (mg/K)$')

ax.set_xticks(np.arange(np.min(x[0]), np.max(x[0]), step=50))  # set label locations.

title = 'Marc'
# title = 'Stalk'
ax.set_title(title)
ax.legend(title=r'$TGA\ (mg)$', loc='best')
ax1.legend(title=r'$\frac{d(TGA)}{dT}\ (mg/T)$', loc='best')
# ax1.legend(loc='best')
plt.grid()

ax.grid(which='both', alpha=1, visible=True)

plt.show()
# input(123)
# END -- BLOCK_1 =============================================================79

# INI -- BLOCK_2 =============================================================79
# subplots-.
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(16, 8))
l_c = ['b', 'r', 'k']
l_c_der = ['g', 'm', 'c']
l_s = ['--', '-.', '-']
l_l = ['5', '10', '15']
ax = [ax0, ax1, ax2]
for idx in range(len(x)):
    # ax.plot(x2, tga, label='true', c= 'b', marker='o')
    ax[idx].plot(x[idx], y[idx], label='{0}'.format(l_l[idx]), c=l_c[idx],
                 ls=l_s[idx])
    ax_1 = ax[idx].twinx()
    ax_1.plot(x_der[idx], y_der[idx], label='{0}'.format(l_l[idx]),
              c=l_c_der[idx], ls=l_s[idx])
    ax[idx].set_xlabel(r'$T\ (K)$')
    ax[idx].set_ylabel(r'$TGA\ (mg)$')
    ax_1.set_ylabel(r'$\frac{d(TGA)}{dT}\ (mg/K)$')
    
    ax[idx].set_xticks(np.arange(np.min(x[0]), np.max(x[0]), step=50))  # set label locations.
    
    title = 'Marc'
    ax[idx].set_title(title)
    ax[idx].legend(title=r'$TGA\ (mg)$', loc='best')
    ax[idx].grid(which='both', alpha=1, visible=True)
    ax_1.legend(title=r'$\frac{d(TGA)}{dT}\ (mg/T)$', loc='best')
    # ax1.legend(loc='best')
plt.grid()
plt.show()

# input(11223344)
# END -- BLOCK_2 =============================================================79

# INI -- BLOCK_3 =============================================================79
'''
# commented 13/08/2024-.
idx = df_list[0].iloc[:-1, 1] == df_list[0].iloc[:-1, 1]  # column 1 ==> TG  (mg)
# iiddxx = np.where(np.unique(df_list[0].iloc[:, 1])) and np.where(np.unique(df_list[0].iloc[:, 0]))
iiddxx = np.where(np.unique(df_list[0].iloc[:, 0]))
# print(iiddxx)
idx = np.append(idx, True)  # add 1 because idx has one less element-.
# print(idx, type(idx), np.shape(idx), np.shape(iiddxx)); input(11)

# tga = np.nan_to_num(df_list[0].iloc[:,1])
x2 = df_list[0].iloc[:, 0].to_numpy()[iiddxx].astype(float)  # column 0 ==> T (K)
tga = df_list[0].iloc[:, 1].to_numpy()[iiddxx].astype(float)  # column 1 ==> TG (%)

# print(np.shape(x2), np.shape(np.unique(x2)))  # ; input(22)
iiddxx = np.where(np.unique(x2))
x2 = x2[iiddxx]
tga = tga[iiddxx]
# print(np.shape(x2), np.shape(np.unique(x2)))
# print(np.shape(tga))

# input(99)

iiddxx = np.where(np.unique(tga))

x2 = x2[iiddxx]
tga = tga[iiddxx]
'''

df = df_list[0].drop_duplicates(subset=['Temp'])
# print('shape: {0}'.format(df.shape))
df = df.drop_duplicates(subset=['TGA'])
# print('shape: {0}'.format(df.shape))

x2 = df.iloc[:, 0].to_numpy().astype(float)  # column 0 ==> T (K)
tga = df.iloc[:, 1].to_numpy().astype(float)  # column 1 ==> TGA (mg)

# print(np.shape(tga), np.shape(np.unique(tga))); input(22)

# iiddxx = np.where(np.unique(df_list[0].iloc[:, 0]))
# tga = tga[iiddxx].astype(float)  # column 1 ==> TGA (mg)
# x2 = x2[iiddxx].astype(float)  # column 0 ==> T (K)

'''
print(tga, type(tga), np.shape(tga)); input(22)
print(np.shape(tga)); input(33)
print(np.shape(tga), np.shape(np.unique(tga))); input(44)
print(x2, type(x2), np.shape(x2)); input(55)
'''

# print(df_list[0].shape)
# tga = df_list[0].index.isin(idx)
# x2 = np.nan_to_num(df_list[0].loc[:, 0][idx]).astype(float)

# print(np.shape(tga))#, np.shape(x2))
# input(33)

# tga = np.nan_to_num(df_list[0].iloc[:, 1])

drtg_true = np.diff(tga)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
drtg_true = np.append(drtg_true, 0.0)

# print(df_list[0].iloc[:, 0].shape, np.shape(drtg_true))
# print(drtg_true)
# print(df_list[0].iloc[:,0].shape, np.shape(drtg_true))

# ax.scatter(df_list[0].iloc[:,0], drtg_true, label='true',
#            c= 'indianred', marker= '<', alpha= 0.5)
# ax.scatter(x2, drtg_true, label='true', c= 'indianred',
#            marker= '<', alpha= 0.5)

# input(44)

# x2 = np.nan_to_num(df_list[0].iloc[:, 0][0:10000]).astype(float)
# y2 = np.nan_to_num(drtg_true[0:10000]).astype(float)

# iiddxx = np.where(np.unique(drtg_true))
x2 = x2
y2 = drtg_true

# print(np.shape(x2), np.shape(np.unique(x2)))
# print(np.shape(y2), np.shape(np.unique(y2)))
# print(x2 == np.nan)
# print(y2 == np.nan)
# input(55)

# print(x2, y2)
# print(np.shape(np.unique(x2)), np.shape(y2))

# fun = interp1d(x=np.unique(x2), y=y2[:-2], kind=5)
iiddxx = np.where(np.unique(x2))

'''
print(np.unique(x2[iiddxx]))
print(np.unique(x2, return_index=True, return_inverse=True, return_counts=True,
                axis=None))
'''

unique, indices = np.unique(x2, return_index=True)
# x2= x2[indices]
'''
print(np.shape(x2), np.shape(np.unique(x2)))
input(99)
'''

fun = interp1d(x=x2[indices], y=y2[indices], kind='next')
# print(np.shape(x2[indices]), np.shape(np.unique(x2[indices])))  # ; input(88)
# x2 = df_list[0].iloc[:, 0]  # np.linspace(start=0, stop=4, num=1000)
x3 = np.linspace(np.min(x2[indices]), np.max(x2[indices]), 10000)
y3 = fun(x3)  # x2[indices]

tck = interpolate.splrep(x3, y3, s=0.0015, k=2, per=True)
tck_new, u = interpolate.splprep([x3, y3], s=0, per=True)

x4 = np.linspace(np.min(x2[indices]), np.max(x2[indices]), 500)
x44 = np.linspace(np.min(x2[indices]), np.max(x2[indices]), 100)
ynew_4 = interpolate.splev(x4, tck, der=0)
ynew_4_der = interpolate.splev(x44, tck, der=1)
ynew_6 = interpolate.splev(x4, tck, der=1)

xnew_5, ynew_5 = interpolate.splev(x4, tck_new, der=0)

'''
print(tck, ynew_4, sep='\n')
print(np.shape(x4), np.shape(ynew_5), sep='\n')
'''

# print(np.shape(x3), np.shape(y3), sep='\n'); input(111)
# ax.plot(x3, y3, label='linear_interp', c= 'b', marker= 'o')
# ax.scatter(x3, tck, label='spline_interp_splrep', c= 'm', marker= '*')
ax.plot(x4, ynew_4, label='spline_interp_splev', c='r')
ax2 = ax.twinx()
ax2.scatter(x44, ynew_4_der, label='der(spline_interp_splev)/dT', c= 'b', marker='o')
# ax.plot(x4, ynew_6, label='d(spline_interp_splev)/dx', c= 'g', marker= '+')
# ax.plot(xnew_5, ynew_5, label='spline_interp_splev', c= 'm', marker= '*')

ax.legend()
ax.set_xlabel('T (K)')
ax.set_ylabel('DrTG (d(TGA)/dT) (\%/T)')
title = 'DrTG (d(TGA)/dT) (mg/T)'
ax.set_title(title)
plt.legend(loc='best')
plt.grid()
plt.show()

# input(66)
# END -- BLOCK_3 =============================================================79


# INI -- BLOCK_4 =============================================================79
for idx in range(len(df_list)):  # for each heating rate calculate DTG-.
    df = df_list[idx].iloc[:, 3:6].sample(n=len(df_list[idx]))  # columns 3, 4, and 5 @ MARC-.
    # df = df_list[0].iloc[:, 3:6].sample(n=len(df_list[0]))  # columns 3, 4, and 5 @ MARC-.
    df.dropna(inplace=True)

    # calculate derivative using my own class and function/procedure-.
    print(df.columns, df.shape, sep='\n')
    calcDeriv_obj = calcDeriv(df, 'Temp', 'TGA', 1.0)  # 1.0 == dt ($\Delta t$ between each $T_{registered}$)
    calcDeriv_obj.calc_and_plot_Drtg()
    
# END -- BLOCK_4 =============================================================79

# INI -- BLOCK_5 =============================================================79
# built lists DataSet-.
def built_dataset(df_l: list):
    dfs_stalk = list()
    dfs_marc = list()
    for idx, i_df_name in enumerate(df_l):
        dfs_stalk.append(i_df_name.iloc[0:, [0, 1, 2, 6]])
        dfs_marc.append(i_df_name.iloc[0:, [3, 4, 5, 6]])
        # print(idx, type(i_df_name))
    return dfs_stalk, dfs_marc


df_stalk_alls, df_marc_alls = built_dataset(df_list)


'''
print(len(df_stalk_alls))
[print(df_stalk_alls[i].shape) for i in range(len(df_stalk_alls))]
[print(type(df_stalk_alls[i])) for i in range(len(df_stalk_alls))]
[print(df_stalk_alls[i]) for i in range(len(df_stalk_alls))]

print(len(df_marc_alls))
[print(df_marc_alls[i].shape) for i in range(len(df_marc_alls))]
[print(type(df_marc_alls[i])) for i in range(len(df_marc_alls))]
[print(df_marc_alls[i]) for i in range(len(df_marc_alls))]
'''

# concatenate all DFs (by rwos)-.
df_stalk = pd.concat(df_stalk_alls, axis=0, ignore_index=True)
df_marc = pd.concat(df_marc_alls, axis=0, ignore_index=True)

# shuffle and re-write indexes-.
# check sklearn.utils.shuffle() and np.random.shuffle().
df_stalk = df_stalk.sample(frac=1).reset_index(drop=True)
df_marc = df_marc.sample(frac=1).reset_index(drop=True)

# print(df_stalk, df_marc, sep='\n')

# save DataSets in csv format-.
df_stalk.to_csv(ds_path+'stalk.csv')
df_marc.to_csv(ds_path+'marc.csv')

# END -- BLOCK_5 =============================================================79
