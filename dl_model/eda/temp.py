#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
#
# script para armar los archivos csv para procesar los datos
# Prof. Dra.- Ing. Rosa Rodriguez - IIQ - CONICET - FI - UNSJ-.
#
# start_date: lun 04 jul 2022 09:50:29 -03-.
# last_modify: mar jun  4 14:54:35 -03 2024-.


# modules importing-.
import pandas as pd
import subprocess as sp
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import interpolate  # interp1d, Rbf, RBFInterpolator
from scipy.signal import savgol_filter

from statsmodels.nonparametric.smoothers_lowess import lowess

from whittaker_eilers import WhittakerSmoother


# wraning configurations-.
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# clean terminal (XTERM in my GNU-Linux)-.
sp.run(['clear'])

# path (folder + file) with dataset (excel file provided by Dras. AF & RR)-.
# root_path = '/home/fcarazo/diploDatos2022/trabajoPirolisis_IIQ/'
# fileName = 'DatosParaRNA_Orig.xlsx'
# root_path = '/home/fdcarazo/my_github/kinetics/ds/'
root_path = '/home/fcarazo/my_github/kinetics/ds/'
fileName = 'DATOScRUDOSpARAeNTRENARlArED_mod.xlsx'


# ==============================================================================
class calc_Deriv():
    '''
    class to calculate derivative of cloud of points provided
    in PandasDataFrame-.
    '''
    def __init__(self, df, df_list, x_var, y_var, dt, type_biomass):
        ''' object builder '''
        # df.dropna(inplace=True) # not nocessary, I did it before-.
        # sort values in function of T (when I load DataFrame I shuffled its)-.
        self.df = df.sort_values(by=['Temp'], axis=0, ascending=False)
        self.df_list = df_list  # list with a dataframes, one for each heating rate-.

        self.df1, self.df2, self.df3 = list(map(lambda x: x.astype(float), self.df_list))
        
        self.x_var, self.y_var = x_var, y_var  # strings-.
        self.dt = dt

        self.t_b = type_biomass
        
    def original_data_sns_regplot(self, dir_save) -> None:
        '''
        plot original data using sns and regplot-.
        '''

        x_list, x_der_list, y_list, y_der_list = list(), list(), list(), list()
        x_smoothed_list, lowess_tight_list = [], []
        for idx in range(len(self.df_list)):
            # print(len(self.df_list), self.df_list[idx].columns, self.df_list[idx].vel_cal.unique())

            # df = self.df_list[0].drop_duplicates(subset=['Temp']).sample(n=1000)
            # df = self.df_list[idx].iloc[:, 0:4].sample(n=len(self.df_list[idx]))  # columns 0, 1, and 2 @ STALK-.
            df = self.df_list[idx].sample(n=len(self.df_list[idx]))
            df_smoothed = self.df_list[idx].sort_values(by=['Temp'],
                                                        axis=0,
                                                        ascending=True)

            '''
            print(df)
            print('shape: {0}'.format(df.shape))
            df = df.drop_duplicates(subset=['TGA'])
            df = df.drop_duplicates(subset=['Temp'])
            print('shape: {0}'.format(df.shape))
            '''
            
            df.dropna(inplace=True)  # it isn't necessary here-.
            df_smoothed.dropna(inplace=True)  # it isn't necessary here-.
            
            # column 0 ==> T (K) - column 1 ==> TGA (mg) - column 2 ==> DrTGA (mg/s)
            x, tga, dr_tga = [df.iloc[:, idx].to_numpy().astype(float) for idx in range(df.shape[1])]
            unique, indices = np.unique(x, return_index=True)  # if not I have problems to plot-.

            '''
            smoothed values using:
            statsmodels.nonparametric.smoothers_lowess import lowess
            exracted from:
            https://pieriantraining.com/python-smoothing-data-a-comprehensive-guide/
            https://towardsdatascience.com/data-smoothing-for-data-science-
            visualization-the-goldilocks-trio-part-1-867765050615
            '''

            drtga = np.diff(df_smoothed.iloc[:, 1].to_numpy().astype(float)) / self.dt
            lowess_tight = lowess(drtga,
                                  df_smoothed.iloc[:, 0].to_numpy().astype(float)[:-1],
                                  frac=.015)  # drtga[indices], x[indices]
            
            '''
            x= x[indices]
            print(np.shape(x), np.shape(np.unique(x)))
            print(x[indices], tga[indices])
            input(99)
            '''
            
            x_list.append(x); y_list.append(tga)
            x_der_list.append(x); y_der_list.append(dr_tga)
            x_smoothed_list.append(df_smoothed.iloc[:, 0].to_numpy().astype(float)[:-1])
            lowess_tight_list.append(lowess_tight)
            
        # plot (TGA -mg- vs. T -K-) and (d(TGA)/dt -mg/s- vs. T (s))-.
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': "sans-serif"})
        params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
        plt.rcParams.update(params)

        fig = plt.figure(figsize=(12, 9))
        fig1 = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        ax_1 = fig1.add_subplot(111)
        ax1 = ax.twinx()

        # custom_cycler = (cycler(ls=['-', '--', ':']))
        # plt.rc('axes', prop_cycle=custom_cycler)
        
        l_c = ['b', 'r', 'k']
        l_c_der = ['g', 'm', 'c']
        l_s = ['--', '-.', 'dashdot']
        p_s = ['8', 'p', 'o']
        l_l = ['5', '10', '15']

        for idx in range(len(x_list)):
            ax.scatter(x_list[idx], y_list[idx], label='{0}'.format(l_l[idx]),
                       c=l_c[idx], marker=p_s[idx], s=0.7)
            ax1.scatter(x_der_list[idx], -1.0*y_der_list[idx], label='{0}'.format(l_l[idx]),
                        c=l_c[idx], marker=p_s[idx], s=0.5)
            '''
            sns.regplot(x=x_der_list[idx],
                        y=-1.0*y_der_list[idx],
                        data=self.df_list[idx],
                        scatter_kws={'s': 0.1},
                        line_kws={'lw': 3.0, 'linestyle': l_s[idx], 'color': l_c[idx]},
                        order=14,
                        ci=10,
                        ax=ax1
                        )
            '''
            ax_1.scatter(x_list[idx], -1.0*y_der_list[idx],
                         label='measured-{0}'.format(l_l[idx]),
                         c=l_c[idx], marker=p_s[idx], s=0.5, alpha=0.4)
            # x_smoothed_list[idx]
            ax_1.plot(x_smoothed_list[idx], -1.0*lowess_tight_list[idx][:, 1],
                      label=r'smmothed-{0}'.format(l_l[idx]), c=l_c[idx],
                      ls=l_s[idx], lw=3.0, alpha=0.7)
            
        title = self.t_b
        
        plt.tight_layout()
        ticks, labels = plt.xticks()
        plt.grid()
        
        # ax.legend()
        ax.set_xlabel(r'$T\ (K)$')
        ax.set_ylabel(r'$TGA\ (mg)$')
        ax1.set_ylabel(r'$\frac{d(TGA)}{dt}\ (mg/s)$')

        ax1.set_xticks(np.arange(np.min(x_list[0]), np.max(x_list[0]), step=50))  # set label locations-.

        # @ smoothed plot-.
        ax_1.set_xlabel(r'$T\ (K)$')
        ax_1.set_ylabel(r'$TGA\ (mg)$')
        ax_1.set_xticks(np.arange(np.min(x_list[0]), np.max(x_list[0]), step=50))  # set label locations-.
        ax_1.set_title(title)
        ax_1.legend(title=r'$\frac{dT}{dt}$ (heating rate) (K/s)', loc='best')
        ax_1.grid(visible=True)
        ax_1.axvline(x=373, ymin=0, ymax=1.0, color='c', ls='--', lw=4.0)
        ax_1.axvline(x=423, ymin=0, ymax=1.0, color='c', ls='--', lw=4.0)
        ax_1.axvline(x=823, ymin=0, ymax=1.0, color='c', ls='--', lw=4.0)
        ax_1.text(0.15, 0.5, 'zone-1', transform=ax_1.transAxes,
                  fontsize=30, color='darkblue', alpha=0.7,
                  ha='center', va='center', rotation=90)
        ax_1.text(0.5, 0.5, 'zone-2', transform=ax_1.transAxes,
                  fontsize=30, color='darkblue', alpha=0.7,
                  ha='center', va='center', rotation=0)
        ax_1.text(0.8, 0.5, 'zone-3', transform=ax_1.transAxes,
                  fontsize=30, color='darkblue', alpha=0.7,
                  ha='center', va='center', rotation=0)

        '''
        ax.text(0.5, 0.5, 'created with matplotlib', transform=ax.transAxes,
        fontsize=40, color='gray', alpha=0.5,
        ha='center', va='center', rotation=30)
        '''

        ax.set_title(title)
        ax.legend(title=r'$\frac{dT}{dt}$ (heating rate) (K/s)', loc='best')
        # ax1.legend(title=r'$\frac{d(TGA)}{dt}$', loc='upper right')
        # ax1.legend(loc='best')

        ax.grid(which='both', alpha=1, visible=True)
        ax1.grid(which='both', alpha=1, visible=True)

        plt.show()
        
        fig.savefig(os.path.join(dir_save+'/figs/',
                                 self.t_b+'_testCurves.png'),format='png',
                    dpi=100)
        fig1.savefig(os.path.join(dir_save+'/figs/',
                                  self.t_b+'_smoothedCurves.png'),format='png',
                     dpi=100)

        # plot three curves joints-.
        # subplots-.
        fig2, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(16, 8))
        ax = [ax0, ax1, ax2]
        ticks, labels = plt.xticks()
        for idx in range(len(x_list)):
            ax[idx].scatter(x_list[idx], y_list[idx], label='{0}'.format(l_l[idx]),c=l_c[idx], marker=p_s[idx], s=0.7)
            ax_1_1 = ax[idx].twinx()
            ax_1_1.plot(x_smoothed_list[idx], -1.0*lowess_tight_list[idx][:, 1],
                        label=r'smmothed-{0}'.format(l_l[idx]), c=l_c[idx],
                        ls=l_s[idx], lw=3.0, alpha=0.7)

            # ax.legend()
            ax[idx].set_xlabel(r'$T\ (K)$')
            ax[idx].set_ylabel(r'$TGA\ (mg)$')
            ax_1_1.set_ylabel(r'$\frac{d(TGA)}{dt}\ (mg/s)$')
            
            ax[idx].legend(title=r'$TGA\ (mg)$', loc='best')
            ax[idx].grid(which='both', alpha=1, visible=True)
            ax_1_1.grid(which='both', alpha=1, visible=True)
            ax_1_1.legend(title=r'$\frac{d(TGA)}{dt}\ (mg/s)$', loc='upper center')

            ax[idx].set_xticks(np.arange(np.min(x_list[0]), np.max(x_list[0]), step=50))  # set label locations-.
            ax_1_1.set_xticks(np.arange(np.min(x_list[0]), np.max(x_list[0]), step=50))  # set label locations-.
            
        # ax1.legend(loc='best')

        title = self.t_b
        ax[0].set_title(title)
        
        # plt.grid()
        plt.show()

        fig2.savefig(os.path.join(dir_save+'/figs/',
                                  self.t_b+'_allHR_andDrTGSmoothed.png'),format='png',
                     dpi=100)

        return None
    
    # def calc_and_plot_Drtg(self, dir_save):
    def calc_and_plot_Drtg_dif_approx(self, x, y, npoints):
        '''
        plot derivative of TGA, i.e. DrTG (d(TGA)/dT) using differents approaches-.
        source: https://stackoverflow.com/questions/16841729/how-do-i-compute-the-derivative-of-an-array-in-python
        https://stackoverflow.com/questions/72511694/derivative-of-a-dataset-using-python-or-pandas
        p2c: print to check-.
        '''
        
        # approximations-.
        fun = interpolate.interp1d(x=x, y=y, kind='next')
        tck = interpolate.splrep(x=x, y=y, s=0, k=3)  #
        xp, yp =x, drtga
        # radial basis function interpolator instance
        # {'multiquadric', 'quintic', 'linear', 'gaussian',
        #  'thin_plate_spline', 'inverse_quadratic', 'cubic', 'inverse_multiquadric'
        rbi = interpolate.RBFInterpolator(xp[:, np.newaxis],
                                          yp[:, np.newaxis],
                                          kernel='thin_plate_spline')

        # define x values to plot-.
        x_toplot = np.linspace(np.min(x), np.max(x), npoints)

        # 1- calculate y values to plot with cubic interpelation-.
        y_cubic = fun(x_toplot)  # x2[indices]

        # 2- calculate y values to plot with splev interpelation-.
        # https://stackoverflow.com/questions/55808363/how-can-i-give-specific-x-values-to-scipy-interpolate-splev
        y_splev = interpolate.splev(x_toplot, tck, der=0)

        # 3- calculate y values to plot with RadialBasisFunction-.
        y_rbi = rbi(x_toplot[:, np.newaxis])  # interpolated values

        # 4- calculate y values to plot with whittaker_smmother-.
        # https://towardsdatascience.com/the-perfect-way-to-smooth-your-noisy-data-4f3fe6b44440
        y_for_smooth = y_cubic  # could be changed !!!, is the function to smooth !!!-.
        whittaker_smoother = WhittakerSmoother(
            lmbda=20, order=3, data_length=len(y_for_smooth)
        )
        smoothed_temp_anom = whittaker_smoother.smooth(y_for_smooth)

        # 5- calculate y values to plot with svgol_filter-.
        # https://pieriantraining.com/python-smoothing-data-a-comprehensive-guide/
        # Calculate smoothed values with Savitzky-Golay method
        y_savgolFilter = savgol_filter(y_for_smooth, window_length=5, polyorder=3)

        '''
        print(np.shape(x[indices]), np.shape(fun))
        print(fun.__dict__)
        input(11)
        '''

        ax.scatter(x_toplot, y_cubic, label='cubic_interpolation', c='indianred', marker='<', alpha=0.5)
        ## ax.plot(x_toplot, y_cubic, label='interpolated_3_line', c='indianred', ls='--', alpha=0.5)

        ax.scatter(x_toplot, y_splev, label='splev_interp', c='coral', marker='o', alpha=0.5)
        ## ax.plot(x_toplot, y_splev, label='splev_interp_line', c='coral', ls='-.', alpha=0.5)
        
        ax.scatter(x_toplot, y_rbi, label='rbi_interp', c='lime', marker='<', alpha=0.5)
        ## ax.plot(x_toplot, y_rbi, label='rbi_interp_line', c='lime', ls='-:', alpha=0.5)

        ax.scatter(x_toplot, smoothed_temp_anom, label='witt_smooth', c='deepskyblue', marker='<', alpha=0.5)
        ## ax.plot(x_toplot, smoothed_temp_anom, label='witt_smooth_line', c='deepskyblue', ls=':', alpha=0.5)

        ax.scatter(x_toplot, y_savgolFilter, label='savgol_smooth', c='slategrey', marker='8', alpha=0.5)
        ## ax.plot(x_toplot, y_savgolFilter, label='savgol_smooth', c='slategrey', ls='':', alpha=0.5)

        ax.legend()
        plt.legend(loc='best')
        ax.set_xlabel('T (K)')
        ax.set_ylabel(r'DrTGA ($\frac{d(TGA)}{dt}$ (mg/s)')

        title = r'DrTGA ($\frac{d(TGA)}{dt}$ (mg/s)'
        ax.set_title(title)

        plt.grid()
        plt.tight_layout()
    
        ticks, labels = plt.xticks()
        ## plt.xticks(ticks[::1], labels[::1])
        plt.xlim([np.min(x), no.max(x)])
        plt.ylim([0.0,0.01250])
    
        # fig.legend(*ax.get_legend_handles_labels(),loc='lower center', ncol=4)
        # plt.legend(lines,labels,loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
        # 2 use \ bar in latexMode-.
        # https://stackoverflow.com/questions/65702154/
        # problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
    
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        # label= r'$abs(\Delta C_{{ij}})= |C_{{ij}}^{{VPSC}}-C_{{ij}}^{{PREDICTED}}|= f(\bar{\varepsilon})$'## +\
            #    ' , for {0}'.format(self.i_run)
        # r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
            # r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
        
        # fig.text(0.6,0.1,label,color='r',fontsize=16,
        #          horizontalalignment='right',verticalalignment='top',
        #          backgroundcolor='1.0')
        
        plt.show()
        
        # fig.savefig(os.path.join(dir_save,'DrTGvsT.png'),format='png',dpi=100)
        return None
# ==============================================================================


# 1- before start I check that folder and file exists-.
# pending-.
biomass_type = ['MCP5', 'MCP10', 'MCP15',
                'MSP5', 'MSP10', 'MSP15',
                'OU10', 'OU15', 'OU20',
                'MV10', 'MV15', 'MV20',
                'EU10', 'EU15', 'EU20',
                'ESP10', 'ESP15', 'ESP20',
                'PP5', 'PP10', 'PP15',
                'OP5', 'OP10', 'OP15',
                'SD5', 'SD10', 'SD15']
df_names = list()
df_list = list()
for i, i_biomass_type in enumerate(biomass_type):
    # print(i_vel_enfr, i)
    df_name = ''.join(['df', i_biomass_type.replace(' ', '')])
    df_names.append(df_name)
    df = pd.read_excel(root_path + fileName, sheet_name=i_biomass_type)
    df['biomass_type'] = str(i_biomass_type)
    df_list.append(df)

    print('DataFrame column names {0}{1}'.format(df_list[i].columns, '\n'))
    print('DataFrame name {0}{1}'.format(df_list[i].index.name, '\n'))
    print('DataFrame content {0}{1}'.format('\n', df_list[i]))
    # dfName.index.name = ''.join(['df', i_biomass_type.replace(' ', '')])

# post-process pandasDataFrame (not apply in this case)-.
# remove row 0 of each dataframe
## df_list = list(map(lambda x: x.drop(labels=0, axis=0).reset_index(), df_list))
# remove column 'index' -index from excel- in each PandasDataFrame-.
## df_list = list(map(lambda x: x.drop(['index'], axis=1), df_list))
'''
# check dataframes after post-processing-.
for i, i_biomass_type in enumerate(biomass_type):
    print('DataFrame name {0}{1}'.format(df_list[i].index.name, '\n'))
    print('DataFrame content {0}{1}'.format('\n', df_list[i]))
    # dfName.index.name = ''.join(['df', i_biomass_type.replace(' ', '')])
'''

# concatenate all DFs-.
df_alls = pd.concat(df_list, axis=0, ignore_index=True)
print(df_alls, df_alls.columns, df_alls.shape, sep='\n')
# apply one_hot_enconding-.
df_alls_encoded = pd.get_dummies(df_alls, columns=['biomass_type'])
print(df_alls)
print(df_alls_encoded, df_alls_encoded.columns, df_alls_encoded.shape, sep='\n')


i_name = 'MSP'
indices = df_alls.index[df_alls['biomass_type'].str.contains(i_name)].tolist()
df_alls_filtered = df_alls.loc[indices]

print(df_alls_filtered)
print(df_alls_filtered.dtypes)
df_alls_filtered['heat_rate'] = df_alls_filtered['heat_rate'].astype(str)
print(df_alls_filtered.dtypes)

import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(6,4))
sns.scatterplot(data=df_alls_filtered.sample(frac=1.0),
                hue= 'heat_rate',
                x= 'T',
                y= 'mass_porc',
                legend='full',
                marker= 'o',
                palette= 'viridis',
                s = 10,
                ax= ax)

'''
plt.scatter(data=df_alls_filtered,
            # hue= 'heat_rate',
            x= 'T',
            y= 'mass_porc',
            # legend='full',
            marker= 'o')
'''
plt.grid(True)
plt.show()
plt.xlabel('T')
plt.ylabel('mass_porc')
