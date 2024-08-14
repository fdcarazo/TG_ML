#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# class to calculate and plot derivative of TG vs. T, i.e. d(TG)/d(T) (mg/K)
#
# start_date (Arg): mar jun  4 14:54:35 -03 2024 -.
# last_modify (Arg): mar 13 ago 2024 20:34:01 -03 -.
##

# import modules/packages/libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import interpolate
from scipy.interpolate import interp1d

from scipy.interpolate import RBFInterpolator
from scipy.interpolate import RBFInterpolator
from scipy.signal import savgol_filter

from statsmodels.nonparametric.smoothers_lowess import lowess

from whittaker_eilers import WhittakerSmoother
# ==============================================================================
class calcDeriv():
    '''
    class to calculate derivative of cloud of points provided
    in PandasDataFrame-.
    '''
    
    def __init__(self, df, x_var, y_var, dt):
        ''' object builder '''
        
        # df.dropna(inplace=True)  # not nocessary, I did it before-.
        # sort values in function of T (when I load DataFrame I shuffled this)-.
        self.df = df.sort_values(by=['Temp'], axis=0, ascending=False)
        self.x_var, self.y_var = x_var, y_var  # strings-.
        self.dt = dt
        
    # def calc_and_plot_Drtg(self, dir_save):
    def calc_and_plot_Drtg(self):
        '''
        plot derivative of TGA, i.e. DrTG (d(TGA)/dT)-.
        source: https://stackoverflow.com/questions/16841729/how-do-i-compute-the-derivative-of-an-array-in-python
        https://stackoverflow.com/questions/72511694/derivative-of-a-dataset-using-python-or-pandas
        p2c: print to check-.
        '''
        
        # drtg_pred= np.gradient(self.y_pred[:,0], self.x[:,0])
        # drtg_true= np.gradient(self.y_true[:,0], self.x[:,0])
        
        # col. 0 ==> T (K) -for one HeatRate-.
        x = self.df.loc[:, self.x_var].to_numpy().astype(float)
        # col. 1 ==> TGA (mg) -for one HeatRate-.
        y = self.df.loc[:, self.y_var].to_numpy().astype(float)
        
        drtga = np.diff(y)/self.dt  # calculate firsth order difference $\Delta TGA=TGA^{t+\Delta t}-TGA^{t}$-.
        # print(np.shape(x[:-1]), np.shape(drtga)); input(11)
        # input(11111)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # ax.scatter(self.x[:,0], drtg_pred, label='predicted', c= 'blueviolet', marker='>', alpha= 0.5)
        # ax.scatter(x[:-1], drtga, label='exp', c='indianred', marker='<', alpha=0.5)

        '''
        sns.regplot(x=x[:-1],  # if I don't do that there will be problems betweeen size(x) and size(y)
                    y=drtga,
                    data=self.df,
                    scatter_kws={'s': 0.1},
                    line_kws={'lw': 3.0, 'linestyle': ':', 'color': 'green'},
                    order=14,
                    ci=10,
                    ax=ax
                    )
        plt.show(); input(9999)
        '''
        
        # define interpolators.
        x = x[:-1]
        unique, indices = np.unique(x, return_index=True)
        '''
        print(np.shape(x), np.shape(np.unique(x)))
        print(x[indices], drtga[indices])
        print(np.shape(x[indices]), np.shape(drtga[indices]))
        input(99)
        '''
        fun = interpolate.interp1d(x=x[indices], y=drtga[indices], kind='next')
        tck = interpolate.splrep(x=x[indices], y=drtga[indices], s=0, k=3)  #

        '''
        print(np.shape(x[indices]), np.shape(fun))
        print(fun.__dict__)
        input(11)
        '''
        
        x3 = np.linspace(np.min(x[indices]), np.max(x[indices]), 500)
        y3 = fun(x3)  # x2[indices]
        y4 = interpolate.splev(x3, tck, der=0)
        # https://stackoverflow.com/questions/55808363/how-can-i-give-specific-x-values-to-scipy-interpolate-splev
        yi = interpolate.splev(x3, tck)

        xp, yp = x[indices], drtga[indices]
        # radial basis function interpolator instance
        # {'multiquadric', 'quintic', 'linear', 'gaussian',
        #  'thin_plate_spline', 'inverse_quadratic', 'cubic', 'inverse_multiquadric'
        rbi = RBFInterpolator(xp[:, np.newaxis], yp[:, np.newaxis], kernel='thin_plate_spline')
        fi = rbi(x3[:, np.newaxis])       # interpolated values
        
        '''
        print(x3, y3, sep='\n')
        input(66)
        '''
        
        # https://towardsdatascience.com/the-perfect-way-to-smooth-your-noisy-data-4f3fe6b44440
        whittaker_smoother = WhittakerSmoother(
            lmbda=20, order=3, data_length=len(y3)
        )
        smoothed_temp_anom = whittaker_smoother.smooth(y3)

        # https://pieriantraining.com/python-smoothing-data-a-comprehensive-guide/
        # calculate smoothed values with Savitzky-Golay method-.
        smoothed = savgol_filter(y3, window_length=5, polyorder=3)

        # == INI -- Local fit using statmodels (nonparametric local fit)  -- -.
        # GREAT AND COMPLETE ARTICLE !!!!
        # https://pieriantraining.com/python-smoothing-data-a-comprehensive-guide/
        # https://towardsdatascience.com/data-smoothing-for-data-science-visualization-the-goldilocks-trio-part-1-867765050615
        lowess_tight = lowess(drtga[indices], x[indices], frac=.015)
        # lowess_mean = lowess(drtga[indices], x[indices], frac=.1)
        lowess_loose = lowess(drtga[indices], x[indices], frac=.2)
        lowess_list = [(lowess_tight[:, 0], lowess_tight[:, 1]),
                       # (lowess_mean[:, 0], lowess_mean[:, 1]),
                       (lowess_loose[:, 0], lowess_loose[:, 1])]
        # lowess_list = [(lowess_tight[:,0], lowess_tight[:,1])]
        # print(lowess_tight, lowess_tight[:,0], lowess_tight[:,1]); input(999)
        #
        
        def graph(x_array, y_array, custom_title, scatter=True,
                  solid_line=False, line_viz=None):
            ''' 
            This function can accept basic x and y arrays for a basic 
            scatter plot or line plot (or both), but it can also accept,
            in its line_viz parameter, a list of x, y array pairs.
            Its configuration for coloring, line weight, and dot size are
            tailored specifically for the smoothing examples presented in
            this blog
            '''
            
            # set up the plotting-.
            figure_proportions = (10, 5)
            plt.figure(figsize=(10, 8))
            x_min, x_max = np.min(x_array), np.max(x_array)
            y_min, y_max = np.min(y_array), 0.0125
            # y_max = 100

            plt.legend()
            plt.xlabel('T (K)')
            plt.ylabel('DTG (d(TG)/dt) (mg/s)')

            # plt.ylim(0, 0.0125)
            title = 'DTG (d(TG)/dt) (mg/s) -- ' + custom_title
            plt.title(title)
        
            plt.legend(loc='best')
            
            plt.grid()
            # ax.legend()
            
            # ax.set_xlabel('T (K)')
            # ax.set_ylabel('dT/dt (K/min)')
            # ax.set_zlabel('TGA (mg)')
            # ax.set_title(title)
            
            # plt.xlabel('time (s)'); plt.ylabel('TGA (mg)')
            
            plt.tight_layout()
            
            ticks, labels = plt.xticks()
            # plt.xticks(ticks[::1], labels[::1])
            # plt.xlim([0.0,2.0])
            
            # fig.legend(*ax.get_legend_handles_labels(),loc='lower center', ncol=4)
            # plt.legend(lines,labels,loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
            # 2 use \ bar in latexMode-.
            # https://stackoverflow.com/questions/65702154/
            # problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
            
            plt.rc('text', usetex=True)
            plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
            # label= r'$abs(\Delta C_{{ij}})= |C_{{ij}}^{{VPSC}}-C_{{ij}}^{{PREDICTED}}|= f(\bar{\varepsilon})$'## +\
            #        ' , for {0}'.format(self.i_run)
            #        r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
            #        r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
                
            # fig.text(0.6,0.1,label,color='r',fontsize=16,
            #          horizontalalignment='right',verticalalignment='top',
            #          backgroundcolor='1.0')
            
            # y_axis_label = 'altitude (feet)'
            # x_axis_label = 'time (seconds from arrival at target)'
            # title = 'Trajectory of flight'
            # custom_title = title + ' - ' + custom_title
            # plt.title(custom_title)
            dot_opacity = line_opacity = 1
            size = 30 if len(x_array) > 15 else 100
            line_weight = 1
            
            # handle the cases where lines or curves are visualized,
            # beyond the obvious-.
            if line_viz != None:
                if len(line_viz) == 1:
                    colors = ['green']
                if len(line_viz) == 2:
                    colors = ['lightseagreen', 'olive']
                    dot_opacity = .3
                    line_weight = 3
                if len(line_viz) == 3:
                    colors = ['lightseagreen', 'olive', 'blue']
                    dot_opacity = .3
                    line_weight = 3
                if len(line_viz) == 4:
                    colors = ['turquoise', 'magenta', 'tomato', 'gold']
                    dot_opacity = .1
                    line_opacity = .1
                for i, (x, y) in enumerate(line_viz):
                    plt.plot(x, y, color=colors[i], lw=3)
                    
            # handle the standard cases-.
            if scatter:
                plt.scatter(x_array, y_array, color='green', marker='o',
                            alpha=dot_opacity, s=size)
            if solid_line:
                plt.plot(x_array, y_array, color='green', alpha=line_opacity)
            # present the plotting-.
            # plt.xlabel(x_axis_label)
            # plt.ylabel(y_axis_label)
            # plt.xlim(x_min, x_max )
            # plt.ylim(0.0, 0.0125)
            # plt.savefig('viz/'+custom_title+'.svg')
            plt.show()
            input(12345676)

        #
        '''
        graph(x[indices], drtga[indices],
              ' - noisy data'+ ' - with lowess smoothing: tight (1.5% bins) and looser (20% bins)',
              scatter=True, solid_line=False, line_viz=lowess_list)
        input(3333)
        '''
        # == END -- Local fit using statmodels (nonparametric local fit) -- -.

        
        # =============================== INI -- fit using whittaker_smoother -- -.
        # ax.scatter(x3, y3, label='interpolated_3_points', c='indianred', marker='<', alpha=0.5)
        # ax.plot(x3, y3, label='interpolated_3_line', c='blue', ls='--', alpha=0.5)
        ax.plot(x3, smoothed_temp_anom, label='interpolated_3_smoothed', c='blue', ls='--', alpha=1.0)
        # ax.plot(x3, smoothed, label='interpolated_3_savgol', c='red', ls='-.', alpha=1.0)
        # ax.scatter(x3, smoothed_temp_anom, label='interpolated_3_smoothed', c='indianred', marker='<', alpha=0.5)
        # ax.scatter(x3, y4, label='interpolated_1', c='blue', marker='o', alpha=0.5)
        # ax.plot(x3, y4, label='interpolated_4', c='blue', ls='--', alpha=0.5)
        # ax.plot(x3, yi, label='interpolated_i', c='blue', ls='-', alpha=0.5)
        # ax.plot(x3, fi, label='interpolated_rbf', c='blue', ls='-', alpha=0.5)
        # ax1.plot(x_der[idx], -1.0*y_der[idx], label='{0}'.format(l_l[idx]), c=l_c_der[idx],
        #         ls=l_s[idx])
        
        ax.legend()
        ax.set_xlabel('T (K)')
        ax.set_ylabel('DrTGA (d(TGA)/dt) (mg/s)')

        plt.ylim(0, 0.0125)

        title = 'DrTGA (d(TGA)/dt) (mg/s)'
        ax.set_title(title)
        
        plt.legend(loc='best')
        
        plt.grid()
        # ax.legend()
        
        # ax.set_xlabel('T (K)')
        # ax.set_ylabel('dT/dt (K/min)')
        # ax.set_zlabel('TAG (mg)')
        # ax.set_title(title)
    
        # plt.xlabel('time (s)'); plt.ylabel('TGA (mg)')
    
        plt.tight_layout()
    
        ticks, labels = plt.xticks()
        # plt.xticks(ticks[::1], labels[::1])
        # plt.xlim([0.0,2.0])
    
        # fig.legend(*ax.get_legend_handles_labels(),loc='lower center', ncol=4)
        # plt.legend(lines,labels,loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
        # 2 use \ bar in latexMode-.
        # https://stackoverflow.com/questions/65702154/
        # problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
    
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        # label= r'$abs(\Delta C_{{ij}})= |C_{{ij}}^{{VPSC}}-C_{{ij}}^{{PREDICTED}}|= f(\bar{\varepsilon})$'## +\
        #        ' , for {0}'.format(self.i_run)
        #        r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
        #        r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
        
        # fig.text(0.6,0.1,label,color='r',fontsize=16,
        #          horizontalalignment='right',verticalalignment='top',
        #          backgroundcolor='1.0')
        
        plt.show()
        # fig.savefig(os.path.join(dir_save,'DrTGvsT.png'),format='png',dpi=100)
        # input(11111)
        # =============================== END -- fit using whittaker_smoother -- -.

        
        # == INI -- Local fit using statmodels (nonparametric local fit) -- -.
        lowess_tight = lowess(drtga[indices], x[indices], frac=.015)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.scatter(x[indices], drtga[indices], label='exp_values',
                   c='indianred', marker='<', alpha=0.5)
        ax.plot(x[indices], lowess_tight[:, 1], label='smoothers_lowes',
                c='blue', ls='-', lw=3.0, alpha=1.0)
        # ax.legend()
        ax.set_xlabel('T (K)')
        ax.set_ylabel('DrTGA (d(TGA)/dt) (mg/s)')

        plt.ylim(0, 0.0125)
        title = 'DrTGA (d(TGA)/dt) (mg/s)'
        ax.set_title(title)
        # plt.legend(loc='best')
        
        plt.grid()
        # ax.legend()
        
        # ax.set_xlabel('T (K)')
        # ax.set_ylabel('dT/dt (K/min)')
        # ax.set_zlabel('TAG (mg)')
        # ax.set_title(title)
    
        # plt.xlabel('time (s)'); plt.ylabel('TGA (mg)')
    
        plt.tight_layout()
    
        ticks, labels = plt.xticks()
        # plt.xticks(ticks[::1], labels[::1])
        # plt.xlim([0.0,2.0])
    
        # fig.legend(*ax.get_legend_handles_labels(),loc='lower center', ncol=4)
        # plt.legend(lines,labels,loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
        # 2 use \ bar in latexMode-.
        # https://stackoverflow.com/questions/65702154/
        # problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
    
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        # label= r'$abs(\Delta C_{{ij}})= |C_{{ij}}^{{VPSC}}-C_{{ij}}^{{PREDICTED}}|= f(\bar{\varepsilon})$'## +\
        #        ' , for {0}'.format(self.i_run)
        #        r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
        #        r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
        
        # fig.text(0.6,0.1,label,color='r',fontsize=16,
        #          horizontalalignment='right',verticalalignment='top',
        #          backgroundcolor='1.0')
        
        plt.show()
        # fig.savefig(os.path.join(dir_save,'DrTGvsT.png'),format='png',dpi=100)
        # input(2222)
        # == END -- Local fit using statmodels (nonparametric local fit) -- -.
        return None  # fig
# ==============================================================================
