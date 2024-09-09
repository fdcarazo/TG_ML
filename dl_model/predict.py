#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# class to define functions to plot predictions-.
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# start_date (Fr): mié 24 abr 2024 09:00:37 -03-.
# last_modify (Ar): mié 21 ago 2024 19:41:49 -03-.
##
# ======================================================================= INI79

# print(dir()); input(1)

# import packages/libraries/modules-.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import os

# @ 3D plots-.
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import griddata

# from sklearn.metrics import PredictionErrorDisplay


class Predict():
    def __init__(self, x, y_true, y_pred, feat_scal, targ_scal,
                 feat_vars, targ_vars, scale_targ):
        # MBC, add a label to control if features and targets
        # were standarized or scaled-.
        self.x = feat_scal.inverse_transform(x)  # used to plot y_pred and y_true vs. [T, dT/dt]-.
        # Apply inverse scaling in a cell before the figure cells-.
        self.y_true = targ_scal.inverse_transform(y_true) if scale_targ else y_true
        self.y_pred = targ_scal.inverse_transform(y_pred) if scale_targ else y_pred
        
        self.feat_vars = feat_vars
        self.targ_vars = targ_vars
        
    def plot_corr_true_pred_mod(self, dir_save):
        ''' 
        plot correlation between predicted and true values-.
        '''
        
        plt.rcParams.update({'font.size': 10})
        # gs= gridspec.GridSpec(2,3)  # 6 subfigures (it's a function of number of targets vars)-.
        gs= gridspec.GridSpec(1,1)  # 1 subfigures (it's a function of number of targets vars)-.

        # fig= plt.figure(figsize=(15, 10))  # @ 6 subfigures-.
        fig= plt.figure(figsize=(8, 4))  # @ 1 subfigure-.
        palette= sns.color_palette('mako_r',4)
        
        for idx,col in enumerate(self.targ_vars, start=0):
            ax= fig.add_subplot(gs[idx])

            y_true= self.y_true[:, idx]; y_pred= self.y_pred[:, idx]

            # plot true values (as cloud of points)-.
            plt.scatter(y_true, y_pred, color='r', marker='o', label=str(col), alpha=0.1)
            # correlation between true & predicted values-.
            pearR= np.corrcoef(y_true, y_pred)[1,0]
            A= np.vstack([y_true, np.ones(len(y_true))]).T
            m, c= np.linalg.lstsq(A, y_true)[0]
            # plot predicted values (as straight line)-.
            plt.plot(y_true, y_true*m+c, color='b', linestyle='--', label='Fit -- r = %6.4f'%(pearR))

            plt.legend(loc=2)
            plt.grid()
            plt.xlabel('true'); plt.ylabel('predicted')
            plt.tight_layout()
            ticks, labels= plt.xticks()
            
            # plt.xticks(ticks[::1], labels[::1])
            # plt.xlim([0.0,2.0])
        
        # fig.legend(*ax.get_legend_handles_labels(),loc='lower center', ncol=4)
        # plt.legend(lines,labels,loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
        # 2 use \ bar in latexMode-.
        # https://stackoverflow.com/questions/65702154/
        # problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
        
        # plt.rc('text', usetex=True)
        # plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        # label= r'$abs(\Delta C_{{ij}})= |C_{{ij}}^{{VPSC}}-C_{{ij}}^{{PREDICTED}}|= f(\bar{\varepsilon})$'## +\
           #     ' , for {0}'.format(self.i_run)
           #     r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
           #     r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
            
        # fig.text(0.6, 0.1, label,color='r', fontsize=16,
        #          horizontalalignment='right', verticalalignment='top',
        #          backgroundcolor='1.0')
        plt.show()
        fig.savefig(os.path.join(dir_save,'corrTruePred.png'), format='png', dpi=100)

    
    def plot_pred_error_display(self):
        fig, ax= plt.subplots(1, 1, figsize=(9, 7))
        display= PredictionErrorDisplay.from_predictions(
            y_true= self.y_true,
            y_pred= self.y_pred,
            kind= 'true_vs_predicted',
            ax= ax,
            scatter_kwargs= {'alpha': 0.2, 'color': 'tab:blue'},
            line_kwargs={'color': 'tab:red'},
        )
        # elapsed_time= 0.2
        name= 'MLP'
        # ax.set_title(f'{name}\nEvaluation in {elapsed_time:.2f} seconds')
        
        # for name, score in scores.items():
        #     ax.plot([], [], " ", label=f"{name}: {score}")
        # ax.legend(loc="upper left")

        plt.suptitle('True_vs_Predicted')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


    def plot_with_uncertainty(self, x, list_dfs, bounds_, bounds_vl_,
                              ds_name:str, dir_save:str, quart:int)->int:
        '''
        plot-.
        '''
        y_mean= np.mean(list_dfs, axis=0)
        y_std= np.std(list_dfs, axis=0)
        
        nr,nc= 2,3 # if str.lower(dim)=='2d' else (7,6) if dim=='3d' else (None,None)
        plt.rcParams.update({'font.size': 10}); gs= gridspec.GridSpec(nr, nc)
        fig= plt.figure(figsize=(15, 10))
        sns.color_palette('mako_r',4)
        
        # feature_in_var-.
        for idx, col in enumerate(self.targ_vars,start=0):
            ax= fig.add_subplot(gs[idx])  # , sharex=True, sharey=False)
            # true values-.
            var_to_plot= str(col)
            x= x  # list_dfs[0].phiF

            plt.scatter(x=x, y=y_mean[:,idx], s=10,facecolors='none',
                        edgecolor='k', marker='^', label='FFNN (Feed Forward Neural Network)'
                        # edgecolor='k',marker='^',label='BNN (Bayesian Neural Network)'
                        # alpha=0.1, c='blue',
                        )
            plt.fill_between(x,
                             y_mean[:,idx]+quart*y_std[:,idx],
                             y_mean[:,idx]-quart*y_std[:,idx], 
                             alpha=0.5,label='Epistemic uncertainty'
                             )
            
            plt.axvline(x=bounds_vl_['lower'], ymin=0.0,
                        ymax=1.0,linewidth=2, linestyle='--', color='r')
            plt.axvline(x=bounds_vl_['upper'], ymin=0.0,
                        ymax=1.0,linewidth=2, linestyle='--', color='r')
            # plt.legend(loc=3)
            plt.grid()
            plt.xlabel(r'$\phi_{frac}$')
            
            # col=str(col).replace('_out','')
            # plt.ylabel(r'$\Delta$'+
            #            '{0}'.format(col)
            #            )
            plt.ylabel(str(col))

            plt.tight_layout()
        
            ticks, labels=plt.xticks()
            # plt.xticks(ticks[::1], labels[::1])
            plt.xlim([0.0, 1.0])
            
            fig.legend(*ax.get_legend_handles_labels(),
                       loc='lower center', ncol=4)
        
        # problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        
        # label= r'Properties prediction in function of $\phi_{frac}$'
        # label= r'$C_{{ij}}^{{VPSC}}-C_{{ij}}^{{REC-PRED}}-C_{{ij}}^{{NON-REC-PRED}}= '\
        #        r'f(\bar{\varepsilon})$'## +\
        #        ' , for {0}'.format(self.i_run)
        #        r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
        #        r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
                
        # fig.text(0.7,0.12,label,color='r',fontsize=16,
        #          horizontalalignment='right',verticalalignment='top',
        #          backgroundcolor='1.0')
        
        plt.show()
        fig.savefig(os.path.join(dir_save, 'predWithUncertainty.png'),
                    format='png', dpi=100)

        return 0


    def plot_residuals(self, dir_save)-> int:
        ''' plot residuals as (predicted - true) values '''

        # gs= gridspec.GridSpec(2,3)  # 6 subfigures (it's a function of number of targets vars)-.

        # gs=gridspec.GridSpec(2,3) # 9 subfigures (function of number of targets vars)-.
        # fig=plt.figure(figsize=(15, 10))

        plt.rcParams.update({'font.size': 10})
        gs= gridspec.GridSpec(1,1)  # 1 subfigures (it's a function of number of targets vars)-.
        fig= plt.figure(figsize=(8, 4))
        
        palette= sns.color_palette('mako_r', 4)
        
        for idx,col in enumerate(self.targ_vars, start=0):
            ax= fig.add_subplot(gs[idx])
            y_true= self.y_true[:,idx]; y_pred= self.y_pred[:,idx]
            res= y_true- y_pred
            sns.distplot(res, kde=True, vertical=False,
                         rug=True, hist=False, kde_kws=dict(fill=True),
                         rug_kws= dict(lw=2, color='orange'))
            plt.grid()
            # plt.xlabel('true'); plt.ylabel('predicted')
            plt.xlabel(str(col))
            
            plt.tight_layout()
        
            ticks,labels= plt.xticks()
            # plt.xticks(ticks[::1], labels[::1])
            # plt.xlim([0.0, 2.0])
        
            # fig.legend(*ax.get_legend_handles_labels(),loc='lower center', ncol=4)
        # plt.legend(lines, labels, loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
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
        fig.savefig(os.path.join(dir_save, 'residualsFigs.png'),
                    format='png',
                    dpi=100)

        plt.show()
        return 0

        
    def y_vs_x(self,dir_save):
        ''' 
        plot predicted vs. true-.
        '''
        
        plt.rcParams.update({'font.size': 10})
        # gs= gridspec.GridSpec(2,3)  # 6 subfigures (it's a function of number of targets vars)-.
        # fig= plt.figure(figsize=(15, 10))
        gs= gridspec.GridSpec(1,1)  # 1 subfigures (it's a function of number of targets vars)-.

        
        fig= plt.figure(figsize=(8, 4))
        palette= sns.color_palette('mako_r',4)
        
        for idx,col in enumerate(self.targ_vars,start=0):
            ax= fig.add_subplot(gs[idx])
            y_true= self.y_true[:,idx]; y_pred= self.y_pred[:,idx]
            # print(np.shape(self.x), np.shape(y_pred), np.shape(y_true))
            plt.scatter(self.x[:,4], y_pred, color='red', marker='>',
                        s=3, label=str(col)+'pred', alpha=0.5)
            plt.scatter(self.x[:,4], y_true, color='greenyellow', marker='<',
                        s=3, label=str(col)+'true', alpha=0.5)
            plt.legend(loc='best')

            plt.grid()
            plt.xlabel('T (K)'); plt.ylabel('TG (mg)')
            
            plt.tight_layout()
        
            ticks,labels= plt.xticks()
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
        fig.savefig(os.path.join(dir_save,'predAndTrueVsT_2D.png'), format='png', dpi=100)

        
    def plot_3d_lls(self, x, y, z, lls_sol, dir_save, title=''):
        
        '''
        plot TG curves in 3D: [x=(T [K]), y=dT/dt [K/min]), z=(TG [mg])]
        '''
        
        fig = plt.figure(figsize=(8, 8))    
        ax = fig.add_subplot(111, projection='3d')
        # print(np.shape(x),np.shape(y), np.shape(z), np.shape(lls_sol), sep='\n')
        
        ax.scatter(x, y, z, label='true', c= 'blueviolet', marker='>', alpha= 0.5)
        ax.scatter(x, y, lls_sol, label='predicted', c= 'indianred', marker= '<', alpha= 0.5)
        ax.legend()
        ax.set_xlabel('T (K)')
        ax.set_ylabel('dT/dt (K/min)')
        ax.set_zlabel('TAG (mg)')
        ax.set_title(title)

        plt.legend(loc='best')
        
        plt.grid()
        # ax.legend()
        
        # ax.set_xlabel('T (K)')
        # ax.set_ylabel('dT/dt (K/min)')
        # ax.set_zlabel('TAG (mg)')
        # ax.set_title(title)
        
        # plt.xlabel('time (s)'); plt.ylabel('TG (mg)')
        
        plt.tight_layout()
        
        ticks,labels=plt.xticks()
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
            
        #fig.text(0.6,0.1,label,color='r',fontsize=16,
        #         horizontalalignment='right',verticalalignment='top',
        #         backgroundcolor='1.0')
        plt.show()
        fig.savefig(os.path.join(dir_save,'predAndTrueVsT_3D.png'),format='png',dpi=100)

        return fig

    
    def y_vs_x_3d(self, dir_save)-> None:

        '''
        plot TG curves in 3D: [x=(T [K]), y=dT/dt [K/min]), z=(TG [mg])]
        '''
        
        plt.rcParams.update({'font.size': 10})
        # gs= gridspec.GridSpec(2,3) # 6 subfigures (it's a function of number of targets vars)-.
        # fig= plt.figure(figsize=(15, 10))  # @ 6 subfigures-.
        # gs= gridspec.GridSpec(1,1) # 1 subfigure (it's a function of number of targets vars)-.
        # fig= plt.figure(figsize=(8, 4))  # @ 1 subfigure-.
        palette= sns.color_palette('mako_r',4)

        # print(np.shape(self.x), np.shape(y_pred), np.shape(y_true))

        self.plot_3d_lls(self.x[:,4],
                         self.x[:,3],
                         self.y_true[:,0],
                         self.y_pred[:,0],
                         dir_save,
                         '$TG=f(T,dT/dt)$')
        return None

    
    def calc_and_plot_Dtg(self, dir_save):
        '''
        plot derivative of TG wrt T, i.e. DTG (d(TG)/dT)-.
        source: https://stackoverflow.com/questions/16841729/how-do-i-compute-the-derivative-of-an-array-in-python
        https://stackoverflow.com/questions/72511694/derivative-of-a-dataset-using-python-or-pandas
        '''

        dt= 1.0  # MBM-.
        
        # drtg_pred= np.gradient(self.y_pred[:,0], self.x[:,0])
        # drtg_true= np.gradient(self.y_true[:,0], self.x[:,0])

        drtg_pred= np.diff(self.y_pred[:,0])
        drtg_true= np.diff(self.y_true[:,0])
        print(np.shape(self.x[:-1,0]), np.shape(drtg_true)); input(11)
        
        
        fig= plt.figure(figsize=(10,8))
        ax= fig.add_subplot(111)
        
        # ax.scatter(self.x[:,0], drtg_pred, label='predicted', c= 'blueviolet', marker='>', alpha= 0.5)
        ax.scatter(self.x[:-1,0], drtg_true, label='true',
                   c= 'indianred', marker= '<', alpha= 0.5)
        ax.legend()
        ax.set_xlabel('T (K)')
        ax.set_ylabel('DTG (d(TG)/dT) (mg/T)')

        title= 'DTG (d(TG)/dT) (mg/T)'
        ax.set_title(title)
        
        plt.legend(loc='best')
        plt.grid()        
        plt.tight_layout()
        
        ticks, labels= plt.xticks()
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
        fig.savefig(os.path.join(dir_save,'DTGvsT.png'),format='png',dpi=100)

        return fig

    
    def plot_surf_with_proj(self, dir_save, num_eles)-> None:
        '''
        plot 3d surface wih projections in coordinate planes-.
        source: https://www.tutorialspoint.com/matplotlib/matplotlib_3d_surface_plot.htm
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
        https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
        https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html
        '''

        # ax= plt.figure().add_subplot(projection='3d')
        # X, Y, Z= axes3d.get_test_data(0.05)

        '''
        self.plot_3d_lls(self.x[:,0],
                         self.x[:,1],
                         self.y_true,
                         self.y_pred,
                         dir_save,
                         '$TG=f(T,dT/dt)$')
        '''
        
        # print(type(self.x[:,0])); input(11)

        x= np.random.choice(self.x[:,0], size=num_eles, replace=False)
        y= np.random.choice(self.x[:,1], size=num_eles, replace=False)
        # print(self.x[:,0], np.shape(self.x[:,0]), sep='\n'); input(11)
        # print(self.y_true[:,0], np.shape(self.y_true[:,0]), sep='\n'); input(22)
        z= np.random.choice(self.y_true[:,0], size=num_eles, replace=False)
        # print(x, np.shape(x), sep='\n' )
        # print(y, np.shape(y), sep='\n' )
        # print(z, np.shape(z), sep='\n' )
        # x= self.x[:,0]
        # y= self.x[:,1]
        # X, Y = np.meshgrid(x, y)
        
        # Creating a regular grid
        # print(x.max()); input(11)
        # xi, yi = np.linspace(x.min(), x.max(), 200), \
        #     np.linspace(y.min(), y.max(), 200)
        xi, yi = np.meshgrid(x, y)
        # print(self.y_true)
        # interpolate irregular data onto the regular grid-.
        # zi = griddata((x, y), self.y_true, (xi, yi), method='cubic')
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        # creating 3D plot-.
        fig= plt.figure()
        ax= fig.add_subplot(111, projection='3d')
        
        # plotting the 3D surface from irregular data using grid interpolation-.
        ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='k')

        # customizing the plot-.
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Surface Plot from Irregular Data (Grid Interpolation)')

        # displaying the plot-.
        plt.show()

        '''
        # plot the 3D surface-.
        ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                        alpha=0.3)
        
        # plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph-.
        ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
        ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
        ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
        
        # ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
        #       xlabel='X', ylabel='Y', zlabel='Z')
        plt.show()
        '''

        return None
