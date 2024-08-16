#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## class to define the functions to plot predictions-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date (Fr): Tue Mar 19 12:33:47 CET 2024-.
## last_modify (Fr): -.
## last_modify (Ar): jue abr 11 11:41:17 -03 2024-.
##
## ======================================================================= INI79

## print(dir()); input(1)

## 1- IMPORT MODULUS ---.
## Import the required packages/libraries/modules-.
## 1-1- GENERAL MODULES -.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import os

## from sklearn.metrics import PredictionErrorDisplay

class Predict():
    def __init__(self,y_true,y_pred,feat_scal,targ_scal,feat_vars,targ_vars):
        ## MBC, add one label to control if features and targets
        ## were standarized or scaled-.
        ## Apply inverse scaling in a cell before the figure cells-.
        self.y_true=targ_scal.inverse_transform(y_true)
        self.y_pred=targ_scal.inverse_transform(y_pred)
        self.feat_vars=feat_vars
        self.targ_vars=targ_vars
    
    def plot_corr_true_pred(self):
        ''' 
        plot correlation between predicted and true values, i.e.:
        plot self.y_true vs. self.y_predicted
        '''
        fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(12,8))
        # parameters at the axes level-.
        axes[0][0].scatter(self.y_true[:,0]/1.0e3 ,self.y_pred[:,0]/1.0e3 ,color='r',marker='1')
        x00=1.0e-3*np.array([self.y_true[:,0].min(),self.y_true[:,0].max()]) # instead of [0,1] as the plot border
        y00=x00
        axes[0][0].plot(x00,y00)

        axes[0][1].scatter(self.y_true[:,1]/1.0e3 ,self.y_pred[:,1]/1.0e3,color='g',marker='2')
        x01=1e-3*np.array([self.y_true[:,1].min(),self.y_true[:,1].max()])
        y01=x01
        axes[0][1].plot(x01,y01)
        
        axes[0][2].scatter(self.y_true[:,2],self.y_pred[:,2],color='b',marker='3')
        x02=np.array([self.y_true[:,2].min(),self.y_true[:,2].max()])
        y02=x02
        axes[0][2].plot(x02,y02)
        
        axes[1][0].scatter(self.y_true[:,3],self.y_pred[:,3],color='y',marker='+')
        x10=np.array([self.y_true[:,3].min(),self.y_true[:,3].max()])
        y10=x10
        axes[1][0].plot(x10,y10)
        
        axes[1][1].scatter(self.y_true[:,4]/1.0e3,self.y_pred[:,4]/1.0e3,color='c',marker='*')
        x11=1e-3*np.array([self.y_true[:,4].min(), self.y_true[:,4].max()])
        y11=x11
        axes[1][1].plot(x11,y11)
        #axes[1][1].ticklabel_format(useOffset=True)
        
        axes[1][2].scatter(self.y_true[:,5]/1.0e3 ,self.y_pred[:,5]/1.0e3,color='black',marker='4')
        x12 = 1e-3*np.array([self.y_true[:,5].min(),self.y_true[:,5].max()])
        y12=x12
        axes[1][2].plot(x12,y12)
        
        #___________________________________
        # # ['E1','v12','E3','v13','G12']  Elastic properties of Woven RVE 
        axes[0][0].set_xlabel('Expected $E_{11} (\mathrm{GPa})$')
        axes[0][1].set_xlabel('Expected $E_{33} (\mathrm{GPa})$')
        axes[0][2].set_xlabel('Expected $v_{12}$')
        axes[1][0].set_xlabel('Expected $v_{23}$')
        axes[1][1].set_xlabel('Expected $G_{12} (\mathrm{GPa})$')
        axes[1][2].set_xlabel('Expected $G_{23} (\mathrm{GPa})$')
        
        axes[0][0].set_ylabel('Predicted $E_{11} (\mathrm{GPa})$')
        axes[0][1].set_ylabel('Predicted $E_{33} (\mathrm{GPa})$')
        axes[0][2].set_ylabel('Predicted $v_{12}$')
        axes[1][0].set_ylabel('Predicted $v_{23}$')
        axes[1][1].set_ylabel('Predicted $G_{12} (\mathrm{GPa})$')
        axes[1][2].set_ylabel('Predicted $G_{23} (\mathrm{GPa})$')
        
        from matplotlib.ticker import (MultipleLocator,
                                       FormatStrFormatter,
                                       AutoMinorLocator)
        # axes[0][0].set_xlim(1e-3*np.array([self.y_true1[:,0].min(), self.y_true1[:,0].max()])) #set the starting point
        # axes[0][0].set_ylim(1e-3*np.array([self.y_true1[:,0].min(), self.y_true1[:,0].max()]))
        axes[0][0].xaxis.set_major_locator(MultipleLocator(25))
        axes[0][0].yaxis.set_major_locator(MultipleLocator(25))
        axes[0][1].xaxis.set_major_locator(MultipleLocator(10))
        axes[0][1].yaxis.set_major_locator(MultipleLocator(10))
        axes[0][2].xaxis.set_major_locator(MultipleLocator(.11))
        axes[0][2].yaxis.set_major_locator(MultipleLocator(.11))
        axes[1][0].xaxis.set_major_locator(MultipleLocator(.17))
        axes[1][0].yaxis.set_major_locator(MultipleLocator(.17))
        axes[1][1].xaxis.set_major_locator(MultipleLocator(2))
        axes[1][1].yaxis.set_major_locator(MultipleLocator(2))
        axes[1][2].xaxis.set_major_locator(MultipleLocator(2))
        axes[1][2].yaxis.set_major_locator(MultipleLocator(2))
            
        ## an important line to adjust the plots-.
        plt.tight_layout()
        ## Use left,right,top, bottom to stretch subplots-.
        ## Use wspace,hspace to add spacing between subplots
        fig.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=0.45,
                            hspace=0.45,)

        #______set_title('mesoscale 3D elastic properties. Predicted by ANN vs. FE simulations in Digimat')
        #______input data extended from microscale ANN model. initial data set with 400 points and the 
        # extended one with 600 data points.
        #plt.savefig("woven_RVE_elastic_coef.pdf", format="pdf", bbox_inches="tight")

        #adjust padding:
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

        plt.plot()
        plt.show()

    def plot_corr_true_pred_mod(self,dir_save):
        ''' 
        plot correlation between predicted and true values-.
        '''
        
        plt.rcParams.update({'font.size': 8})
        gs=gridspec.GridSpec(2,3) # 6 subfigures (function of number of targets vars)-.

        fig=plt.figure(figsize=(15, 10))
        palette=sns.color_palette('mako_r',4)
        
        for idx,col in enumerate(self.targ_vars,start=0):
            ax=fig.add_subplot(gs[idx])
            ## var= ['E' in a or 'G' in a for a in s if 'E' in a or 'G' in a]
            if 'E' in col or 'G' in col: 
                y_true=self.y_true[:,idx]/1.0e+3; y_pred=self.y_pred[:,idx]/1.0e3
            else:
                y_true=self.y_true[:,idx]; y_pred=self.y_pred[:,idx]

            plt.scatter(y_true,y_pred,color='r',marker='o',label=str(col),alpha=0.5)
            
            pearR=np.corrcoef(y_true,y_pred)[1,0]
            A=np.vstack([y_true,np.ones(len(y_true))]).T
            m,c=np.linalg.lstsq(A,y_true)[0]
            plt.plot(y_true,y_true*m+c,color='b',linestyle='--',label='Fit -- r = %6.4f'%(pearR))
            plt.legend(loc=2)

            plt.grid()
            plt.xlabel('true'); plt.ylabel('predicted')
            
            plt.tight_layout()
        
            ticks,labels=plt.xticks()
            ## plt.xticks(ticks[::1], labels[::1])
            ## plt.xlim([0.0,2.0])
        
            ## fig.legend(*ax.get_legend_handles_labels(),loc='lower center', ncol=4)
        ## plt.legend(lines,labels,loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
        ## 2 use \ bar in latexMode-.
        ## https://stackoverflow.com/questions/65702154/
        ## problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
        
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        ## label= r'$abs(\Delta C_{{ij}})= |C_{{ij}}^{{VPSC}}-C_{{ij}}^{{PREDICTED}}|= f(\bar{\varepsilon})$'## +\
            ##    ' , for {0}'.format(self.i_run)
        ## r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
            ## r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
            
        ##fig.text(0.6,0.1,label,color='r',fontsize=16,
        ##         horizontalalignment='right',verticalalignment='top',
        ##         backgroundcolor='1.0')
        plt.show()
        fig.savefig(os.path.join(dir_save,'correlationFigs.png'),format='png',dpi=100)
    '''
    def plot_pred_error_display(self):
        fig,ax=plt.subplots(1,1,figsize=(9, 7))
        display=PredictionErrorDisplay.from_predictions(
            y_true=self.y_true,
            y_pred=self.y_pred,
            kind="actual_vs_predicted",
            ax=ax,
            scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
            line_kwargs={"color": "tab:red"},
        )
        elapsed_time=0.2
        name='MLP'
        ax.set_title(f"{name}\nEvaluation in {elapsed_time:.2f} seconds")
        
        ##for name, score in scores.items():
        ##    ax.plot([], [], " ", label=f"{name}: {score}")
        ##ax.legend(loc="upper left")

        plt.suptitle("Single predictors versus stacked predictors")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    '''

    def plot_with_uncertainty(self,x,list_dfs,bounds_,bounds_vl_,ds_name:str,dir_save:str,quart:int)->int:
        '''
        plot ----
        '''
        y_mean=np.mean(list_dfs,axis=0)
        y_std=np.std(list_dfs,axis=0)
        
        nr,nc=2,3 # if str.lower(dim)=='2d' else (7,6) if dim=='3d' else (None,None)
        plt.rcParams.update({'font.size': 10}); gs=gridspec.GridSpec(nr,nc)
        fig=plt.figure(figsize=(15, 10))
        sns.color_palette('mako_r',4)
        
        ## feature_in_var-.
        for idx, col in enumerate(self.targ_vars,start=0):
            ax=fig.add_subplot(gs[idx])  # , sharex=True, sharey=False)
            # true values-.
            var_to_plot=str(col)
            x=x  # list_dfs[0].phiF

            if 'E' in col or 'G' in col:
                ## 
                plt.scatter(x=x,y=y_mean[:,idx]/1.0e+3,s=10,facecolors='none',
                            edgecolor='k',marker='^',label='FFNN (Feed Forward Neural Network)'
                            ## edgecolor='k',marker='^',label='BNN (Bayesian Neural Network)'
                            # alpha=0.1, c='blue',
                            )
                plt.fill_between(x,
                                 (y_mean[:,idx]+quart*y_std[:,idx])/1.0e+3,
                                 (y_mean[:,idx]-quart*y_std[:,idx])/1.0e+3, 
                                 alpha=0.5,label='Epistemic uncertainty'
                                 )
            else:
                plt.scatter(x=x,y=y_mean[:,idx],s=10,facecolors='none',
                            edgecolor='k',marker='^',label='FFNN (Feed Forward Neural Network)'
                            ## edgecolor='k',marker='^',label='BNN (Bayesian Neural Network)'
                            # alpha=0.1, c='blue',
                            )
                plt.fill_between(x,
                                 y_mean[:,idx]+quart*y_std[:,idx],
                                 y_mean[:,idx]-quart*y_std[:,idx], 
                                 alpha=0.5,label='Epistemic uncertainty'
                                 )
            plt.axvline(x=bounds_vl_['lower'],ymin=0.0,ymax=1.0,linewidth=2,linestyle='--',color='r')
            plt.axvline(x=bounds_vl_['upper'],ymin=0.0,ymax=1.0,linewidth=2,linestyle='--',color='r')
            ## plt.legend(loc=3)
            plt.grid()
            plt.xlabel(r'$\phi_{frac}$')
            
            ## col=str(col).replace('_out','')
            ##plt.ylabel(r'$\Delta$'+
            ##           '{0}'.format(col)
            ##           )
            plt.ylabel(str(col))

            plt.tight_layout()
        
            ticks, labels=plt.xticks()
            ## plt.xticks(ticks[::1], labels[::1])
            plt.xlim([0.0,1.0])
            
            fig.legend(*ax.get_legend_handles_labels(),
                       loc='lower center', ncol=4)
        
        ## problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        
        #label= r'Properties prediction in function of $\phi_{frac}$'
        ##label= r'$C_{{ij}}^{{VPSC}}-C_{{ij}}^{{REC-PRED}}-C_{{ij}}^{{NON-REC-PRED}}= '\
        ##    r'f(\bar{\varepsilon})$'## +\

            ##    ' , for {0}'.format(self.i_run)
            ## r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
            ## r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
                
        ##fig.text(0.7,0.12,label,color='r',fontsize=16,
        ##         horizontalalignment='right',verticalalignment='top',
        ##         backgroundcolor='1.0')
        
        plt.show()
        fig.savefig(os.path.join(dir_save,'predWith_WithUncertainty.png'),format='png',dpi=100)

        return 0

    def plot_residuals(self,dir_save)-> int:
        ''' plot residuals  between predicted and true values-.'''
        plt.rcParams.update({'font.size': 8})
        gs=gridspec.GridSpec(2,3) # 9 subfigures (function of number of targets vars)-.

        fig=plt.figure(figsize=(15, 10))
        palette=sns.color_palette('mako_r',4)
        
        for idx,col in enumerate(self.targ_vars,start=0):
            ax=fig.add_subplot(gs[idx])
            y_true=self.y_true[:,idx]/1.0e+3;y_pred=self.y_pred[:,idx]/1.0e3
            res=y_true-y_pred
            sns.distplot(res,kde=True,vertical=False,rug=True,hist=False,kde_kws=dict(fill=True),
                         rug_kws=dict(lw=2, color='orange'))
            plt.grid()
            ## plt.xlabel('true'); plt.ylabel('predicted')
            plt.xlabel(str(col))
            
            plt.tight_layout()
        
            ticks,labels=plt.xticks()
            ## plt.xticks(ticks[::1], labels[::1])
            ## plt.xlim([0.0,2.0])
        
            ## fig.legend(*ax.get_legend_handles_labels(),loc='lower center', ncol=4)
        ## plt.legend(lines,labels,loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
        ## 2 use \ bar in latexMode-.
        ## https://stackoverflow.com/questions/65702154/
        ## problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
        
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        ## label= r'$abs(\Delta C_{{ij}})= |C_{{ij}}^{{VPSC}}-C_{{ij}}^{{PREDICTED}}|= f(\bar{\varepsilon})$'## +\
            ##    ' , for {0}'.format(self.i_run)
        ## r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
            ## r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
            
        ##fig.text(0.6,0.1,label,color='r',fontsize=16,
        ##         horizontalalignment='right',verticalalignment='top',
        ##         backgroundcolor='1.0')
        plt.show()
        fig.savefig(os.path.join(dir_save,'residualsFigs.png'),format='png',dpi=100)

        plt.show()
        return 0

        
