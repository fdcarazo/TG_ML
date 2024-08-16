#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# script to do an EDA analysis in DataSet used to train and validate
# ANN's  model to predict ELASTIC EFFECTIVE properties in fiber's composites-.
#
# synthetic DATASET obtained using PMM's Barberos model-.
# provided by Dr. Ing. Dario Barulich - CONICET - UTN - FRC -.
#
# @author: Fernando Diego Carazo (@buenaluna) -.
#
# start_date (Arg): dom jun 16 10:22:29 -03 2024-.
# last_modify (Arg): -.
##
# ======================================================================= END79

# ======================================================================= INI79
# include modulus-.
from pathlib import Path
import numpy as np
import time as time

# from version import mod_versions as mv
from src.read_config_file import Config as cfg
from src.dataset import DataObject as do

import utils  # IMPORTANT: if I don't write this 2plot/utils/__init__.py is not read-.
from utils.load_df import load_ds as ld
from utils.gen_tools import get_args as ga
from utils.gen_tools import convert_to_preferred_format

from exploratory import Data_Analysis
from missValues import Missing_Values
from outliers import Outliers
from feat_targ_boxplots import Feat_Targ_Boxplots
from feat_targ_distribution import Feat_Targ_Distributions
from bivariate_analysis import Bivariate_Analysis_Plots

from feature_selection import Feature_Selection
from tuning_df import Tuning_Data_Frame
# ======================================================================= END79


# ======================================================================= INI79
def main(config) -> int:
    '''
    main: driver -.
    '''
    # start_time
    start_tot_ex_time = time.time()
    
    # import sys
    # sys.path.append('/home/fcarazo/my_github/predictElasticAnysNN/models/')
    # 0- -.
    cfg_obj = cfg(config)

    # 1- load dataset (2test), as pandas.DataFrame, features and targers vars
    #    names-.
    df = ld(str.split(cfg_obj.ds_file, '.')[-1],
            cfg_obj.ds_path+cfg_obj.ds_file,
            cfg_obj.sheet_name, cfg_obj.vars_names)

    '''
    import matplotlib.pyplot as plt
    x = df.iloc[:, 4]
    y = df.iloc[:, 1]
    plt.scatter(x, y)
    plt.show()
    input(99)
    '''

    df.drop(columns=['Unnamed: 0','biomass_type'], inplace=True)
    
    # print(df); print(df.columns); print(df.shape); input(33)
    # print(df1[cfg_obj.feat_names]); print(df1[cfg_obj.feat_names].columns);
    # print(df.shape); input(33)
    
    # create an Object/instance of dataAnalysis class-.
    # analysis = AnalisisDatos(df.sample(frac=1))
    analysis = Data_Analysis(df, cfg_obj.feat_names, cfg_obj.targ_names)

    '''  # BLOCK__1 -- INI -.
    # run different procedures/functions methods to do analysis-.
    analysis.info_and_descriptive_statistics()

    missing_values_obj = Missing_Values(df, cfg_obj.feat_names,
                                        cfg_obj.targ_names)
    missing_df = missing_values_obj.missing_values_table()
    df_droped = missing_values_obj.columns_porc_missing(10, missing_df)
    print(df_droped)

    # check outliers: detect and remove-.
    outliers_obj = Outliers(df, cfg_obj.feat_names, cfg_obj.targ_names)
    df_wo = outliers_obj.detect_remove_outliers()
    
    # create an object without outliers.-
    # analysis = Data_Analysis(df_wo, cfg_obj.feat_names, cfg_obj.targ_names)
    # plots without ouliers (I don't do that beacuse in this case I don't have
    # outliers)-.
    # ...

    # UNIVARIATE distribution plots of Features and Targets-.
    # features and targets distribution-.
    analysis.univ_hist_plots('distribution')
    # features and targets statistics plots (.describe()) as bar-.
    analysis.univ_hist_plots('var_stat')
    
    # to analyze VARIABILITY of features and targets.
    # yFeatures and Targets Boxplots-.
    feat_targ_boxplots_obj = Feat_Targ_Boxplots(df, cfg_obj.feat_names,
                                                cfg_obj.targ_names)
    feat_targ_boxplots_obj.plot_boxplots()
    
    # univariate features and targets values distribution-.-.
    # features and targets distribution + (plus) values in 'y' axe-.
    feat_targ_distr_obj = Feat_Targ_Distributions(df, cfg_obj.feat_names,
                                                  cfg_obj.targ_names)
    feat_targ_distr_obj.plot_feat_target_dist(10,
                                              feat_targ_distr_obj.__class__.__name__)
    '''  # BLOCK__1 -- END -.

    '''  # BLOCK__2 -- INI -.
    # Correlation between Features and Targets
    # Analyze correlations between features and between features and targets-.
    # This information is very usefull to do feature engineering and
    # features selection-.
    analysis.correlation(analysis.__class__.__name__)
    analysis.plot_corr(3)  # 1 ===> option (there are more than one option to plot)-.
    input(11)
    '''  # BLOCK__2 -- END -.

    '''  # BLOCK__3 -- INI -.
    # Bivariate_analysis-.
    # bivariate_analysis ==> bivariate plots-.
    bivariate_obj = Bivariate_Analysis_Plots(df, cfg_obj.feat_names,
                                             cfg_obj.targ_names)
    bivariate_obj.bivariate_plot(1.0, bivariate_obj.__class__.__name__)
    analysis.features_bivariate(analysis.__class__.__name__)
    '''  # BLOCK__3 -- END -.

    # BLOCK__4 -- INI -.
    # Feature Engineering and Selection-.
    # ==================================
    # Feature Engineering:
    # ====================
    # Creating new features that allow a machine learning model to learn a
    # mapping between these features and the target. This might mean taking
    # transformations of variables (for example log and square root, or
    # one-hot encoding categorical variables so they can be used in a model).
    # Generally:
    # FEATURE ENGINEERING: add additional features derived from the raw data-.
    # Feature Selection:
    # ==================
    # Process of choosing the most relevant features in your data. "Most
    # relevant" can depend on many factors, but it might be something as
    # simple as the highest correlation with the target, or the features
    # with the most variance. In feature selection, we remove features
    # that do not help our model learn the relationship between features
    # and the target. This can help the model generalize better to new
    # data and results in a more interpretable model.
    # FEATURE SELECTION: subtract features so we are left with only those
    # that are most important.
    #
    #
    # Additional Feature Selection
    # ============================
    # There are plenty of more methods for feature selection. Some popular
    # methods include principal components analysis (PCA) which transforms
    # the features into a reduced number of dimensions that preserve the
    # greatest variance, or independent components analysis (ICA) which
    # aims to find the independent sources in a set of features-.
    
    # feature engineering using linear_Model and Recursive_Model_Elimination-.
    feat_sel_obj = Feature_Selection(df, cfg_obj.feat_names,
                                     cfg_obj.targ_names)
    feat_sel_obj.get_feature_selection(feat_sel_obj.__class__.__name__)
    
    input('Press_Enter')
    
    # Statsitical Inferential analysis-.
    analysis.classical_inferential_analysis(analysis.__class__.__name__)
    # analysis.bayesian_inferential_analysis(analysis.__class__.__name__)
    # analysis.logistic_inferential_analysis(analysis.__class__.__name__)
    # BLOCK__4 -- END -.
    
    '''  # BLOCK__5 -- INI -.
    # 4 modify PandasDataframe-.
    tuning_df_obj = Tuning_Data_Frame(df, cfg_obj.feat_names,
                                      cfg_obj.targ_names)
    df_edited = tuning_df_obj.get_edited_df(tuning_df_obj.__class__.__name__)
    # print(df_edited)
    # Remove Collinear Features
    df_wcc = tuning_df_obj.remove_collinear_features(df_edited, 0.6)
    # print(df.columns, df_edited.columns, df_wcc.columns, sep='\n')
    '''  # BLOCK__5 -- END -.

    #
    # control TOTAL EXECUTION TIME-.
    end_tot_ex_time=time.time()
    print('Total execution time {0}{1}'.
          format('\n',convert_to_preferred_format(end_tot_ex_time-start_tot_ex_time)\
                 )\
          )

    return 0
# ======================================================================= END79

# ======================================================================= INI79
if __name__ == '__main__':    
    # my auxiliaries methods-.
    # from utils.gen_tools import *
    # print(dir())
    config_file = Path(__file__).parent/'config_file.yaml'
    config = ga(config_file)
    
    # list the name and versions of the main modules used-.
    # a=mv(); a.open_save_modules()

    # call main-.
    val = main(config)
else:
    print('{0} imported as Module'.format(__file__.split('/')[-1]))
# ======================================================================= END79
