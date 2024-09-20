
#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
# script to Load, Train, Test and HYPERPARAMETERS OPTIMIZZTION DL
# models to predict the biochart of pyrolysis-.
#
# DATASET Provided by Dra.-Ing. Anabel Fernandez
# IIQ - FI -UNSJ (associated unit of PROBEin - CONICET - UNCO) -.
#
# @author: Fernando Diego Carazo (@buenaluna) -.
#
# start_date (Fr): dom 01 sep 2024 08:48:44 -03-.
# last_modify (Arg): vie 06 sep 2024 17:50:44 -03-.
##
# ======================================================================= END79

# ======================================================================= INI79
# include modules/packages/libraries-.
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time as time
import joblib
from model.weight_init import weights_init
# import sys

# from version import mod_versions as mv

# from src folder/directory-.
from src.read_config_file import Config as cfg
from src.eda_analysis import Eda
from src.test_val import test_val as tv

from src.dataset import DataObject as do
from src.dataset import DatasetTorch as datas  # torch.utils.data.Dataset

from src.dataset import DataloaderTorch as datal  # torch.utils.data.DataLoader

# from utils folder/directory-.
import utils  # IMPORTANT: if I don't write this 2plot/utils/__init__.py is not read-.
from utils.load_df import load_ds as ld
from utils.gen_tools import get_args as ga
from utils.gen_tools import convert_to_preferred_format
from utils.predict import Predict
from utils.predict_values import Predict_NN

from model.RegressionDataset import RegressionDataset
from model.RegressionDataLoader import RegressionDataLoader
from model.plot_losses import PlotLosses
from model.save import SaveDLModelLoss

# 1- FFNN-.
from model.ffnn import MLP
from model.train_val_FFNN import TrainPredict as TrainPredict_FFNN

# 2- bayesianNN-.
import torchbnn as bnn
from model.ffnn_bnn import BayesianNet
from model.train_val_BNN import TrainPredict as TrainPredict_BNN

# Hyperparameter Optimization-.
# optuna-.
import optuna as optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history

from model.nn_hyper_param import nn_hyper_param_optim
from model.nn_for_optuna import RegressionModel

from model.train_val_FFNN_optuna_1 import TrainPredict as TrainPredict_FFNN_opt

# 2don't have problems between float data type of torch and bnnNN-.
# if torch.is_tensor(xx) else torch.tensor(xx,dtype=float)  # (float64) and float

# of NN (float32)-.
torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)

from torch import tensor as tt, float32 as tf32  # if you have problems import FloatTensor()
# ======================================================================= END79


# ======================================================================= INI79
def main(config) -> int:
    ''' main: driver
    Args:
    ====
    config: config_file-.
    Returns:
    ========
    int-.
    '''
    
    # total start_time-.
    start_tot_ex_time = time.time()

    # to add specific folder to  my path-.
    # sys.path.append('/home/fdcarazo/my_github/tga_project/model/')
    
    # 0- -.
    cfg_obj = cfg(config)
    # print(cfg_obj.ds_path, cfg_obj.ds_file, cfg_obj.vars_names); input(11)

    # 1- load dataset as pandas.DataFrame, features and targers vars names-.
    df = ld(cfg_obj.ds_path+cfg_obj.ds_file, cfg_obj.vars_names)
    
    '''
    vars = cfg_obj.feat_names + cfg_obj.targ_names
    for var_i in vars: print(var_i, df[var_i].min(), df[var_i].max(),
                             df[var_i].mean(), df[var_i].median()); input(1)
    print(df.shape)
    df.dropna(inplace=True)
    print(df.shape)
    rows_with_nan = df[df.isnull().any(axis=1)]
    print(rows_with_nan)
    input(11)
    '''
    
    '''
    print(df); print(df.describe()); print(df.columns); print(df.shape), input(33)
    print(df[cfg_obj.feat_names]); print(df[cfg_obj.feat_names].columns); print(df.shape); input(33)
    '''

    '''
    # 2- split df in train-val/test datasets-.
    tv_obj = tv(df, cfg_obj.feat_names, cfg_obj.targ_names)  # create test_val Object-.
    X_train, X_val, y_train, y_val = tv_obj.train_val(cfg_obj.rand,
                                                      cfg_obj.test_frac,
                                                      cfg_obj.shuffle
                                                      )
    X_train_sca, X_val_sca, y_train_sca, y_val_sca = tv_obj.scaler(cfg_obj.scaler,
                                                                   X_train,
                                                                   X_val,
                                                                   y_train,
                                                                   y_val)
    
    # print(X_train, X_val, y_train, y_val,sep='\n')
    # tv_obj.plot_train_val()  # plot distr of feat & targ in train and val datasets-.
    
    # 3- convert and get torch.Dataset and torch.DataLoader-.
    # 3-1- torchDataset-.
    train_dataset_obj = RegressionDataset(X_train_sca, y_train.to_numpy(dtype=float))
    val_dataset_obj = RegressionDataset(X_val_sca, y_val.to_numpy(dtype=float))
    # 3-2- torchDataLoader-.
    train_loader_obj = RegressionDataLoader(train_dataset_obj,
                                            cfg_obj.batch_size,
                                            cfg_obj.shuffle_torch)
    val_loader_obj = RegressionDataLoader(val_dataset_obj,
                                          cfg_obj.batch_size,
                                          cfg_obj.shuffle_torch)
    print(len(train_dataset_obj), len(val_dataset_obj), sep='\n')
    '''

    # 2- dataset, standarscaler, and Pytorch Dataset and DataLoader-.
    # 2-1- dataset as PandasDataframe-.
    do_obj = do(df, cfg_obj.feat_names, cfg_obj.targ_names)
    # 2-2- standarization-scaler and train and validation split-.
    df_scal, df_feat_scal, feat_scal, targ_scal = do_obj.scalerStandarize(cfg_obj.scaler_type,
                                                                          cfg_obj.scale_targ,
                                                                          cfg_obj.dir_save)
    # print(feat_scal, type(feat_scal), sep='\n'); input(12345)
    df_split = df_scal if cfg_obj.scale_targ else df_feat_scal
    X_train, X_val, y_train, y_val = do_obj.train_test(df, df_split, 42,
                                                       cfg_obj.test_frac, True)
    '''
    print(type(X_train), type(X_val), type(y_train), type(y_val)); input(33)
    print(np.shape(X_train), np.shape(X_val), np.shape(y_train), np.shape(y_val)); input(44)
    '''
    
    # 2-3- Pytorch Dataset-.
    dataset_pytorch_train = datas(X_train, y_train)
    dataset_pytorch_val = datas(X_val, y_val)
    # print(list(dataset_pytorch_train), list(dataset_pytorch_val), sep='\n')
    
    # 2-4- Pytorch DataLoader-.
    train_loader_1 = datal(dataset_pytorch_train, cfg_obj.batch_size,
                         cfg_obj.shuffle)
    train_loader = torch.utils.data.DataLoader(dataset_pytorch_train,
                                               batch_size=cfg_obj.batch_size, shuffle=True)
    val_loader_1 = datal(dataset_pytorch_val, np.shape(y_val)[0], False)
    val_loader = torch.utils.data.DataLoader(dataset_pytorch_val, np.shape(y_val)[0], False)

    # 3- train and validation loop-.
    # DL MODELS - training and validation  == INI ==-.
    input_size, output_size = np.shape(X_train)[1],\
        np.shape(y_train)[1]  # len(feat_var), len(tar_var)

    # OPTUNA -- ini-.
    if cfg_obj.hyper_optim_apply:  # 2do Hyperparameter Optimization (boolean value: True/False)-.
        if cfg_obj.hyper_optim_name == 'optuna':
            # create the model (Optuna object)-.
            study = optuna.create_study(direction='minimize')
            # https://optuna.readthedocs.io/en/latest/faq.html#how-to-define-objective-functions-that-have-own-arguments
            # https://www.kaggle.com/code/corochann/optuna-tutorial-for-hyperparameter-optimization/notebook
            # https://www.kaggle.com/code/bextuychiev/no-bs-guide-to-hyperparameter-tuning-with-optuna/notebook
            nn_hyper_param_optim_obj = nn_hyper_param_optim(train_loader,
                                                            val_loader, cfg_obj.lr,
                                                            eval(cfg_obj.optimizer),
                                                            eval(cfg_obj.loss),
                                                            cfg_obj.weight_decay,
                                                            cfg_obj.momentum)
            func = lambda trial: nn_hyper_param_optim_obj.objective(trial, input_size,
                                                                    output_size, cfg_obj.optuna_epochs)
            study.optimize(func, n_trials=20, n_jobs=-1)
            
            '''
            # optimize hyperparameters using Optuna
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)
            '''
            
            # print best hyperparameters
            # print('Best hyperparameters: ')
            # for key, value in study.best_params.items(): print(f'{key}: {value}')
            
            print(f'Best value (validation loss): {study.best_value}')  # 

            '''
            # plot training and validation loss
            best_train_losses = study.best_trial.user_attrs['train_loss']
            best_val_losses = study.best_trial.user_attrs['val_loss']
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(best_train_loss, label='Train Loss')
            plt.plot(best_val_loss, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.show()
            input(22)
            '''
            
            # save the best model and scaler-.
            best_trial_params = study.best_params
            ## print(study.best_params.items); input(99)
            input_size = len(cfg_obj.feat_names)   # input size-.

            best_model = RegressionModel(input_size, output_size,
                                         best_trial_params['hidden_layers'],
                                         best_trial_params['hidden_units'],
                                         best_trial_params['dropout_prob'],
                                         best_trial_params['learning_rate'],                                         
                                         eval(cfg_obj.optimizer),
                                         eval(cfg_obj.loss),
                                         cfg_obj.weight_decay,
                                         cfg_obj.momentum)
            '''
            best_model = RegressionModel(input_size, output_size,
                                         best_trial_params['hidden_layers'],
                                         best_trial_params['hidden_units'],
                                         best_trial_params['dropout_prob'])
            '''
            
            # load the best trial's parameters and save the model-.
            # best_model.load_state_dict(torch.load('best_model.pth'))
            # joblib.dump(scaler, 'best_scaler.pkl')

            '''
            # only For GNU/Linux: print some methods of study Object get the best
            # hyperparamters finded-.
            print('Best_trial_values: {0}{1}'.format(study.best_trial, '\n'))
            print('Best_params: {0}{1}'.format(study.best_params, '\n'))
            print('{0}{1}Params: '.format('\n', '\t'))        
            for key, value in study.best_trial.params.items():
                print(' {0}: {1}'.format(key, value))
            '''

            ''' # if I decide to use the same loop for training and validate -.
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # to use in final predictions (in this case it isn't necesary)-.
            print(f"optim.{study.best_params['optimizer']}")
            optimizer = eval(f"optim.{study.best_params['optimizer']}")  # optim.Adam
            best_optimizer = optimizer  # bestoptim.Adam(best_model.parameters(), lr=1.0e-2)
            criterion = torch.nn.MSELoss()
            best_criterion = criterion  # best_criterion= nn.MSELoss()
            '''
        
            if cfg_obj.plot_optim_hist:
                plotly_config = {'staticPlot': True}
                fig = plot_optimization_history(study)
                fig.show(config=plotly_config)
                plot_optimization_history(study)
                # import matplotlib.pyplot as plt
                # plt.show()
            
                from optuna.visualization import plot_param_importances
                fig = plot_param_importances(study)
                fig.show(config=plotly_config)
                fig.show(config=plotly_config)
                
                # plt.show()
            
        # input(88)
        #  .... AFTER WE GET BEST MODEL-.
        # final values to train and validate the results with the best hyperparameters
        # (to plot loss and other thing, if not only need to validate not train)-.
        ## best_model= RegressionModel(input_size, best_params['hidden_size'], output_size, best_params['dropout_prob'])
        ## best_optimizer= torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
        ## best_criterion= nn.MSELoss()
        
        # Prediction -.
        # https://perlitz.github.io/hyperparameter-optimization-with-optuna/

        if cfg_obj.plot_loss:
            # plot losses (training and validation)-.
            plotlosses_obj = PlotLosses('_best_Opt')
            plt_pl = lambda x, y: plotlosses_obj.plot_loss(x, y, cfg_obj.dir_save)
            plt_pl('', study.best_trial.user_attrs['all_loss'])
            # plt_pl('train', ld_['train_loss'])
            # plt_pl('val', ld_['val_loss'])
            # plotlosses_obj.plot_loss('val', ld_['val_loss'])

        # SAVE Block -.
        # save model-.
        best_model.save_params_model(cfg_obj.dir_logs, cfg_obj.batch_size,
                                     cfg_obj.batch_size,  # bath_size_{train,test}
                                     cfg_obj.epochs, cfg_obj.optuna_epochs, cfg_obj.optimizer,
                                     cfg_obj.loss,  # I need str and not torch..-.
                                     cfg_obj.ds_file, cfg_obj.dir_results, '_best_Opt')
        # save train and validatiton losses and DL's model-.
        if cfg_obj.model_loss_save:
            save_dl_loss_obj = SaveDLModelLoss(cfg_obj.dir_save, '_best_Opt')
            save_dl_loss_obj.save_model(best_model)
            save_dl_loss_obj.save_loss(study.best_trial.user_attrs['train_loss'])
            joblib.dump(best_trial_params, cfg_obj.dir_save+'best_params.pkl')
        print('Best model and scaler saved.')
        
        # input(8888)
        
        # TRAIN BEST MODEL-.
        '''
        best_trial_params = study.best_params
        input_size = len(cfg_obj.feat_names)   # input size-.
        best_model = RegressionModel(input_size, output_size,
                                     best_trial_params['hidden_layers'],
                                     best_trial_params['hidden_units'],
                                     best_trial_params['dropout_prob'])

        '''

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        '''
        train_predict_obj = TrainPredict_FFNN_opt(best_model,
                                                  best_trial_params['hidden_layers'],
                                                  best_trial_params['hidden_units'],
                                                  best_trial_params['dropout_prob'],
                                                  train_loader_1.dataloader,
                                                  val_loader_1.dataloader,
                                                  input_size, output_size,
                                                  device, best_trial_params['learning_rate'],
                                                  cfg_obj.optimizer,)
        '''
        train_predict_obj = TrainPredict_FFNN_opt(best_model,
                                                  train_loader,
                                                  val_loader,
                                                  device,)
        
        train_time, ld_ = train_predict_obj.train(cfg_obj.epochs,
                                                  val=True)  # train_time and loss_dict-.
        '''
        if cfg_obj.plot_loss:
            # plot losses (training and validation)-.
            plotlosses_obj = PlotLosses()
            plt_pl = lambda x, y: plotlosses_obj.plot_loss(x, y, cfg_obj.dir_save)
            plt_pl('', ld_)
            # plt_pl('train', ld_['train_loss'])
            # plt_pl('val', ld_['val_loss'])
            # plotlosses_obj.plot_loss('val', ld_['val_loss'])
        '''

        '''
        # save train and validatiton losses and DL's model-.
        if cfg_obj.model_loss_save:
            save_dl_loss_obj = SaveDLModelLoss(cfg_obj.dir_save, , '_fine_tun_best_Opt')
            save_dl_loss_obj.save_model(best_model)
            save_dl_loss_obj.save_loss(ld_)
        '''
        
        model = best_model  # to save at the end-.
        
        '''
        for name, param in model.named_parameters():
            print(f'{name}: {param.size()}')
            print(param.data)
        input(11111)
        '''
        
    else:  # we didn't Hyperparameter Optimization-.
    
        # 3- train and validation loop-.
        # DL MODELS - training and validation  == INI ==-.
        input_size, output_size = np.shape(X_train)[1],\
            np.shape(y_train)[1]  # len(feat_var), len(tar_var)

        # TEMPORARY is here (I can't pass in yaml config file, I don't know how do that)
        # layers=[(2048, nn.ReLU(),None), # until now it doesn't work from yaml file-.
        #        (output_size,None,None)]
        layers = [(1024, nn.ReLU(), None),  # for now it isn't work from yaml file-.
                  (512, nn.ReLU(), None),  # for now it isn't work from yaml file-.
                  (256 , nn.ReLU(),None),  # for now it isn't work from yaml file-.
                  (128 , nn.ReLU(),None),  # for now it isn't work from yaml file-.
                  (output_size, None, None)]
        '''
        layers = [(30, nn.ReLU(), None),  # for now it isn't work from yaml file-.
                  (10, nn.ReLU(), None),  # for now it isn't work from yaml file-.
                  (10, nn.ReLU(),None),   # for now it isn't work from yaml file-.
                  (5 , nn.ReLU(),None),   # for now it isn't work from yaml file-.
                  (1 , nn.ReLU(),None),   # for now it isn't work from yaml file-.
                  (output_size, None, None)]
        '''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 3-1- DLmodel - FFNN-.
        if str.split(cfg_obj.DL_name, '-')[0] == 'MLP':
            print('Training using {} DL model'.format(cfg_obj.DL_name))
            model = MLP(input_size, output_size, layers, device, cfg_obj.lr,
                        eval(cfg_obj.optimizer), eval(cfg_obj.loss),
                        cfg_obj.weight_decay, cfg_obj.momentum)
            # print(model); input(66); exit(11)
            # save DL ml's log file-.
            model.save_params_model(cfg_obj.dir_logs, cfg_obj.batch_size,
                                    cfg_obj.batch_size,  # bath_size_{train,test}
                                    cfg_obj.epochs, cfg_obj.optimizer,
                                    cfg_obj.loss,  # I need str and not torch..-.
                                    cfg_obj.ds_file, cfg_obj.dir_results)
            train_predict_obj = TrainPredict_FFNN(model, train_loader_1.dataloader,
                                                  val_loader_1.dataloader,
                                                  input_size, output_size, layers,
                                                  device, cfg_obj.lr,
                                                  cfg_obj.optimizer,
                                                  cfg_obj.weight_decay,
                                                  cfg_obj.momentum)
            train_time, ld_ = train_predict_obj.train(cfg_obj.epochs,
                                                      val=True)  # train_time and loss_dict-.

            # 3-2- DLmodel - BNN-.
        elif str.split(cfg_obj.DL_name, '-')[0] == 'BNNbnn':
            print('Training using {} DL model'.format(cfg_obj.DL_name))
            model_bnn = BayesianNet(input_size, output_size, eval(cfg_obj.loss),
                                    eval(cfg_obj.optimizer),
                                    eval(cfg_obj.kl_l), device, cfg_obj.weight_decay,
                                    cfg_obj.lr, cfg_obj.kl_w)
            # save DL ml's log file-.
            model_bnn.save_params_model(cfg_obj.dir_logs, cfg_obj.batch_size,
                                        cfg_obj.batch_size,  # bath_size_{train,test}-.
                                        cfg_obj.epochs, cfg_obj.optimizer,
                                        cfg_obj.loss,  # I need str and not torch-.
                                        cfg_obj.ds_file, cfg_obj.dir_results,
                                        cfg_obj.kl_l)
            # print(model_bnn); input(77); exit(11)
            train_predict_obj = TrainPredict_BNN(model_bnn,
                                                 train_loader_1.dataloader,
                                                 val_loader_1.dataloader,
                                                 input_size, output_size,
                                                 eval(cfg_obj.loss),
                                                 eval(cfg_obj.optimizer),
                                                 eval(cfg_obj.kl_l), device,
                                                 cfg_obj.weight_decay, cfg_obj.lr,
                                                 cfg_obj.kl_w)
            train_time, ld_ = train_predict_obj.train(cfg_obj.epochs,
                                                      val=True)  # train_time and loss_dict-.
            # print(loss_dict['train_loss']), input(88)
            model = model_bnn; del(model_bnn)
        else:
            print("{0}The DL model's named {1} doesn't exist.{0}".
                  format('\n', cfg_obj.DL_name)); exit(11)
        # DL MODELS - training and validation  == END ==-.

    # print size of allocated tensors in GPU-.
    if device == 'cuda':
        print('Memory allocated in Cuda: {0} Kbytes.'.
              format(torch.cuda.memory_allocated(device=device)/1024))
        print('Max. memory allocated in Cuda: {0} Kbytes. {1}'.
              format(torch.cuda.max_memory_allocated(device=device)/1024,
                     '\n'*2))
        
    if cfg_obj.plot_loss:
        # plot losses (training and validation)-.
        plotlosses_obj = PlotLosses('_fine_tun_best_Opt')
        # se asigna a la variable plt_pl toma dos parametros x e y ('',ld_), y luego llama al metodo plot_loss del objeto
        # plotlosses_obj, pasando x, y, cfg_obj.dir_save.
        plt_pl = lambda x, y: plotlosses_obj.plot_loss(x, y, cfg_obj.dir_save)
        plt_pl('', ld_)
        # plt_pl('train', ld_['train_loss'])
        # plt_pl('val', ld_['val_loss'])
        # plotlosses_obj.plot_loss('val', ld_['val_loss'])
        
    # save train and validatiton losses and DL's model-.
    if cfg_obj.model_loss_save:
        save_dl_loss_obj = SaveDLModelLoss(cfg_obj.dir_save, '_fine_tun_best_Opt')
        save_dl_loss_obj.save_model(model)
        save_dl_loss_obj.save_loss(ld_)
        
    # control TOTAL EXECUTION TIME-.
    end_tot_ex_time = time.time()
    print('Total execution time {0}{1}'.
          format('\n', convert_to_preferred_format(end_tot_ex_time-start_tot_ex_time)))

    # calculate and plot prediction-.
    with torch.no_grad():
        model.eval()
        X_val, y_val = next(iter(val_loader_1.dataloader))
        # X_val, y_val = next(iter(val_loader))
        y_pred = model(X_val)
        # print(y_val.size(), y_pred.size(), sep='\n')
        # print(X_val[:,0], X_val[:,1], sep='\n')
        # temp= feat_scal.inverse_transform(X_val)[:, 0]
        # vel_cal= feat_scal.inverse_transform(X_val)[:, 1]
        pred_obj = Predict(X_val, y_val, y_pred, feat_scal, targ_scal,
                           cfg_obj.feat_names, cfg_obj.targ_names,
                           cfg_obj.scale_targ)
    # pred_obj.plot_corr_true_pred()
    pred_obj.plot_corr_true_pred_mod(cfg_obj.dir_save)
    # pred_obj.plot_pred_error_display()
    pred_obj.plot_residuals(cfg_obj.dir_save)
    # plot predicted and true values vs. -.
    pred_obj.y_vs_x(cfg_obj.dir_save)
    pred_obj.y_vs_x_3d(cfg_obj.dir_save)
    
    # pred_obj.calc_and_plot_Dtg(cfg_obj.dir_save)  # ok, but I need to order the array-.
    # num_eles = 1500
    # pred_obj.plot_surf_with_proj(cfg_obj.dir_save, num_eles)
        
    return 0
# ======================================================================= END79


# ======================================================================= INI79
if __name__ == '__main__':
    '''' to execute main methof -if this exist ''' 
    
    # my auxiliaries methods-.
    # from utils.gen_tools import *
    # print(dir())
    config_file = Path(__file__).parent/'config_file_all_withoutOH.yaml'
    config = ga(config_file)
    
    # list the name and versions of the modules used-.
    # a = mv(); a.open_save_modules()

    # call main-.
    val = main(config)
else:
    print('{0} imported as Module'.format(__file__.split('/')[-1]))
# ======================================================================= END79
