#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from exploratory import Data_Analysis

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error as mse, \
    PredictionErrorDisplay, make_scorer, mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

# set global font settings-.
plt.rcParams.update({
    'font.size': 10,             # set font size-.
    'font.family': 'serif',      # set font family-.
    'font.weight': 'normal',     # set font weight-.
    'axes.labelsize': 2,         # set xlabel and ylabel font size-.
    'axes.titlesize': 2,         # set xlabel and ylabel font size-.
    'axes.labelweight': 'bold',  # set xlabel and ylabel font weight-.
})

# set global font settings using rc parameters-.
sns.set(rc={
    'font.size': 8,           # set font size-.
    'font.family': 'serif',   # set font family-.
    'font.weight': 'normal',  # set font weight-.
    'axes.labelsize': 8
})


class Feature_Selection(Data_Analysis):
    def __init__(self, df, features, targets):
        super().__init__(df, features, targets)

    def get_feature_selection(self, class_name: str) -> None:
        '''
        get feature selection-.
        '''

        # exploratory plots-.
        print('From {0} class get feature selection.'.
              format(class_name)
              )

        # split df in train/test-.
        X_train, X_val, y_train, y_val = train_test_split(
            self.df[self.features],
            self.df[self.targets],
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        # apply scaler and transform-.
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)

        # 1- linear models-.
        lr = LinearRegression()
        # lr = Ridge()  # L1 regularization-.
        # lr = Lasso()  # L2 regularization-.
        # lr = ElasticNet()  # L1 + L2 regularization-.

        # ensemble models-.
        lr = RandomForestRegressor()
        
        lr.fit(X_train_sc, y_train)

        y_pred = lr.predict(X_val_sc)
        y_pred_cv = cross_val_predict(lr, X_val_sc, y_val, cv=10)
        print(mse(y_pred, y_val))
        print(np.sqrt(mse(y_pred, y_val)))
        
        # print(np.shape(y_pred), np.shape(y_val))

        '''
        for i_target, target_var in enumerate(self.targets):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)

            ax.scatter(y_pred[:, i_target], y_val.loc[:, target_var],
                       color='b',
                       marker='o',
                       label=f'r={np.corrcoef(y_pred[:, i_target],y_val.loc[:,target_var])[0][1]}')
            ax.set_xlabel(r'$y^{pred}$')
            ax.set_ylabel(r'$y^{pred}$')
            ax.legend()
            ax.set_title(r'y^{true}$ vs. $y^{pred}')

            plt.tight_layout()
            plt.show()
        
        input(88)
        
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y_val,
            y_pred=y_pred_cv,
            kind="actual_vs_predicted",
            subsample=100,
            ax=axs[0],
            random_state=0,
        )
        axs[0].set_title("Actual vs. Predicted values")
        PredictionErrorDisplay.from_predictions(
            y_val,
            y_pred=y_pred_cv,
            kind="residual_vs_predicted",
            subsample=100,
            ax=axs[1],
            random_state=0,
        )
        axs[1].set_title("Residuals vs. Predicted Values")
        fig.suptitle("Plotting cross-validated predictions")
        plt.tight_layout()
        plt.show()

        input(99)
        '''

        # Recursive Feature Selection-.
        rfecv = RFECV(
            estimator=lr,
            step=1,
            cv=10,
            scoring=make_scorer(mae, greater_is_better=False),
            min_features_to_select=1
        )
        
        rfecv.fit(X_train_sc, y_train)
        print(rfecv.fit(X_train_sc, y_train))
        print('Optimal number of features : {0:d}'.format(rfecv.n_features_))

        fig = plt.figure(figsize=(12, 6))
        # ax = fig.add_subplot(111)
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
                 # need to negate sign here due to scorer behavior
                 -rfecv.cv_results_['mean_test_score'], marker='o',
                 color='r', linestyle='--', linewidth=2, markersize=8)
        plt.xlabel('Number of Features')
        plt.ylabel('Cross-Validation Score (MAPE)')
        # plt.savefig('rfe_me.png', dpi=300)
        plt.tight_layout()
        plt.show()
        
        input(99)

        # print for each number of features
        print('{0}{1}{0} score for each number of feature {2}{0}{1}'.
              format('\n', '*'*10,
                     -rfecv.cv_results_['mean_test_score']))
        print('{0}{1}{0} Rank of feature importance {2}{0}{1}'.
              format('\n', '*'*10,
                     rfecv.ranking_))
        print('{0}{1}{0} Feature importance for estimator {2}{0}{1}'.
              format('\n', '*'*10,
                     rfecv.estimator_.feature_importances_))
        
        # Only for RandomForestRegressor, features importance in function of GINI-.
        # source: https://www.geeksforgeeks.org/feature-importance-with-random-forests/
        # Built-in feature importance (Gini Importance)
        importances = lr.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': self.features,
            'Gini Importance': importances}).sort_values('Gini Importance',
                                                         ascending=False)
        print(feature_imp_df)

        # Create a bar plot for feature importance
        plt.figure(figsize=(8, 4))
        plt.barh(self.features,
                 importances,
                 color='skyblue')
        plt.xlabel('Gini Importance')
        plt.title('Feature Importance - Gini Importance')
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization
        plt.show()
        
        input(100)
        
        return None
