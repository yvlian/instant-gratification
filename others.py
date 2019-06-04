import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import os

start = time.time()

os.listdir('./data')
# Loading Train/Test Data
train, test = pd.read_csv('./data/train.csv'),pd.read_csv('./data/test.csv')
oof=np.zeros(len(train))
preds=np.zeros(len(test))

auc_all=np.array([])

clf_name='QDA'
make_submission=1   # generate a submission file

NFOLDS = 5  # 25
RS = 42

rp_values = np.linspace(0.01, 0.9, num=100)

rp_best = {'rp': np.array([]),
           'auc': np.array([])
           }

print(f'Parameter tuning for the {clf_name} classifier:\n')

magic_max = train['wheezy-copper-turtle-magic'].max()
magic_min = train['wheezy-copper-turtle-magic'].min()

# BUILD 512 SEPARATE NON-LINEAR MODELS
for i in range(magic_min, magic_max + 1):
    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS i
    X = train[train['wheezy-copper-turtle-magic'] == i].copy()
    Y = X.pop('target').values
    X_test = test[test['wheezy-copper-turtle-magic'] == i].copy()
    idx_train = X.index
    idx_test = X_test.index
    X.reset_index(drop=True, inplace=True)

    cols = [c for c in X.columns if c not in ['id', 'wheezy-copper-turtle-magic']]
    X = X[cols].values  # numpy.ndarray
    X_test = X_test[cols].values  # numpy.ndarray

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    vt = VarianceThreshold(threshold=1.5).fit(X)
    X = vt.transform(X)  # numpy.ndarray
    X_test = vt.transform(X_test)  # numpy.ndarray

    # STRATIFIED K FOLD
    folds = StratifiedKFold(n_splits=NFOLDS, random_state=RS)

    auc_rp = np.array([])
    for rp in rp_values:

        auc_folds = np.array([])
        for fold_num, (train_index, val_index) in enumerate(folds.split(X, Y), 1):
            X_train, Y_train = X[train_index, :], Y[train_index]
            X_val, Y_val = X[val_index, :], Y[val_index]

            params = {'reg_param': rp}

            # BUILDING THE PIPELINE FOR THE CLASSIFIER
            pipe = Pipeline([('scaler', StandardScaler()),
                             (clf_name, QuadraticDiscriminantAnalysis(**params))
                             ])

            pipe.fit(X_train, Y_train)

            oof[idx_train[val_index]] = pipe.predict_proba(X_val)[:, 1]
            preds[idx_test] += pipe.predict_proba(X_test)[:, 1] / NFOLDS

            auc = roc_auc_score(Y_val, oof[idx_train[val_index]])
            auc_folds = np.append(auc_folds, auc)

        auc_folds_av = np.mean(auc_folds)
        auc_rp = np.append(auc_rp, auc_folds_av)

    rp_best_value = rp_values[np.argmax(auc_rp)]
    auc_best_value = np.max(auc_rp)

    rp_best['rp'] = np.append(rp_best['rp'], rp_best_value)
    rp_best['auc'] = np.append(rp_best['auc'], auc_best_value)

# UNCOMMENT IF YOU WANT TO SEE RESULTS FOR ALL 512 MODLES
# (WARNING: IT IS A VERY LONG LIST!!!)
#     print('Model: i=', i)
#     print('rp = ', rp_best_value)
#     print('AUC = ', auc_best_value, '\n')

print(f'Cross-validation for the {clf_name} classifier:')

magic_max = train['wheezy-copper-turtle-magic'].max()
magic_min = train['wheezy-copper-turtle-magic'].min()

auc_all = np.array([])

# BUILD 512 SEPARATE NON-LINEAR MODELS
for i in range(magic_min, magic_max + 1):
    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS i
    X = train[train['wheezy-copper-turtle-magic'] == i].copy()
    Y = X.pop('target').values
    X_test = test[test['wheezy-copper-turtle-magic'] == i].copy()
    idx_train = X.index
    idx_test = X_test.index
    X.reset_index(drop=True, inplace=True)

    cols = [c for c in X.columns if c not in ['id', 'wheezy-copper-turtle-magic']]
    X = X[cols].values  # numpy.ndarray
    X_test = X_test[cols].values  # numpy.ndarray

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    vt = VarianceThreshold(threshold=1.5).fit(X)
    X = vt.transform(X)  # numpy.ndarray
    X_test = vt.transform(X_test)  # numpy.ndarray

    # STRATIFIED K FOLD
    auc_folds = np.array([])
    folds = StratifiedKFold(n_splits=NFOLDS, random_state=RS)

    for fold_num, (train_index, val_index) in enumerate(folds.split(X, Y), 1):
        X_train, Y_train = X[train_index, :], Y[train_index]
        X_val, Y_val = X[val_index, :], Y[val_index]

        params = {'reg_param': rp_best['rp'][i - magic_min]}

        # BUILDING THE PIPELINE FOR THE CLASSIFIER
        pipe = Pipeline([('scaler', StandardScaler()),
                         (clf_name, QuadraticDiscriminantAnalysis(**params))
                         ])

        pipe.fit(X_train, Y_train)

        oof[idx_train[val_index]] = pipe.predict_proba(X_val)[:, 1]
        preds[idx_test] += pipe.predict_proba(X_test)[:, 1] / NFOLDS

        auc = roc_auc_score(Y_val, oof[idx_train[val_index]])
        auc_folds = np.append(auc_folds, auc)

    auc_all = np.append(auc_all, np.mean(auc_folds))

# PRINT CROSS-VALIDATION AUC FOR THE CLASSFIER
auc_combo = roc_auc_score(train['target'].values, oof)
auc_folds_average = np.mean(auc_all)
std = np.std(auc_all) / (np.sqrt(NFOLDS) * np.sqrt(magic_max + 1))

print(f'The combined AUC CV score is {round(auc_combo, 5)}.')
print(f'The folds average AUC CV score is {round(auc_folds_average, 5)}.')
print(f'The standard deviation is {round(std, 5)}.')


end = time.time()
if make_submission:
    sub = pd.read_csv('./data/sample_submission.csv')
    sub['target'] = preds
    sub.to_csv('submission.csv',index=False)