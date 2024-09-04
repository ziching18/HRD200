## import libraries
import numpy as np
import math
from numpy import interp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy.stats import spearmanr, pearsonr
from os import path
import pickle
import seaborn as sns
from copy import deepcopy

import sklearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, confusion_matrix, auc, mean_squared_error, precision_score, jaccard_score, fowlkes_mallows_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

sns.set_palette(sns.color_palette("Spectral"))

def plotStyle():
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['font.size'] = 15
    rcParams['axes.linewidth'] = 2
    rcParams['grid.linewidth'] = 2
    rcParams['grid.color'] = 'gainsboro'
    rcParams['font.weight'] = 'normal'
    rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.labelsize'] = 15
    rcParams['legend.edgecolor'] = 'none'
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.top"] = False

def split_data(X_in, y_in, sets):
    X = X_in.copy()
    y = y_in["label"]
    setsplit = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=sets)
    tr, ts = next(setsplit.split(X, y))
    X_train = X.iloc[tr,:]
    y_train = y.iloc[tr]
    X_test = X.iloc[ts,:]
    y_test = y.iloc[ts]
    train_ids = pd.Series(list(X_train.index), name="id")
    test_ids = pd.Series(list(X_test.index), name="id")
    return X_train, X_test, y_train, y_test, train_ids, test_ids

def defineSplits(X, ycateg, random_state):
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(random_state))
    splits = []
    for (tr,ts) in cv.split(X, ycateg):
        splits.append((tr,ts))
    return splits

class DropCollinear(BaseEstimator, TransformerMixin):
    def __init__(self, thresh):
        self.uncorr_columns = None
        self.thresh = thresh

    def fit(self, X, y):
        cols_to_drop = []

        # Find variables to remove
        X_corr = X.corr()
        large_corrs = X_corr>self.thresh
        indices = np.argwhere(large_corrs.values)
        indices_nodiag = np.array([[m,n] for [m,n] in indices if m!=n])

        if indices_nodiag.size>0:
            indices_nodiag_lowfirst = np.sort(indices_nodiag, axis=1)
            correlated_pairs = np.unique(indices_nodiag_lowfirst, axis=0)
            resp_corrs = np.array([[np.abs(spearmanr(X.iloc[:,m], y).correlation), np.abs(spearmanr(X.iloc[:,n], y).correlation)] for [m,n] in correlated_pairs])
            element_to_drop = np.argmin(resp_corrs, axis=1)
            list_to_drop = np.unique(correlated_pairs[range(element_to_drop.shape[0]),element_to_drop])
            cols_to_drop = X.columns.values[list_to_drop]

        cols_to_keep = [c for c in X.columns.values if c not in cols_to_drop]

        self.uncorr_columns = cols_to_keep

        return self

    def transform(self, X):
        return X[self.uncorr_columns]

    def get_params(self, deep=False):
        return {'thresh': self.thresh}

class SelectAtMostKBest(SelectKBest):
    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            # set k to "all" (skip feature selection), if less than k features are available
            self.k = "all"

# class function to get average prediction from 3 models
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x.best_estimator_) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    # predict for cloned models and average them
    def predict_proba(self, X):
        #self.models_ = self.models
        if not hasattr(self, "models_"):
            self.models_ = self.models

        predictions_0 = np.column_stack([
            model.predict_proba(X)[:,0] for model in self.models_
        ])

        predictions_1 = np.column_stack([
            model.predict_proba(X)[:,1] for model in self.models_
        ])
        means_0 = np.mean(predictions_0, axis=1) # label = 0
        means_1 = np.mean(predictions_1, axis=1) # label = 1
        return np.column_stack([means_0, means_1])

    # make prediction
    def predict(self, X):
        #self.models_ = self.models
        if not hasattr(self, "models_"):
            self.models_ = self.models

        predictions_0 = np.column_stack([
            model.predict_proba(X)[:,0] for model in self.models_
        ])

        predictions_1 = np.column_stack([
            model.predict_proba(X)[:,1] for model in self.models_
        ])
        means_0 = np.mean(predictions_0, axis=1) # label = 0
        means_1 = np.mean(predictions_1, axis=1) # label = 1
        proba = np.column_stack([means_0, means_1])
        preds = []
        for i in proba:
            if i[0] > i[1]:
                preds.append(0)
            else:
                preds.append(1)
        return np.asarray(preds)

# support vector machine optimisation
def optimise_SVC_featsel(X, y, cv=5):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(0.8)
    svc = SVC(random_state=0, max_iter=-1, probability=True)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('svc', svc)])

    param_grid = { 'kbest__k': np.arange(2,X.shape[1],1),
                    'svc__kernel': ['rbf','sigmoid','linear'],
                    'svc__gamma': np.logspace(-9,-2,60),
                    'svc__C': np.logspace(-3,3,60)}

    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, scoring='roc_auc', return_train_score=True, cv=cv,
                                n_jobs=6, verbose=1, n_iter=1000, random_state=0)
    search.fit(X,y)

    return search

# random forest optimisation
def optimise_rf_featsel(X, y, cv=5):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(0.8)
    rf = RandomForestClassifier(random_state=0)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('rf', rf)])
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                    "rf__max_depth": [3, None],
                    "rf__n_estimators": [5, 10, 25, 50, 100],
                    "rf__max_features": [0.05, 0.1, 0.2, 0.5, 0.7],
                    "rf__min_samples_split": [2, 3, 6, 10, 12, 15]
                    }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, scoring='roc_auc', return_train_score=True, cv=cv,
                                n_jobs=6, verbose=1, n_iter=1000, random_state=1)
    search.fit(X,y)

    return search

# search for best set of parameters
def run_all_models(X, y, splits):
    svc_result = optimise_SVC_featsel(X, y, cv=splits)
    rf_result = optimise_rf_featsel(X, y, cv=splits)
    averaged_models = AveragingModels(models=(svc_result, rf_result))
    results = {}
    results['svc'] = svc_result
    results['rf'] = rf_result
    results['avg'] = averaged_models
    return results

# train and validate model, plot ROC curves (individual model)
def plot_and_refit(X, y, model, splits, s, rs, name, label="", ids=None):
    aucs = []
    ypreds = []     ### prediction
    yreals = []     ### actual label
    ypreds_cv = []
    yreals_cv = []
    pred = []
    truf = []
    pred_id = []
    aucs = []
    ypreds = []     # prediction
    yreals = []     # actual label
    mses = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 10)

    fout = open('output/s{}/rs{}/rs{}_{}_patpreds_cv_.txt'.format(s, rs, rs, name), "w")
    cv_models = []
    for i,(tr,ts) in enumerate(splits): # tr = train, ts = validation
        model.fit(X.iloc[tr,:], y.iloc[tr])     # train model
        #try: print(model._final_estimator)
        #except: pass
        cv_models.append(deepcopy(model))       # append to some list
        y_pred = model.predict_proba(X.iloc[ts,:])[:,1]     # make prediction using trained model on validation set
        ytest = y.iloc[ts]
        ytest_ids = ids.iloc[ts]

        # Precision
        ypreds.extend(y_pred)
        yreals.extend(ytest)
        pred.append(list(y_pred))
        truf.append(list(ytest))
        pred_id.append(list(ytest_ids))
        ypreds_cv.append(y_pred)
        yreals_cv.append(ytest)
        roc_auc = roc_auc_score(ytest, y_pred)
        aucs.append(roc_auc)

        # AUC
        fpr, tpr, thresholds = roc_curve(ytest, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
   
        # Write-up the predictions
        for eachtestpat,eachpred in enumerate(y_pred):
           fout.write('{}, {}, {}\n'.format(ytest_ids.values[eachtestpat],name,eachpred))
    fout.close()

    # Mean curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    median_auc = np.median(aucs)
    std_auc = np.std(aucs)

    f = open("output/s{}/rs{}/rs{}_{}_train_output.txt".format(s, rs, rs, name), "w")
    f.write("cv, {}, {}, {}, {}\n".format(name, mean_auc, median_auc, std_auc))
    f.close()

    # Refit
    model.fit(X,y) ## fit to entire training cohort
    return [model, cv_models], pred, truf, pred_id #ypreds, yreals

# use model with best results for training and validation, plot overall ROC curve
def refit_all_models(X, y, results, splits, s, rs, labels, ids=None):
    refit = {}
    preds = {}
    trufs = {}
    pred_ids = {}
    tprs = []
    mean_fpr = np.linspace(0, 1, 10)
    
    for model in results.keys():
        try: # get best parameters
            refit[model], preds[model], trufs[model], pred_ids[model] = plot_and_refit(X, y, results[model].best_estimator_, splits, s, rs, model, label=labels[model], ids=ids)
        except: # get parameters
            refit[model], preds[model], trufs[model], pred_ids[model] = plot_and_refit(X, y, results[model], splits, s, rs, model, label=labels[model], ids=ids)

    return refit, preds, trufs, pred_ids

# test model with testing set (individual models)
def final_test(X, y, model, s, rs, name):
    y_pred = model.predict_proba(X)[:,1]
    yreals = y.values

    # Precision
    roc_auc = roc_auc_score(y, y_pred)
    #print("Testing AUC for {}: {}".format(name, roc_auc))
    #print(roc_auc)

    # AUC
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    df_roc = pd.DataFrame().from_dict({'fpr':fpr, 'tpr':tpr})
    #df_roc.to_csv('all_outputs/{}/{},{},{}/rs{}/rs{}_{}_test_roc.csv'.format(score, no_iter, med, thresh, rs, rs, label))
    df_roc.to_csv("output/s{}/rs{}/rs{}_test_roc.csv".format(s, rs, rs))

    # Precision-recall curve
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from inspect import signature
    precision, recall, thresholds = precision_recall_curve(yreals, y_pred)
    rand_perf = (yreals==1).sum()/yreals.shape[0]
    average_precision = average_precision_score(yreals, y_pred)

    f = open("output/s{}/rs{}/rs{}_{}_test_output.txt".format(s, rs, rs, name), "w")
    f.write('test,{},{}\n'.format(name,roc_auc))
    f.close()

    f = open("output/s{}/rs{}/rs{}_predictions.txt".format(s, rs, rs), "w")
    f.write('{} '.format(name))
    for eachy in y_pred:
        f.write('{} '.format(eachy))
    f.write('\n')
    f.close()
    
    return y_pred

# test models and plot overall ROC curves
def test_all_models(X, y, results, s, rs, labels):
    res = []
    test_result = {}
    for model in results.keys():
        test = final_test(X, y, results[model][0], s, rs, model)
        test_result[model] = test
        res.append(test)

    return test_result, res

def validate(X, results, rs, labels):
    val_probas = {}
    val_preds = {}
    for model in results.keys():
        #print(model)
        #test = final_test(X, results[model], rs, model, label=labels[model])
        y_probas = results[model].predict_proba(X)[:,1]
        val_probas[model] = y_probas
        y_preds = results[model].predict(X)
        val_preds[model] = y_preds
    return val_probas, val_preds

def metrics_tests(df):
    test = df["truth"]
    preds = df["pred"]
    probs = df["avg"]
    from sklearn import metrics

    metrix = ["TPR","TNR","PPV","NPV","FPR","FNR","FRD","F1","ACC","PRE","REC","AUC"]
    df = pd.DataFrame(index=metrix)

    cm = metrics.confusion_matrix(test, preds)
    print(cm)
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

    mets = []
    # Sensitivity, hit rate, recall, or true positive rate
    mets.append(TP/(TP+FN))
    # Specificity or true negative rate
    mets.append(TN/(TN+FP))
    # Precision or positive predictive value
    mets.append(TP/(TP+FP))
    # Negative predictive value
    mets.append(TN/(TN+FN))
    # Fall out or false positive rate
    mets.append(FP/(FP+TN))
    # False negative rate
    mets.append(FN/(TP+FN))
    # False discovery rate
    mets.append(FP/(TP+FP))
    # F1 score
    mets.append((2*TP)/((2*TP)+FP+FN))

    mets.append(metrics.accuracy_score(test, preds))
    mets.append(metrics.precision_score(test, preds))
    mets.append(metrics.recall_score(test, preds))
    mets.append(metrics.roc_auc_score(test, probs))
    
    df["avg"] = mets

    return df, cm