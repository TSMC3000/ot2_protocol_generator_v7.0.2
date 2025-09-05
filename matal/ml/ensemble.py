import gc
import platform
from typing import Union, Tuple
 
from sklearn.ensemble import VotingRegressor as _BaseVotingRegressor
from sklearn.ensemble import VotingClassifier as _BaseVotingClassifier

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin, is_regressor, is_classifier, ClassifierMixin
 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from joblib import Parallel, delayed

import warnings
warnings.simplefilter("ignore", UserWarning)
 

class VotingRegressor(_BaseVotingRegressor):
    ''' VotingRegressor supports loading pre-trained models. '''
 
    def assume_fitted(self):
        names, clfs = self._validate_estimators()
        self.estimators_ = [c for c in clfs]
 
    def add_models(self, estimators: list):
        assert isinstance(estimators, list)
        self.estimators += estimators
        self.assume_fitted()
    def predict_raw(self, X):
        check_is_fitted(self)
 
        pred = self._predict(X)
        return pred
 
    def predict(self, X, with_std=False, scale_func=None):
        pred = self.predict_raw(X)
        if scale_func:
            pred = scale_func(pred)
        mean = np.average(pred, axis=-1, weights=self._weights_not_none)
        variance = np.var(pred, axis=-1)
 
        if len(pred.shape) > 0:
            mean = mean.T
            variance = variance.T
 
        if with_std:
            return mean, np.sqrt(variance)
        else:
            return mean




class VotingClassifier(_BaseVotingClassifier):
    ''' VotingClassifier supports loading pre-trained models. '''
 
    def assume_fitted(self):
        names, clfs = self._validate_estimators()
        self.estimators_ = [c for c in clfs]
 
    def add_models(self, estimators: list):
        assert isinstance(estimators, list)
        self.estimators += estimators
        self.assume_fitted()
    
    def predict_raw(self, X):
        check_is_fitted(self)
 
        pred = self._predict(X)
        return pred

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray(Parallel(n_jobs=self.n_jobs, backend='loky')(delayed(est.predict)(X) for est in self.estimators_)).T
 
    def predict(self, X, with_std=False, scale_func=None):
        pred = self.predict_raw(X)
        if scale_func:
            pred = scale_func(pred)
        mean = np.average(pred, axis=-1, weights=self._weights_not_none)
        variance = np.var(pred, axis=-1)
 
        if len(pred.shape) > 0:
            mean = mean.T
            variance = variance.T
 
        if with_std:
            return mean, np.sqrt(variance)
        else:
            return mean