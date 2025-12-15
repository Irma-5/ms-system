"""
Model classes for Survival Analysis
These classes are needed to unpickle saved models
"""
import numpy as np
import copy
from typing import List
from sklearn.ensemble import RandomForestClassifier
from lifelines import CoxPHFitter
from lifelines import AalenJohansenFitter


def get_y_arr(y):
    """Convert DataFrame y to structured array"""
    y_arr = np.empty(dtype=[('event', bool), ('duration', np.float64)], shape=len(y))
    y_arr['event'] = y['event'].astype(bool)
    y_arr['duration'] = y['duration']
    return y_arr


def step_to_array(step_functions):
    shape_ = (step_functions.shape[0], step_functions[0].x.shape[0])
    arr = np.empty(shape=shape_)
    for i in range(len(step_functions)):
        arr[i] = step_functions[i].y
    return arr, step_functions[0].x


def transform_timegrid(curves, time, grid):
    if time.max() < grid.max():
        time = np.hstack([time, np.array([grid.max() + 1])])
        if len(curves.shape) == 1:
            curves = np.hstack([curves, np.array([0])])
        elif len(curves.shape) == 2:
            curves = np.hstack([curves, np.zeros(shape=(curves.shape[0], 1))])
    ind = np.searchsorted(time, grid)
    if len(curves.shape) == 1:
        return curves[ind]
    elif len(curves.shape) == 2:
        return curves[:, ind]
    else:
        return None


# Default time grid
TIME_GRID = np.linspace(1, 309, 100)


class OvR:
    def __init__(self, estimator, mode, early_threshold=1.0):
        self.estimator = estimator
        self.mode = mode       # early, all, single
        self.models = None
        self.events = None
        self.TIME_GRID = None
        self.thrsh = early_threshold
        self.model = []

    def step_to_array(self, step_functions):
        shape_ = (step_functions.shape[0], step_functions[0].x.shape[0])
        arr = np.empty(shape=shape_)
        for i in range(len(step_functions)):
            arr[i] = step_functions[i].y
        return arr, step_functions[0].x

    def transform_timegrid(self, curves, time, grid):
        if time.max() < grid.max():
            time = np.hstack([time, np.array([grid.max() + 1])])
            if len(curves.shape) == 1:
                curves = np.hstack([curves, np.array([0])])
            elif len(curves.shape) == 2:
                curves = np.hstack([curves, np.zeros(shape=(curves.shape[0], 1))])
        ind = np.searchsorted(time, grid)
        if len(curves.shape) == 1:
            return curves[ind]
        elif len(curves.shape) == 2:
            return curves[:, ind]
        else:
            return None

    def transform_xy(self, X, y, events: List = [], event_of_interest=None):
        if self.mode == 'all':
            X, y = self.case1(X, y, events)
        elif self.mode == 'single':
            X, y = self.case2(X, y, events)
        elif self.mode == 'early':
            X, y = self.case3(X, y, events, early_threshold=self.thrsh)
        elif self.mode == 'mix':
            if event_of_interest in {5, 6}:
                X, y = self.case3(X, y, events, early_threshold=self.thrsh)
            else:
                X, y = self.case2(X, y, events)
        else:
            raise ValueError('Wrong mode')
        return X, y

    def case1(self, X_, y_, events: List = []):
        y, X = y_.copy(), X_.copy()
        mask = y_.event.isin(events)
        y.event = mask.astype('int')
        return X, y

    def case2(self, X_, y_, events: List = []):
        y, X = y_.copy(), X_.copy()
        mask = y_.event.isin(events)
        y.event = mask
        y = y[mask]
        X = X[mask]
        y.event = y.event.astype('int')
        return X, y

    def case3(self, X_, y_, events: List = [], early_threshold=1.0):
        y, X = y_.copy(), X_.copy()
        mask = y_.event.isin(events)
        maxm = y_.duration[mask].quantile(early_threshold)
        y.event = mask
        X = X[y.duration <= maxm]
        y = y[y.duration <= maxm]
        y.event = y.event.astype('int')
        return X, y

    def fit(self, X_, y_):
        self.TIME_GRID = np.linspace(y_['duration'].min(), y_['duration'].max(), 100)
        self.model = []
        self.events = y_.event.unique()
        for i in sorted(self.events):
            if i:
                tmp = copy.deepcopy(self.estimator)
                X, y = self.transform_xy(X_, y_, [i], i)
                if type(self.estimator) == CoxPHFitter:
                    X = X.join(y)
                    tmp.fit(X, duration_col='duration', event_col='event')
                elif isinstance(self.estimator, AalenJohansenFitter):
                    T, E = y.duration, y.event
                    tmp.fit(T, E, event_of_interest=1)
                else:
                    y = get_y_arr(y)
                    tmp.fit(X, y)
                self.model.append(tmp)

    def predict(self, X_):
        """
        Returns predictions with shape (n_events, n_samples, n_timepoints)
        e.g. (6, 4800, 100)
        """
        predictions = np.empty(shape=(len(self.events) - 1, X_.shape[0], 100))
        for i in sorted(self.events):
            if i:
                if type(self.estimator) == CoxPHFitter:
                    sf = (self.model[i - 1].predict_survival_function(X_, times=self.TIME_GRID)).T
                elif isinstance(self.estimator, AalenJohansenFitter):
                    sf = 1 - self.model[i - 1].cumulative_density_[f'CIF_1']
                    sf = self.transform_timegrid(sf.values, self.model[i - 1].cumulative_density_.index, self.TIME_GRID).T.astype(float)
                    sf = np.repeat(sf[np.newaxis, :], X_.shape[0], axis=0)
                else:
                    sf = self.model[i - 1].predict_survival_function(X_)
                    sf, times = self.step_to_array(sf)
                    sf = self.transform_timegrid(sf, times, self.TIME_GRID)
                predictions[i - 1] = sf
        return predictions


class MetaModel:
    def __init__(self, estimator, mode='weighted'):
        self.ovr = estimator
        self.meta_model = RandomForestClassifier(n_jobs=-1, random_state=42)
        self.mode = mode

    def fit(self, X_, y_, X_val=None, y_val=None):
        if X_val is not None:
            self.ovr.fit(X_, y_, X_val, y_val)
        else:
            self.ovr.fit(X_, y_)
        mask = y_.event > 0
        self.meta_model.fit(X_[mask], y_.event[mask])

    def predict(self, X_):
        all_preds = self.ovr.predict(X_)  # (7, 4800, 100)
        if self.mode == 'best':
            events_pred = self.meta_model.predict(X_)
            selected_preds = np.zeros((X_.shape[0], 100))  # (4800, 100)
            for i in range(X_.shape[0]):
                selected_preds[i] = all_preds[events_pred[i] - 1, i, :]
            return selected_preds
        elif self.mode == 'weighted':
            events_pred = self.meta_model.predict_proba(X_)
            tmp = np.moveaxis(all_preds, (0, 1, 2), (1, 0, 2))
            return np.sum(tmp * events_pred[..., np.newaxis], axis=1)
        else:
            return all_preds


class SurvivalBoost:
    """Placeholder for SurvivalBoost model"""
    pass
