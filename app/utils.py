import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from functools import reduce
from typing import List


dtypes = {
    'credit_score': 'Int16', 'first_payment_date': 'str', 'first_time_homebuyer_flag': 'str',
    'maturity_date': 'str', 'MSA': 'Int32', 'MI_%': 'Int16', 'units_numb': 'Int8', 
    'occupancy_status': 'str', 'CLTV': 'Int16', 'DTI_ratio': 'Int16', 'orig_UPB': 'Int64', 
    'LTV': 'Int16', 'orig_interest_rate': 'str', 'channel': 'str', 'PPM_flag': 'str', 
    'amortization_type': 'str', 'property_state': 'str', 'property_type': 'str', 
    'postal_code': 'Int32', 'id_loan': 'str', 'loan_purpose': 'str', 'orig_loan_term': 'Int16', 
    'borrowers_num': 'Int8', 'seller_name': 'str', 'service_name': 'str', 'super_conf_flag': 'str', 
    'id_loan_preharp': 'str', 'program_ind': 'str', 'HARP_ind': 'str', 'property_val_method': 'Int64',
    'int_only_flag': 'str', 'MI_cancel_flag': 'str', 'orig_interest_rate': 'float32'
}

static = [
    'credit_score', 'first_time_homebuyer_flag', 'units_numb', 'MSA', 'MI_%', 
    'occupancy_status', 'CLTV', 'DTI_ratio', 'orig_UPB', 'LTV', 'orig_interest_rate', 
    'channel', 'PPM_flag', 'amortization_type', 'property_state', 'property_type', 
    'loan_purpose', 'orig_loan_term', 'borrowers_num', 'super_conf_flag',
    'int_only_flag', 'property_val_method'
]

categ = [
    'occupancy_status', 'first_time_homebuyer_flag', 'channel', 'PPM_flag', 
    'amortization_type', 'property_state', 'borrowers_num', 'int_only_flag', 
    'property_val_method', 'modification_flag', 'step_mod_flag', 'deferred_payment_plan',
    'ELTV', 'delinq_due_disaster', 'borrowe_asistance_stat_code', 'property_type', 
    'loan_purpose', 'super_conf_flag'
]


def get_y(cens, time):
    cens, time = np.array(cens), np.array(time)
    y = np.empty(dtype=[('event', int), ('duration', np.float64)], shape=cens.shape[0])
    y['event'] = cens
    y['duration'] = time
    return y


def get_y_arr(y):
    cens, time = np.array(y.event), np.array(y.duration)
    y = np.empty(dtype=[('event', bool), ('duration', np.float64)], shape=cens.shape[0])
    y['event'] = cens
    y['duration'] = time
    return y


def get_y_event(y_, events: List = []):
    y = np.empty(dtype=[('event', bool), ('duration', np.float64)], shape=y_.shape[0])
    y['event'] = y_.event.isin(events)
    y['duration'] = y_.duration
    return y



def case1(X_, y_, events: List = []):
    y, X = y_.copy(), X_.copy()
    mask = y_.event.isin(events)
    y.event = mask.astype('int')
    return X, y


def case2(X_, y_, events: List = []):
    y, X = y_.copy(), X_.copy()
    mask = y_.event.isin(events)
    y.event = mask
    y = y[mask]
    X = X[mask]
    return X, y


def case3(X_, y_, events: List = [], ):
    y, X = y_.copy(), X_.copy()
    mask = y_.event.isin(events)
    maxm = y_.duration[mask].max()
    y.event = mask
    X = X[y.duration <= maxm]
    y = y[y.duration <= maxm]
    return X, y



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


def transform_curves(curves):
    if len(curves.shape) == 1:
        curves = curves[None, :]
    return np.array(list(map(
        lambda tmp: reduce(
            lambda c, x: (c[0], c[1] + [c[0]]) if x > c[0] else (x, c[1] + [x]),
            tmp[1:], (tmp[0], [tmp[0]])
        )[1],
        curves
    )))


def transform_events(y):
    events = sorted(y.event.unique())
    d = {events[i]: i for i in range(len(events))}
    return y.replace({"event": d}), d


def step_to_array(step_functions):
    shape_ = (step_functions.shape[0], step_functions[0].x.shape[0])
    arr = np.empty(shape=shape_)
    for i in range(len(step_functions)):
        arr[i] = step_functions[i].y
    return arr, step_functions[0].x


def str_to_categ(df_col):
    uniq = df_col.unique()
    return df_col.map(dict(zip(uniq, range(len(uniq)))))


class Scaler:
    
    def __init__(self):
        self.constant_cols = ['int_only_flag', 'property_val_method', 'super_conf_flag', 'amortization_type']
        self.categs = list((set(static) & set(categ)) - set(self.constant_cols))
        self.enc = ColumnTransformer(
            transformers=[('ohe', OneHotEncoder(sparse_output=False).set_output(transform="pandas"), self.categs)],
            remainder='passthrough'
        )
    
    def fit(self, list_of_df):
        X = pd.concat(list_of_df, axis=0)
        X.drop(self.constant_cols, inplace=True, axis=1)
        self.enc.fit(X)
    
    def transform(self, X):
        X.MSA.fillna(X.MSA.median(), inplace=True)
        X.drop(self.constant_cols, inplace=True, axis=1)
        X = self.enc.transform(X)
        scaler = StandardScaler().set_output(transform="pandas")
        X = scaler.fit_transform(X)
        return X


def bal21_sample(file_path):
    df = pd.read_csv(file_path, dtype=dtypes)
    df['event'] = df.zero_balance_code.astype('int') * (df.cens.astype('int'))
    df = df[static + ['time', 'event']]
    df = df.apply(lambda x: str_to_categ(x) if x.name in categ else x, axis=0)
    sign = sorted(list(set(df.columns) - {'time', 'event'}))
    y = get_y(df['event'], df['time'] + 1)
    X = df.loc[:, sign]
    return y, X, sign, categ, df


def rand21_sample(file_path):
    df = pd.read_csv(file_path, dtype=dtypes)
    df['event'] = df.zero_balance_code.astype('int') * (df.cens.astype('int'))
    df = df[static + ['time', 'event']]
    df = df.apply(lambda x: str_to_categ(x) if x.name in categ else x, axis=0)
    sign = sorted(list(set(df.columns) - {'time', 'event'}))
    y = get_y(df['event'], df['time'] + 1)
    X = df.loc[:, sign]
    return y, X, sign, categ, df


def bal280_sample(file_path):
    df = pd.read_csv(file_path, dtype=dtypes)
    df['event'] = df.zero_balance_code.astype('int') * (df.cens.astype('int'))
    df = df[static + ['time', 'event']]
    df = df.apply(lambda x: str_to_categ(x) if x.name in categ else x, axis=0)
    sign = sorted(list(set(df.columns) - {'time', 'event'}))
    y = get_y(df['event'], df['time'] + 1)
    X = df.loc[:, sign]
    return y, X, sign, categ, df



def preprocess_data(file_path, test_size=0.2, val_size=0.25, random_state=1):

    y_large, x_large, _, _, _ = bal280_sample(file_path)
    sc = Scaler()
    sc.fit([x_large])
    x_large = sc.transform(x_large)
    y_large = pd.DataFrame(y_large)
    y_large, dct = transform_events(y_large)
    x_train, x_test, y_train, y_test = train_test_split(
        x_large, y_large, test_size=test_size, stratify=y_large.event, random_state=random_state
    )
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, stratify=y_train.event, random_state=random_state
    )
    
    TIME_GRID = np.linspace(y_train['duration'].min(), y_train['duration'].max(), 100)
    
    return x_train, x_val, x_test, y_train, y_val, y_test, TIME_GRID, sc


def check_y_survival(y, allow_all_censored=True):
    """Extract event and time from structured array"""
    event = y['event']
    time = y['duration']
    return event, time


def get_y_self_event(y_, events: List = []):
    """Get survival data for specific events (for metrics)"""
    mask = np.isin(y_['event'], events)
    l = np.sum(mask)
    y = np.empty(dtype=[('event', bool), ('duration', np.float64)], shape=l)
    y["duration"] = y_[mask]["duration"]
    y["event"] = True
    return y