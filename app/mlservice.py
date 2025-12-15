from flask import Flask, request, jsonify
import requests
import model_classes
import numpy as np
import pandas as pd
import pickle
from typing import List
from config import *
from functools import wraps
from sksurv.metrics import check_y_survival
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
import traceback
import gzip
import sys


app = Flask(__name__)
config = Config('mlservice')
TIME_GRID_SIZE = config.get('ml.time_grid_size', 100)
sys.modules['__main__'].MetaModel = model_classes.MetaModel
sys.modules['__main__'].OvR = model_classes.OvR
sys.modules['__main__'].SurvivalBoost = model_classes.SurvivalBoost

def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"File not found error in {f.__name__}: {str(e)}")
            return jsonify({
                'error': 'File not found',
                'message': str(e),
                'endpoint': f.__name__
            }), 404
        except ValueError as e:
            logger.error(f"Value error in {f.__name__}: {str(e)}")
            return jsonify({
                'error': 'Invalid value',
                'message': str(e),
                'endpoint': f.__name__
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'error': 'Internal server error',
                'message': str(e),
                'endpoint': f.__name__,
                'traceback': traceback.format_exc() if config.get('server.debug', False) else None
            }), 500
    return decorated_function


def get_y_self_event(y_, events: List = []):
    mask = np.isin(y_['event'], events)
    l = np.sum(mask)
    y = np.empty(dtype=[('event', bool), ('duration', np.float64)], shape=l)
    y["duration"] = y_[mask]["duration"]
    y["event"] = True
    return y


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



def ibs_remain(survival_train, survival_test, estimate, times, axis=-1):
    test_event, test_time = check_y_survival(survival_test, allow_all_censored=True)
    estimate = np.array(estimate)
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)
    estimate[estimate == -np.inf] = 0
    estimate[estimate == np.inf] = 0
    
    estim_before = np.square(estimate) * test_event[np.newaxis, :].T
    estim_after = np.square(1 - estimate)
    
    brier_scores = np.array([np.where(test_time < t,
                                      estim_before[:, i],
                                      estim_after[:, i])
                             for i, t in enumerate(times)])
    
    N = np.sum(np.array([np.where(test_time < t, test_event, 1)
                         for i, t in enumerate(times)]), axis=1)
    
    time_diff = times[-1] - times[0] if times[-1] > times[0] else 1
    
    if axis == -1:  # mean ibs for each time and observation
        brier_scores = np.where(N > 0, 1 / N, 0) * np.sum(brier_scores, axis=1)
        return np.trapz(brier_scores, times) / time_diff
    elif axis == 0:  # ibs for each observation
        return np.trapz(brier_scores, times, axis=0) / time_diff
    elif axis == 1:  # bs in time (for graphics)
        brier_scores = np.where(N > 0, 1 / N, 0) * np.sum(brier_scores, axis=1)
        return brier_scores
    return None


def calculate_survival_auprc_vectorized(y_true, preds, time_points, num_phi_steps=100):

    n_obs = preds.shape[0]
    ext_time_points = np.concatenate([[0], time_points])
    ext_probs = np.concatenate([np.ones((n_obs, 1)), preds], axis=1)
    ext_probs = np.minimum.accumulate(ext_probs, axis=1)
    true_times = y_true['duration'].reshape(-1, 1)
    is_event = y_true['event'].astype(bool).reshape(-1, 1)
    true_times[true_times == 0] = 1e-8
    phi_grid = np.linspace(0, 1, num_phi_steps).reshape(1, -1)
    t_early_matrix = true_times * phi_grid
    phi_grid_safe = np.copy(phi_grid)
    phi_grid_safe[phi_grid_safe == 0] = 1e-8
    t_late_matrix = true_times / phi_grid_safe
    t_late_matrix[:, phi_grid[0] == 0] = np.inf
    indices_early = np.searchsorted(ext_time_points, t_early_matrix, side='right') - 1
    indices_late = np.searchsorted(ext_time_points, t_late_matrix, side='right') - 1
    row_indexer = np.arange(n_obs).reshape(-1, 1)
    s_early = ext_probs[row_indexer, indices_early]
    s_late = ext_probs[row_indexer, indices_late]
    integral_values = np.where(is_event, s_early - s_late, s_early)
    auprc_scores = np.sum(integral_values, axis=1) / num_phi_steps
    
    return np.mean(auprc_scores)



@app.route('/models/list', methods=['GET'])
def list_models():
    try:
        response = requests.get(f'http://{STORAGE_HOST}:{STORAGE_PORT}/models/list')
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    JSON:
    {
        "model_filename": "model__name=cox__timestamp=20240101.pkl",
        "batch_filename": "batch__strategy=balanced__size=1000.pkl",
        "save_results": true/false,
        "metadata": {...}
    }
    """
    try:
        data = request.json
        model_filename = data.get('model_filename')
        batch_filename = data.get('batch_filename')
        time_grid_size = TIME_GRID_SIZE
        save_results = data.get('save_results', True)
        metadata = data.get('metadata', {})
        
        if not model_filename or not batch_filename:
            return jsonify({'error': 'model_filename and batch_filename required'}), 400
        
        model_response = requests.get(
            f'http://{STORAGE_HOST}:{STORAGE_PORT}/models/load/{model_filename}'
        )
        if model_response.status_code != 200:
            return jsonify({'error': 'Model not found'}), 404
        model_data = model_response.json()
        filepath = model_data['model_data']
        with gzip.open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded")
        # model_data = model_response.json()
        # logger.info(f"model_data unpacked")
        # model = pickle.loads(eval(model_data['model_data']))
        
        batch_response = requests.get(
            f'http://{COLLECTOR_HOST}:{COLLECTOR_PORT}/preprocess/{batch_filename}'
        )
        logger.info(f"batch loaded")
        if batch_response.status_code != 200:
            return jsonify({'error': 'Batch not found'}), 404
        
        batch_data = batch_response.json()
        X = np.array(batch_data['X'])
        y = batch_data['y']
        logger.info(f"batch x, y")
        y_duration = np.array(y['duration'])
        TIME_GRID = np.linspace(1, 309, 100)
        logger.info(f"y_duration, time_grid")
        try:
            predictions = model.predict(X)
            logger.info(f"predicted")
        except Exception as e:
            logger.info(f"Error predict")
            traceback.print_exc()  
        
        metadata['model_filename'] = model_filename
        metadata['batch_filename'] = batch_filename
        metadata['n_samples'] = len(X)
        metadata['time_grid_size'] = time_grid_size
        metadata['TIME_GRID'] = TIME_GRID.tolist()
        
        result = {
            'success': True,
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'metadata': metadata
        }
        
        if save_results:
            save_response = requests.post(
                f'http://{STORAGE_HOST}:{STORAGE_PORT}/results/save',
                json={
                    'predictions': result['predictions'],
                    'metadata': metadata
                }
            )
            if save_response.status_code == 200:
                result['saved_filename'] = save_response.json()['filename']
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()  
        return jsonify({'error': str(e)}), 500


@app.route('/metrics/compute', methods=['POST'])
def compute_metrics():
    """
    JSON:
    {
        "predictions_filename": "results__model=cox__batch=test.pkl",
        "batch_filename": "batch__strategy=balanced.pkl",
        "train_batch_filename": "batch__strategy=train.pkl",
        "save_metrics": true/false,
        "metadata": {...}
    }
    """
    try:
        logger.info('read data')
        data = request.json
        predictions_filename = data.get('predictions_filename')
        logger.info(predictions_filename)
        batch_filename = data.get('batch_filename')
        logger.info(batch_filename)
        train_batch_filename='batch_train.pkl'
        # train_batch_filename = data.get('train_batch_filename')
        save_metrics = data.get('save_metrics', True)
        metadata = data.get('metadata', {})
        
        pred_response = requests.get(
            f'http://{STORAGE_HOST}:{STORAGE_PORT}/results/load/{predictions_filename}'
        )
        if pred_response.status_code != 200:
            return jsonify({'error': 'Predictions not found'}), 404
        
        pred_data = pred_response.json()
        sf = np.array(pred_data['predictions'])
        logger.info(sf.shape)
        TIME_GRID = np.linspace(1, 309, 100)
        
        batch_response = requests.get(
            f'http://{STORAGE_HOST}:{STORAGE_PORT}/batches/load/{batch_filename}'
        )
        if batch_response.status_code != 200:
            return jsonify({'error': 'Test batch not found'}), 404
        
        batch_data = batch_response.json()
        y_test = batch_data['y']
        # print()
        # logger.info("Тип y: %s", type(y_test))
        # print(type(y_test))
        # print()
        y_test_event = np.array(y_test['event'])
        y_test_duration = np.array(y_test['duration'])
        
        y_test_struct = np.empty(dtype=[('event', int), ('duration', np.float64)], 
                                 shape=len(y_test_event))
        y_test_struct['event'] = y_test_event
        y_test_struct['duration'] = y_test_duration
        
        train_response = requests.get(
            f'http://{STORAGE_HOST}:{STORAGE_PORT}/batches/load/{train_batch_filename}'
        )
        if train_response.status_code != 200:
            return jsonify({'error': 'Train batch not found'}), 404
        
        train_data = train_response.json()
        y_train = train_data['y']
        y_train_event = np.array(y_train['event'])
        y_train_duration = np.array(y_train['duration'])
        
        y_train_struct = np.empty(dtype=[('event', int), ('duration', np.float64)], 
                                  shape=len(y_train_event))
        y_train_struct['event'] = y_train_event
        y_train_struct['duration'] = y_train_duration
        
        unique_events = np.unique(y_train_event)
        unique_events = unique_events[unique_events > 0]
        
        ibs_scores = {}
        auprc_scores = {}
        
        for i in unique_events:
            event_idx = i - 1

            if i > 0:
            
            # if event_idx < predictions.shape[0]:
                # sf = predictions[event_idx, ...]
                
                y_train_i = get_y_self_event(y_train_struct, [i])
                y_test_i = get_y_self_event(y_test_struct, [i])
                
                event_mask = y_test_event == i
                sf_event = sf[event_mask, :]
                
                ibs = ibs_remain(y_train_i, y_test_i, sf_event, 
                                    times=TIME_GRID, axis=-1)
                ibs_scores[f'event_{i}'] = float(ibs)
                    
                auprc = calculate_survival_auprc_vectorized(
                        y_test_i, sf_event, TIME_GRID
                    )
                auprc_scores[f'event_{i}'] = float(auprc)
        
        metrics_result = {
            'ibs': ibs_scores,
            'auprc': auprc_scores,
            'ibs_mean': float(np.mean(list(ibs_scores.values()))) if ibs_scores else None,
            'auprc_mean': float(np.mean(list(auprc_scores.values()))) if auprc_scores else None
        }
        
        metadata['predictions_filename'] = predictions_filename
        metadata['batch_filename'] = batch_filename
        metadata['train_batch_filename'] = train_batch_filename
        metadata['n_events'] = len(unique_events)
        
        result = {
            'success': True,
            'metrics': metrics_result,
            'metadata': metadata
        }
        
        if save_metrics:
            save_response = requests.post(
                f'http://{STORAGE_HOST}:{STORAGE_PORT}/metrics/save',
                json={
                    'metrics': metrics_result,
                    'metadata': metadata
                }
            )
            if save_response.status_code == 200:
                result['saved_filename'] = save_response.json()['filename']
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'service': 'MLService',
        'status': 'healthy' # TODO придумать что проверять
    })


if __name__ == '__main__':
    host = os.environ.get('SERVICE_HOST', '0.0.0.0')
    port = int(os.environ.get('SERVICE_PORT', MLSERVICE_PORT))
    print(config.service_name)
    app.run(host=host, port=port, debug=True)