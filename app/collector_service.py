from flask import Flask, request, jsonify
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from config import *
import traceback

from functools import reduce, wraps

app = Flask(__name__)
config = Config('collector')


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
        except PermissionError as e:
            logger.error(f"Permission error in {f.__name__}: {str(e)}")
            return jsonify({
                'error': 'Permission denied',
                'message': str(e),
                'endpoint': f.__name__
            }), 403
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


def str_to_categ(df_col):
    uniq = df_col.unique()
    return df_col.map(dict(zip(uniq, range(len(uniq)))))


def get_y(cens, time):
    cens, time = np.array(cens), np.array(time)
    y = np.empty(dtype=[('event', int), ('duration', np.float64)], shape=cens.shape[0])
    y['event'] = cens
    y['duration'] = time
    return y


def transform_events(y):
    events = sorted(y.event.unique())
    d = {events[i]: i for i in range(len(events))}
    return y.replace({"event": d}), d


class Scaler:
    
    def __init__(self):
        self.constant_cols = ['int_only_flag', 'property_val_method', 'super_conf_flag', 'amortization_type']
        self.categs = list((set(STATIC_COLS) & set(CATEG_COLS)) - set(self.constant_cols))
        self.enc = ColumnTransformer(
            transformers=[('ohe', OneHotEncoder(sparse_output=False).set_output(transform="pandas"), self.categs)],
            remainder='passthrough'
        )
    
    def fit(self, list_of_df):
        X = pd.concat(list_of_df, axis=0)
        # X.drop(self.constant_cols, inplace=True, axis=1)
        self.enc.fit(X)
    
    def transform(self, X):
        X.MSA.fillna(X.MSA.median(), inplace=True)
        # X.drop(self.constant_cols, inplace=True, axis=1)
        X = self.enc.transform(X)
        scaler = StandardScaler().set_output(transform="pandas")
        X = scaler.fit_transform(X)
        return X
    

@app.route('/batch/create_from_storage', methods=['POST'])
def create_batch_from_storage(): # создает из batch_0 по заданной стратегии батч и сохраняет его без предобработки
    """
    JSON:
    {
        "source_batch": "batch_0" (default),
        "batch_size": 1000,
        "strategy": "balanced" | "imbalanced" | "random",
        "stratify_on_event": true/false,
        "metadata": {...}
    }
    """
    data = request.json
    
    source_batch = data.get('source_batch', 'batch_0')
    batch_size = data.get('batch_size', config.get('batch.default_size', 1000))
    strategy = data.get('strategy', config.get('batch.default_strategy', 'random'))
    metadata = data.get('metadata', {})
    
    max_size = config.get('batch.max_size', 50000)
    if batch_size > max_size:
        raise ValueError(f'Batch size {batch_size} exceeds maximum {max_size}')
    
    allowed_strategies = config.get('batch.strategies', ['balanced', 'imbalanced', 'random'])
    if strategy not in allowed_strategies:
        raise ValueError(f'Strategy must be one of {allowed_strategies}')
    
    logger.info(f"Loading source batch: {source_batch}")
    
    response = requests.get(f'{STORAGE_URL}/batches/list', timeout=config.get('dependencies.timeout_seconds', 60))
    if response.status_code != 200:
        raise Exception('Failed to list batches from storage')
    
    batches = response.json()['batches']
    source_file = None
    
    for batch in batches:
        if source_batch in batch['metadata']['filename']:
            source_file = batch['metadata']['filename']
            break
    
    if not source_file:
        raise FileNotFoundError(f'Source batch "{source_batch}" not found in storage')
    
    response = requests.get(
        f'{STORAGE_URL}/batches/load/{source_file}',
        timeout=config.get('dependencies.timeout_seconds', 60)
    )
    
    if response.status_code != 200:
        raise Exception(f'Failed to load batch {source_file}')
    
    batch_data = response.json()
    X = pd.DataFrame(batch_data['X'])
    y = pd.DataFrame(batch_data['y'])
    
    logger.info(f"Loaded source batch with {len(X)} samples")
    
    if strategy == 'balanced':
        event_groups = []
        events = y['event'].unique()
        samples_per_event = batch_size // len(events)
        
        for event in events:
            event_mask = y['event'] == event
            event_indices = y[event_mask].index
            if len(event_indices) > samples_per_event:
                sampled_indices = np.random.RandomState(42).choice(
                    event_indices, samples_per_event, replace=False
                )
            else:
                sampled_indices = event_indices
            event_groups.append(sampled_indices)
        
        selected_indices = np.concatenate(event_groups)
        X_batch = X.loc[selected_indices]
        y_batch = y.loc[selected_indices]
        
    elif strategy == 'random':
        if len(X) > batch_size:
            selected_indices = np.random.RandomState(42).choice(len(X), batch_size, replace=False)
            X_batch = X.iloc[selected_indices]
            y_batch = y.iloc[selected_indices]
        else:
            X_batch = X
            y_batch = y
            
   
    
    metadata['strategy'] = strategy
    metadata['batch_size'] = len(X_batch)
    metadata['source_batch'] = source_file
    
    event_dist = y_batch['event'].value_counts().to_dict()
    metadata['event_distribution'] = {str(k): int(v) for k, v in event_dist.items()}
    
    logger.info(f"Created batch with {len(X_batch)} samples, strategy: {strategy}")
    
    batch_data = {
        'X': X_batch.to_numpy().tolist(),
        'y': {
            'event': y_batch['event'].tolist(),
            'duration': y_batch['duration'].tolist()
        },
        'metadata': metadata
    }
    
    response = requests.post(
        f'{STORAGE_URL}/batches/save',
        json=batch_data,
        timeout=config.get('dependencies.timeout_seconds', 60)
    )
    
    if response.status_code == 200:
        result = response.json()
        result['success'] = True
        result['event_distribution'] = metadata['event_distribution']
        logger.info(f"Batch saved: {result['filename']}")
        return jsonify(result)
    else:
        raise Exception('Failed to save batch to storage')



@app.route('/batch/create', methods=['POST'])
def create_batch(): # создает данные из загруженных
    """
    JSON:
    {
        "data_source": "path/to/data.csv",
        "batch_size": 1000,
        "strategy": "balanced" | "imbalanced" | "random",
        "test_size": 0.2,
        "metadata": {...} - если "imbalanced" то для каждого события число наблюдений
    }
    """
    try:
        data = request.json
        
        data_source = data.get('data_source')
        batch_size = data.get('batch_size', 1000)
        strategy = data.get('strategy', 'random')  # balanced, imbalanced, random
        test_size = data.get('test_size', 0.2)
        metadata = data.get('metadata', {})
        
        # TODO - доделать создание батча из существующих данных, пока только загружаются
        if 'X' in data and 'y' in data:
            X = pd.DataFrame(data['X'])
            y = pd.DataFrame(data['y'])
        else:
            return jsonify({'error': 'No data provided. Include X and y in request'}), 400
        
        if strategy == 'balanced':
            event_groups = []
            events = y['event'].unique()
            samples_per_event = batch_size // len(events)
            
            for event in events:
                event_mask = y['event'] == event
                event_indices = y[event_mask].index
                if len(event_indices) > samples_per_event:
                    sampled_indices = np.random.choice(event_indices, samples_per_event, replace=False)
                else:
                    samples_per_event = len(event_indices) # 11
                    sampled_indices = event_indices
                event_groups.append(sampled_indices)
            
            selected_indices = np.concatenate(event_groups)
            X_batch = X.loc[selected_indices]
            y_batch = y.loc[selected_indices]
            
        elif strategy == 'imbalanced':
            pass
            # TODO сюда добавить обработку метадата, и взять столько событий каждого типа, сколько в метадата
                
        elif strategy == 'random':
            if len(X) > batch_size:
                selected_indices = np.random.choice(len(X), batch_size, replace=False)
                X_batch = X.iloc[selected_indices]
                y_batch = y.iloc[selected_indices]
            else:
                X_batch = X
                y_batch = y
        else:
            return jsonify({'error': f'Unknown strategy: {strategy}'}), 400
        
        metadata['strategy'] = strategy
        metadata['batch_size'] = len(X_batch)
        event_dist = y_batch['event'].value_counts().to_dict()
        metadata['event_distribution'] = {str(k): int(v) for k, v in event_dist.items()}
        
        batch_data = {
            'X': X_batch.to_numpy().tolist(),
            'y': {
                'event': y_batch['event'].tolist(),
                'duration': y_batch['duration'].tolist()
            },
            'metadata': metadata
        }
        
        response = requests.post(
            f'http://{STORAGE_HOST}:{STORAGE_PORT}/batches/save',
            json=batch_data
        )
        
        if response.status_code == 200:
            result = response.json()
            result['event_distribution'] = metadata['event_distribution']
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to save batch'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch/statistics/<filename>', methods=['GET'])
def batch_statistics(filename):
    response = requests.get(
        f'{STORAGE_URL}/batches/load/{filename}',
        timeout=config.get('dependencies.timeout_seconds', 60)
    )
    
    if response.status_code != 200:
        raise FileNotFoundError(f'Batch not found: {filename}')
    
    batch_data = response.json()
    y = pd.DataFrame(batch_data['y'])
    
    stats = {
        'total_samples': len(y),
        'event_counts': y['event'].value_counts().to_dict(),
    }
    
    events = sorted(y['event'].unique())
    per_event_stats = {}
    
    for event in events:
        event_mask = y['event'] == event
        event_durations = y[event_mask]['duration']
        
        per_event_stats[str(event)] = {
            'count': int(len(event_durations)),
            'min': float(event_durations.min()),
            'max': float(event_durations.max()),
            'mean': float(event_durations.mean()),
            'median': float(event_durations.median()),
            'std': float(event_durations.std())
        }
    
    stats['per_event_duration_stats'] = per_event_stats
    
    time_bins = config.get('statistics.time_bins', 20)
    duration_max = y['duration'].max()
    duration_min = y['duration'].min()
    bins = np.linspace(duration_min, duration_max, time_bins + 1)
    
    event_time_dist = {}
    for event in events:
        event_mask = y['event'] == event
        event_durations = y[event_mask]['duration']
        hist, _ = np.histogram(event_durations, bins=bins)
        event_time_dist[str(event)] = {
            'histogram': hist.tolist(),
            'durations': event_durations.tolist() 
        }
    
    stats['event_time_distribution'] = {
        'bins': bins.tolist(),
        'distributions': event_time_dist
    }
    
    logger.info(f"Computed statistics for batch: {filename}")
    
    return jsonify({
        'success': True,
        'statistics': stats,
        'metadata': batch_data['metadata']
    })

@app.route('/preprocess/<source_batch>', methods=['GET'])
@handle_errors
def preprocess_from_storage(source_batch):

    response = requests.get(f'{STORAGE_URL}/batches/list')
    if response.status_code != 200:
        logger.info('responce failed to list batches')
        raise Exception('Failed to list batches')
    
    batches = response.json()['batches']
    source_file = None
    
    for batch in batches:
        if source_batch in batch['metadata']['filename']:
            source_file = batch['metadata']['filename']
            break
    
    if not source_file:
        logger.info('batch not found')
        raise FileNotFoundError(f'Batch "{source_batch}" not found')
    
    response = requests.get(f'{STORAGE_URL}/batches/load/{source_file}')
    if response.status_code != 200:
        raise Exception(f'Failed to load batch {source_file}')
    
    logger.info('received batch')
    batch_data = response.json()
    X = pd.DataFrame(batch_data['X'])
    logger.info('X = pd.DataFrame')
    scaler = Scaler()
    logger.info('scaler')
    # scaler.fit([X])
    logger.info('scaler fit')
    # X_transformed = scaler.transform(X.copy())
    logger.info('scaler trasnm')
    y = pd.DataFrame(batch_data['y'])
    logger.info('y = pd.DataFrame')
    y, dct = transform_events(y)
    logger.info(type(y))
    y = {'event': y['event'].to_numpy().tolist(), 'duration': y['duration'].to_numpy().tolist()}

    
    result = {
        'success': True,
        'X': X.to_numpy().tolist(),
        'y': y
    }
    
    logger.info(f"Preprocessed batch: {source_file}")
    
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'service': 'Collector',
        'status': 'healthy' # TODO придумать что проверять
    })


if __name__ == '__main__':
    host = os.environ.get('SERVICE_HOST', '0.0.0.0')
    port = int(os.environ.get('SERVICE_PORT', COLLECTOR_PORT))
    print(config.service_name)
    app.run(host=host, port=port, debug=True)