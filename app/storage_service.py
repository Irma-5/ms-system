"""
Storage Service - manages models, batches, results, and metrics storage
With error handling and configuration loading
"""
from flask import Flask, request, jsonify
import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import traceback
from functools import wraps
from config import Config, STORAGE_HOST, STORAGE_PORT

app = Flask(__name__)
config = Config('storage')
logging.basicConfig(
    level=getattr(logging, config.get('logging.level', 'INFO')),
    format=config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)

STORAGE_DIR = config.get('storage.base_dir', '/app/storage')
MODELS_DIR = config.get('storage.models_dir', f'{STORAGE_DIR}/models')
BATCHES_DIR = config.get('storage.batches_dir', f'{STORAGE_DIR}/batches')
RESULTS_DIR = config.get('storage.results_dir', f'{STORAGE_DIR}/results')
METRICS_DIR = config.get('storage.metrics_dir', f'{STORAGE_DIR}/metrics')

for directory in [STORAGE_DIR, MODELS_DIR, BATCHES_DIR, RESULTS_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)


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



class StorageManager:
    """Manages file storage with metadata in filenames"""
    
    @staticmethod
    def encode_filename(base_name, metadata):
        """Create filename with metadata"""
        try:
            meta_str = '__'.join([f"{k}={v}" for k, v in metadata.items()])
            return f"{base_name}__{meta_str}"
        except Exception as e:
            logger.error(f"Error encoding filename: {e}")
            raise ValueError(f"Invalid metadata format: {e}")
    
    @staticmethod
    def decode_filename(filename):
        """Extract metadata from filename"""
        try:
            parts = filename.split('__')
            base_name = parts[0]
            metadata = {}
            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    # value = value.replace('.pkl', '').replace('.csv', '').replace('.json', '')
                    metadata[key] = value
            return base_name, metadata
        except Exception as e:
            logger.error(f"Error decoding filename: {e}")
            return filename, {}
    
    @staticmethod
    def list_files_with_metadata(directory):
        """List all files in directory with their metadata"""
        files_info = []
        try:
            if not os.path.exists(directory):
                logger.warning(f"Directory does not exist: {directory}")
                return files_info
            
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    try:
                        base_name, metadata = StorageManager.decode_filename(filename)
                        metadata['filename'] = filename
                        metadata['filepath'] = filepath
                        metadata['size_mb'] = round(os.path.getsize(filepath) / (1024*1024), 2)
                        metadata['created'] = datetime.fromtimestamp(
                            os.path.getctime(filepath)
                        ).strftime('%Y-%m-%d %H:%M:%S')
                        files_info.append({
                            'base_name': base_name,
                            'metadata': metadata
                        })
                    except Exception as e:
                        logger.warning(f"Error processing file {filename}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {e}")
        
        return files_info
    
    @staticmethod
    def validate_file_size(filepath, max_size_mb=None):
        """Validate file size"""
        if max_size_mb is None:
            max_size_mb = config.get('storage.max_file_size_mb', 1000)
        
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValueError(f"File size ({size_mb:.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)")
        return True



@app.route('/models/save', methods=['POST'])
@handle_errors
def save_model():
    data = request.json
    model_data = data.get('model_data')
    metadata = data.get('metadata', {})
    
    if not model_data:
        raise ValueError('model data is required')
    
    if 'model_name' not in metadata:
        raise ValueError('model name is required in metadata')
    
    # понадобится если добавим версионирование
    if 'timestamp' not in metadata:
        metadata['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    filename = StorageManager.encode_filename('model', metadata) + '.pkl'
    filepath = os.path.join(MODELS_DIR, filename)
    
    with open(filepath, 'wb') as f:
        f.write(pickle.loads(eval(model_data)))
    
    # StorageManager.validate_file_size(filepath)
    
    logger.info(f"Model saved: {filename}")
    return jsonify({
        'success': True,
        'filename': filename,
        'filepath': filepath
    })


@app.route('/models/load/<filename>', methods=['GET'])
@handle_errors
def load_model(filename):
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Model not found: {filename}')
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    base_name, metadata = StorageManager.decode_filename(filename)
    
    logger.info(f"Model loaded: {filename}")
    return jsonify({
        'success': True,
        'metadata': metadata,
        'model_data': str(pickle.dumps(model))
    })


@app.route('/models/list', methods=['GET'])
@handle_errors
def list_models():
    """List all saved models"""
    models = StorageManager.list_files_with_metadata(MODELS_DIR)
    logger.info(f"Listed {len(models)} models")
    return jsonify({
        'success': True,
        'count': len(models),
        'models': models
    })


@app.route('/models/delete/<filename>', methods=['DELETE'])
@handle_errors
def delete_model(filename):
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Model not found: {filename}')
    
    os.remove(filepath)
    logger.info(f"Model deleted: {filename}")
    return jsonify({'success': True, 'message': 'Model deleted'})


@app.route('/batches/save', methods=['POST'])
@handle_errors
def save_batch():
    data = request.json
    X = np.array(data['X'])
    y = data['y']
    metadata = data.get('metadata', {})
    
    if 'timestamp' not in metadata:
        metadata['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    metadata['size'] = str(len(X))
    
    filename = StorageManager.encode_filename('batch', metadata) + '.pkl'
    filepath = os.path.join(BATCHES_DIR, filename)
    
    batch_data = {'X': X, 'y': y, 'metadata': metadata}
    
    with open(filepath, 'wb') as f:
        pickle.dump(batch_data, f)
    
    StorageManager.validate_file_size(filepath)
    
    logger.info(f"Batch saved: {filename}")
    return jsonify({
        'success': True,
        'filename': filename,
        'filepath': filepath
    })


@app.route('/batches/load/<filename>', methods=['GET'])
@handle_errors
def load_batch(filename):
    filepath = os.path.join(BATCHES_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Batch not found: {filename}')
    with open(filepath, 'rb') as f:
        batch_data = pickle.load(f)
    logger.info(f"Batch loaded: {filename}")
    return jsonify({
        'success': True,
        'X': batch_data['X'].tolist(),
        'y': batch_data['y'],
        'metadata': batch_data['metadata']
    })


@app.route('/batches/list', methods=['GET'])
@handle_errors
def list_batches():
    batches = StorageManager.list_files_with_metadata(BATCHES_DIR)
    logger.info(f"Listed {len(batches)} batches")
    return jsonify({
        'success': True,
        'count': len(batches),
        'batches': batches
    })


@app.route('/batches/delete/<filename>', methods=['DELETE'])
@handle_errors
def delete_batch(filename):
    filepath = os.path.join(BATCHES_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Batch not found: {filename}')
    os.remove(filepath)
    logger.info(f"Batch deleted: {filename}")
    return jsonify({'success': True, 'message': 'Batch deleted'})



@app.route('/results/save', methods=['POST'])
@handle_errors
def save_results():
    data = request.json
    predictions = np.array(data['predictions'])
    metadata = data.get('metadata', {})
    if 'timestamp' not in metadata:
        metadata['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = StorageManager.encode_filename('results', metadata) + '.pkl'
    filepath = os.path.join(RESULTS_DIR, filename)
    results_data = {'predictions': predictions, 'metadata': metadata}
    with open(filepath, 'wb') as f:
        pickle.dump(results_data, f)
    logger.info(f"Results saved: {filename}")
    return jsonify({
        'success': True,
        'filename': filename,
        'filepath': filepath
    })


@app.route('/results/load/<filename>', methods=['GET'])
@handle_errors
def load_results(filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Results not found: {filename}')
    
    with open(filepath, 'rb') as f:
        results_data = pickle.load(f)
    logger.info(f"Results loaded: {filename}")
    return jsonify({
        'success': True,
        'predictions': results_data['predictions'].tolist(),
        'metadata': results_data['metadata']
    })


@app.route('/results/list', methods=['GET'])
@handle_errors
def list_results():
    """List all saved results"""
    results = StorageManager.list_files_with_metadata(RESULTS_DIR)
    logger.info(f"Listed {len(results)} results")
    return jsonify({
        'success': True,
        'count': len(results),
        'results': results
    })


@app.route('/metrics/save', methods=['POST'])
@handle_errors
def save_metrics():
    data = request.json
    metrics = data['metrics']
    metadata = data.get('metadata', {})
    if 'timestamp' not in metadata:
        metadata['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = StorageManager.encode_filename('metrics', metadata) + '.json'
    filepath = os.path.join(METRICS_DIR, filename)
    metrics_data = {'metrics': metrics, 'metadata': metadata}
    with open(filepath, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Metrics saved: {filename}")
    return jsonify({
        'success': True,
        'filename': filename,
        'filepath': filepath
    })


@app.route('/metrics/load/<filename>', methods=['GET'])
@handle_errors
def load_metrics(filename):
    filepath = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Metrics not found: {filename}')
    with open(filepath, 'r') as f:
        metrics_data = json.load(f)
    logger.info(f"Metrics loaded: {filename}")
    return jsonify({
        'success': True,
        'metrics': metrics_data['metrics'],
        'metadata': metrics_data['metadata']
    })


@app.route('/health', methods=['GET'])
def health():
    try:
        dirs_status = {}
        for name, path in [
            ('models', MODELS_DIR),
            ('batches', BATCHES_DIR),
            ('results', RESULTS_DIR),
            ('metrics', METRICS_DIR)
        ]:
            dirs_status[name] = {
                'accessible': os.path.exists(path) and os.access(path, os.W_OK),
                'count': len(os.listdir(path)) if os.path.exists(path) else 0
            }
        
        return jsonify({
            'service': 'Storage',
            'status': 'healthy',
            'directories': dirs_status
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'service': 'Storage',
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    host = config.get('server.host', STORAGE_HOST)
    port = config.get('server.port', STORAGE_PORT)
    debug = config.get('server.debug', False)
    
    logger.info(f"Starting Storage Service on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
