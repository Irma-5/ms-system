import os
import numpy as np
import yaml
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    
    def __init__(self, service_name: str = None):
        self.service_name = service_name or os.getenv('SERVICE_NAME', 'default')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        config_dir = os.getenv('CONFIG_DIR', '/app/config')
        config_file = os.path.join(config_dir, f'{self.service_name}.yaml')
        
        if not os.path.exists(config_file):
            config_dir = os.path.join(os.path.dirname(__file__), 'config')
            config_file = os.path.join(config_dir, f'{self.service_name}.yaml')
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded configuration from {config_file}")
                    return config
            except Exception as e:
                logger.error(f"Error loading config from {config_file}: {e}")
                return self._default_config()
        else:
            logger.warning(f"Config file not found: {config_file}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'service': {
                'name': self.service_name,
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
MODELS_DIR = os.path.join(STORAGE_DIR, 'models')
BATCHES_DIR = os.path.join(STORAGE_DIR, 'batches')
RESULTS_DIR = os.path.join(STORAGE_DIR, 'results')
METRICS_DIR = os.path.join(STORAGE_DIR, 'metrics')

STORAGE_URL = os.getenv('STORAGE_URL', 'http://localhost:8003')
COLLECTOR_URL = os.getenv('COLLECTOR_URL', 'http://localhost:8001')
MLSERVICE_URL = os.getenv('MLSERVICE_URL', 'http://localhost:8002')
VISUALIZATION_URL = os.getenv('VISUALIZATION_URL', 'http://localhost:8004')
WEBMASTER_URL = os.getenv('WEBMASTER_URL', 'http://localhost:8000')

WEBMASTER_HOST = os.getenv('WEBMASTER_HOST', '0.0.0.0')
WEBMASTER_PORT = int(os.getenv('SERVICE_PORT', 8000)) if os.getenv('SERVICE_NAME') == 'webmaster' else 8000

COLLECTOR_HOST = os.getenv('COLLECTOR_HOST', '0.0.0.0')
COLLECTOR_PORT = int(os.getenv('SERVICE_PORT', 8001)) if os.getenv('SERVICE_NAME') == 'collector' else 8001

MLSERVICE_HOST = os.getenv('MLSERVICE_HOST', '0.0.0.0')
MLSERVICE_PORT = int(os.getenv('SERVICE_PORT', 8002)) if os.getenv('SERVICE_NAME') == 'mlservice' else 8002

STORAGE_HOST = os.getenv('STORAGE_HOST', '0.0.0.0')
STORAGE_PORT = int(os.getenv('SERVICE_PORT', 8003)) if os.getenv('SERVICE_NAME') == 'storage' else 8003

VISUALIZATION_HOST = os.getenv('VISUALIZATION_HOST', '0.0.0.0')
VISUALIZATION_PORT = int(os.getenv('SERVICE_PORT', 8004)) if os.getenv('SERVICE_NAME') == 'visualization' else 8004

TIME_GRID_SIZE = 100
TIME_GRID = np.linspace(1, 309, 100) 
AVAILABLE_EVENTS = [1, 2, 3, 4, 5, 6, 7]

STATIC_COLS = ['credit_score', 'first_time_homebuyer_flag', 'units_numb', 'MSA', 'MI_%', 
               'occupancy_status', 'CLTV', 'DTI_ratio', 'orig_UPB', 'LTV', 'orig_interest_rate', 
               'channel', 'PPM_flag', 'amortization_type', 'property_state', 'property_type', 
               'loan_purpose', 'orig_loan_term', 'borrowers_num', 'super_conf_flag',
               'int_only_flag', 'property_val_method']

CATEG_COLS = ['occupancy_status', 'first_time_homebuyer_flag', 'channel', 'PPM_flag', 
              'amortization_type', 'property_state', 'borrowers_num', 'int_only_flag', 
              'property_val_method', 'property_type', 'loan_purpose', 'super_conf_flag']

for directory in [STORAGE_DIR, MODELS_DIR, BATCHES_DIR, RESULTS_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)
