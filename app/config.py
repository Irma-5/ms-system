"""
Configuration for Survival Analysis Microservices
Works both locally and in Docker containers
"""
import os
import yaml
import logging

# =============================================================================
# SERVICE HOSTS AND PORTS
# =============================================================================
# В Docker используем имена сервисов, локально - localhost
# Переменные окружения имеют приоритет

STORAGE_HOST = os.environ.get('STORAGE_HOST', 'storage')
STORAGE_PORT = int(os.environ.get('STORAGE_PORT', 8003))

COLLECTOR_HOST = os.environ.get('COLLECTOR_HOST', 'collector')
COLLECTOR_PORT = int(os.environ.get('COLLECTOR_PORT', 8001))

MLSERVICE_HOST = os.environ.get('MLSERVICE_HOST', 'mlservice')
MLSERVICE_PORT = int(os.environ.get('MLSERVICE_PORT', 8002))

VISUALIZATION_HOST = os.environ.get('VISUALIZATION_HOST', 'visualization')
VISUALIZATION_PORT = int(os.environ.get('VISUALIZATION_PORT', 8004))

WEBMASTER_HOST = os.environ.get('WEBMASTER_HOST', '0.0.0.0')
WEBMASTER_PORT = int(os.environ.get('WEBMASTER_PORT', 8000))

# URL shortcuts
STORAGE_URL = f'http://{STORAGE_HOST}:{STORAGE_PORT}'
COLLECTOR_URL = f'http://{COLLECTOR_HOST}:{COLLECTOR_PORT}'
MLSERVICE_URL = f'http://{MLSERVICE_HOST}:{MLSERVICE_PORT}'
VISUALIZATION_URL = f'http://{VISUALIZATION_HOST}:{VISUALIZATION_PORT}'

# =============================================================================
# SERVICE PORT MAPPING - каждый сервис знает свой порт
# =============================================================================
SERVICE_PORTS = {
    'storage': 8003,
    'collector': 8001,
    'mlservice': 8002,
    'visualization': 8004,
    'webmaster': 8000,
}

def get_service_port(service_name: str) -> int:
    """Get port for a specific service. Priority: ENV > SERVICE_PORTS > default"""
    # 1. Сначала проверяем SERVICE_PORT из environment (docker-compose)
    env_port = os.environ.get('SERVICE_PORT')
    if env_port:
        return int(env_port)
    
    # 2. Затем ищем по имени сервиса
    if service_name in SERVICE_PORTS:
        return SERVICE_PORTS[service_name]
    
    # 3. Default
    return 8000

def get_service_host() -> str:
    """Get host to bind to. In Docker always 0.0.0.0"""
    host = os.environ.get('SERVICE_HOST', '0.0.0.0')
    # Если host - это имя сервиса (storage, collector, etc), слушаем на всех интерфейсах
    if host in SERVICE_PORTS:
        return '0.0.0.0'
    return host

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
STATIC_COLS = [
    'credit_score', 'MI_pct', 'unit_cnt', 'occ_status', 'ocltv', 'dti',
    'orig_upb', 'oltv', 'orig_ir', 'channel', 'property_type', 'zip3',
    'loan_purpose', 'orig_term', 'cnt_borr', 'seller', 'first_flag',
    'MSA', 'first_time_hb_flag', 'MI_cancel_flag', 'int_only_flag', 
    'property_val_method', 'super_conf_flag', 'amortization_type'
]

CATEG_COLS = [
    'occ_status', 'channel', 'property_type', 'loan_purpose',
    'seller', 'first_flag', 'first_time_hb_flag', 'MI_cancel_flag',
    'int_only_flag', 'property_val_method', 'super_conf_flag', 
    'amortization_type'
]

#TODO нормальные названия

EVENTS = {
    0: "in process",
    1: "prepayment", 
    2: "default",
    3: "paid_off",
    4: "foreclosure",
    5: "short_sale",
    6: "modification"
}

TIME_GRID_SIZE = 100

class Config:
    """Configuration manager that loads from YAML files with fallbacks"""
    
    def __init__(self, service_name: str = None):
        self.service_name = service_name
        self.config = {}
        self._load_config()
        print(service_name)
    
    def _load_config(self):
        """Load configuration from YAML file if exists"""
        config_paths = [
            f'/app/config/{self.service_name}.yaml' if self.service_name else None,
            f'/app/config.yaml',
            f'/app/{self.service_name}_config.yaml' if self.service_name else None,
            'config.yaml',
            f'{self.service_name}_config.yaml' if self.service_name else None,
        ]
        
        for path in config_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        loaded = yaml.safe_load(f)
                        if loaded:
                            self.config.update(loaded)
                except Exception as e:
                    pass
        
        # Set defaults based on service name
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values based on service name"""
        # Определяем порт для текущего сервиса
        service_port = get_service_port(self.service_name) if self.service_name else 8000
        
        defaults = {
            'server': {
                'host': '0.0.0.0',
                'port': service_port,  # Порт зависит от сервиса!
                'debug': False
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'storage': {
                'base_dir': '/app/storage',
                'models_dir': '/app/storage/models',
                'batches_dir': '/app/storage/batches',
                'results_dir': '/app/storage/results',
                'metrics_dir': '/app/storage/metrics',
                'max_file_size_mb': 1000
            },
            'batch': {
                'default_size': 1000,
                'max_size': 50000,
                'default_strategy': 'random',
                'strategies': ['balanced', 'imbalanced', 'random']
            },
            'dependencies': {
                'timeout_seconds': 60
            },
            'statistics': {
                'time_bins': 20
            }
        }
        
        # Merge defaults with loaded config
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                for k, v in value.items():
                    if k not in self.config[key]:
                        self.config[key][k] = v
    
    def get(self, key: str, default=None):
        """Get config value using dot notation (e.g., 'server.port')"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


# =============================================================================
# LOGGER SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)