from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
from config import *

app = Flask(__name__)
config = Config('webmaster')
logging.basicConfig(
    level=getattr(logging, config.get('logging.level', 'INFO')),
    format=config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
CORS(app)

def call_service(service_url, endpoint, method='GET', data=None, timeout=300):
    url = f'http://{service_url}{endpoint}'
    try:
        if method == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        elif method == 'DELETE':
            response = requests.delete(url, timeout=timeout)
        else:
            return {'error': 'Invalid method'}, 400
        
        return response.json(), response.status_code
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Survival Analysis System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .section {
                background-color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .button {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
            }
            .button:hover {
                background-color: #2980b9;
            }
            .info-box {
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }
            select, input {
                padding: 8px;
                margin: 5px;
                border-radius: 3px;
                border: 1px solid #ddd;
            }
            h2 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .result {
                margin-top: 15px;
                padding: 10px;
                background-color: #d5f4e6;
                border-radius: 5px;
                display: none;
            }
            .error {
                background-color: #f8d7da;
                color: #721c24;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                display: none;
            }
            img {
                max-width: 100%;
                border-radius: 5px;
                margin-top: 10px;
            }
            .status-online {
                color: #27ae60;
                font-weight: bold;
            }
            .status-offline {
                color: #e74c3c;
                font-weight: bold;
            }
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .loading-message {
                display: flex;
                align-items: center;
                padding: 15px;
                background-color: #fff3cd;
                border-radius: 5px;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Survival Analysis System</h1>
            <p>Microservices architecture for competing risks analysis</p>
        </div>

        <!-- System Status -->
        <div class="section">
            <h2>System Status</h2>
            <button class="button" onclick="checkHealth()">Check Services Health</button>
            <div id="healthResult" class="result"></div>
        </div>

        <!-- Model Selection -->
        <div class="section">
            <h2>Model Selection</h2>
            <button class="button" onclick="loadModels()">Load Available Models</button>
            <div id="modelsList"></div>
            <div class="info-box" style="display:none;" id="selectedModelInfo">
                <strong>Selected Model:</strong> <span id="selectedModelName"></span>
            </div>
        </div>

        <!-- Batch Creation -->
        <div class="section">
            <h2>Batch Creation</h2>
            <div class="info-box">
                <label>Batch Size: <input type="number" id="batchSize" value="1000"></label><br>
                <label>Strategy: 
                    <select id="strategy">
                        <option value="balanced">Balanced</option>
                        <option value="imbalanced">Imbalanced</option>
                        <option value="random">Random</option>
                    </select>
                </label><br>
                <label>Stratify on Event: <input type="checkbox" id="stratify" checked></label><br>
                <button class="button" onclick="createBatch()">Create Batch</button>
            </div>
            <div id="batchResult" class="result"></div>
            
            <h3>Available Batches</h3>
            <button class="button" onclick="loadBatches()">Refresh Batch List</button>
            <div id="batchesList"></div>
        </div>

        <!-- Batch Statistics -->
        <div class="section">
            <h2>Batch Statistics</h2>
            <button class="button" onclick="showBatchStats()">Show Statistics</button>
            <div id="statsResult" class="result"></div>
        </div>

        <!-- Predictions -->
        <div class="section">
            <h2>Predictions</h2>
            <div class="info-box">
                <p>Select model and batch from lists above, then:</p>
                <button class="button" onclick="makePredictions()">Run Predictions</button>
            </div>
            <div id="predictionsResult" class="result"></div>
            <h3>Available Results</h3>
            <button class="button" onclick="loadResults()">Refresh Results List</button>
            <div id="resultsList"></div>
        </div>

        <!-- Metrics Computation -->
        <div class="section">
            <h2>Compute Metrics</h2>
            <button class="button" onclick="computeMetrics()">Calculate IBS & AUPRC</button>
            <div id="metricsResult" class="result"></div>
            <h3>Available Metrics</h3>
            <button class="button" onclick="loadMetrics()">Refresh Metrics List</button>
            <div id="metricsList"></div>
        </div>

        <!-- Visualization -->
        <div class="section">
            <h2>Visualizations</h2>
            <button class="button" onclick="plotBatchDistribution()">Plot Event Distribution</button>
            <button class="button" onclick="plotSurvivalCurves()">Plot Survival Curves</button>
            <button class="button" onclick="plotMetrics()">Plot Metrics</button>
            <div id="visualizationResult"></div>
        </div>

        <script>
            let selectedModel = null;
            let selectedBatch = null;
            let selectedResults = null;
            let selectedMetrics = null;

            async function checkHealth() {
                const services = [
                    {name: 'Storage', endpoint: '/api/health/storage'},
                    {name: 'Collector', endpoint: '/api/health/collector'},
                    {name: 'MLService', endpoint: '/api/health/mlservice'},
                    {name: 'Visualization', endpoint: '/api/health/visualization'}
                ];
                
                let html = '<ul>';
                for (let service of services) {
                    try {
                        const response = await fetch(service.endpoint);
                        const data = await response.json();
                        const status = data.status === 'healthy' ? 'online' : 'offline';
                        const statusClass = status === 'online' ? 'status-online' : 'status-offline';
                        html += `<li>${service.name}: <span class="${statusClass}">${status}</span></li>`;
                    } catch (error) {
                        console.error(`Error fetching health for ${service.name}:`, error);
                        html += `<li>${service.name}: <span class="status-offline">offline</span></li>`;
                    }
                }
                html += '</ul>';
                
                document.getElementById('healthResult').innerHTML = html;
                document.getElementById('healthResult').style.display = 'block';
            }

            async function loadModels() {
                try {
                    const response = await fetch('/api/models/list');
                    const data = await response.json();
                    
                    if (!data.models || data.models.length === 0) {
                        document.getElementById('modelsList').innerHTML = 
                            '<p style="color: orange;">No models found. Upload models first.</p>';
                        return;
                    }
                    
                    let html = '<select id="modelSelect" onchange="selectModel(this.value)" style="width: 100%; padding: 10px;">';
                    html += '<option value="">-- Select a model --</option>';
                    
                    data.models.forEach(model => {
                        const name = model.metadata.filename;
                        html += `<option value="${name}">${name}</option>`;
                    });
                    html += '</select>';
                    
                    document.getElementById('modelsList').innerHTML = html;
                } catch (error) {
                    console.error('Error loading models:', error);
                    document.getElementById('modelsList').innerHTML = 
                        '<p style="color: red;">Error loading models. Check console.</p>';
                }
            }

            function selectModel(filename) {
                selectedModel = filename;
                document.getElementById('selectedModelName').textContent = filename;
                document.getElementById('selectedModelInfo').style.display = 'block';
            }

            async function loadBatches() {
                try {
                    const response = await fetch('/api/batches/list');
                    const data = await response.json();
                    
                    const filteredBatches = data.batches ? data.batches.filter(batch => {
                        const baseName = batch.base_name;
                        return baseName !== 'batch_0.pkl' && baseName !== 'batch_train.pkl';
                    }) : [];
                    
                    if (filteredBatches.length === 0) {
                        document.getElementById('batchesList').innerHTML = 
                            '<p style="color: orange;">No batches found. Create a batch first.</p>';
                        return;
                    }
                    
                    let html = '<select id="batchSelect" onchange="selectBatch(this.value)" style="width: 100%; padding: 10px;">';
                    html += '<option value="">-- Select a batch --</option>';
                    
                    filteredBatches.forEach(batch => {
                        const name = batch.metadata.filename;
                        html += `<option value="${name}">${name} (${batch.metadata.size_mb} MB)</option>`;
                    });
                    html += '</select>';
                    
                    document.getElementById('batchesList').innerHTML = html;
                } catch (error) {
                    console.error('Error loading batches:', error);
                    document.getElementById('batchesList').innerHTML = 
                        '<p style="color: red;">Error loading batches. Check console.</p>';
                }
            }

            function selectBatch(filename) {
                selectedBatch = filename;
            }

            async function showBatchStats() {
                if (!selectedBatch) {
                    alert('Please select a batch first!');
                    return;
                }
                
                try {
                    const response = await fetch(`/api/batch/statistics/${selectedBatch}`);
                    const data = await response.json();
                    
                    if (data.success) {
                        let html = '<strong>Batch Statistics:</strong><br>';
                        html += `<strong>Total samples:</strong> ${data.statistics.total_samples}<br>`;
                        html += '<strong>Event counts:</strong><br>';
                        
                        for (const [event, count] of Object.entries(data.statistics.event_counts)) {
                            html += `&nbsp;&nbsp;Event ${event}: ${count}<br>`;
                        }
                        
                        if (data.statistics.per_event_duration_stats) {
                            html += '<br><strong>Duration stats per event:</strong><br>';
                            for (const [event, stats] of Object.entries(data.statistics.per_event_duration_stats)) {
                                html += `&nbsp;&nbsp;Event ${event}: mean=${stats.mean.toFixed(1)}, median=${stats.median.toFixed(1)}, std=${stats.std.toFixed(1)}<br>`;
                            }
                        }
                        
                        document.getElementById('statsResult').innerHTML = html;
                        document.getElementById('statsResult').style.display = 'block';
                    } else {
                        document.getElementById('statsResult').innerHTML = 
                            `<span style="color: red;">Error: ${data.error || 'Unknown error'}</span>`;
                        document.getElementById('statsResult').style.display = 'block';
                    } 
                } catch (error) {
                    console.error('Error getting statistics:', error);
                    document.getElementById('statsResult').innerHTML = 
                        `<span style="color: red;">Error: ${error.message}</span>`;
                    document.getElementById('statsResult').style.display = 'block';
                }
            }

            async function createBatch() {
                const batchSize = document.getElementById('batchSize').value;
                const strategy = document.getElementById('strategy').value;
                const stratify = document.getElementById('stratify').checked;
                
                document.getElementById('batchResult').innerHTML = 
                    `<div class="loading-message">
                        <div class="spinner"></div>
                        <span>Creating batch... this may take 1-2 minutes</span>
                    </div>`;
                document.getElementById('batchResult').style.display = 'block';
                
                try {
                    const response = await fetch('/api/batch/create_from_storage', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            source_batch: 'batch_0',
                            batch_size: parseInt(batchSize),
                            strategy: strategy,
                            stratify_on_event: stratify
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        document.getElementById('batchResult').innerHTML = 
                            `<strong>✓ Batch created!</strong><br>
                            Filename: ${data.filename}<br>
                            Size: ${data.event_distribution ? Object.values(data.event_distribution).reduce((a,b) => a+b, 0) : 'N/A'}`;
                        document.getElementById('batchResult').style.display = 'block';
                        loadBatches();
                    } else {
                        document.getElementById('batchResult').innerHTML = 
                            `<span style="color: red;">Error: ${data.error || 'Unknown error'}</span>`;
                        document.getElementById('batchResult').style.display = 'block';
                    }
                } catch (error) {
                    console.error('Error creating batch:', error);
                    document.getElementById('batchResult').innerHTML = 
                        `<span style="color: red;">Error: ${error.message}</span>`;
                    document.getElementById('batchResult').style.display = 'block';
                }
            }

            async function makePredictions() {
                if (!selectedModel || !selectedBatch) {
                    alert('Please select both a model and a batch first!');
                    return;
                }

                document.getElementById('predictionsResult').innerHTML = 
                    `<div class="loading-message">
                        <div class="spinner"></div>
                        <span>Running predictions... </span>
                    </div>`;
                document.getElementById('predictionsResult').style.display = 'block';
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            model_filename: selectedModel,
                            batch_filename: selectedBatch,
                            save_results: true
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        selectedResults = data.saved_filename;
                        document.getElementById('predictionsResult').innerHTML = 
                            `<strong>✓ Predictions completed!</strong><br>
                             Saved as: ${data.saved_filename}<br>
                             Samples: ${data.metadata?.n_samples || 'N/A'}`;
                        document.getElementById('predictionsResult').style.display = 'block';
                        loadResults();
                    } else {
                        document.getElementById('predictionsResult').innerHTML = 
                            `<span style="color: red;">Error: ${data.error || 'Unknown error'}</span>`;
                        document.getElementById('predictionsResult').style.display = 'block';
                    }
                } catch (error) {
                    console.error('Error making predictions:', error);
                    document.getElementById('predictionsResult').innerHTML = 
                        `<span style="color: red;">Error: ${error.message}</span>`;
                    document.getElementById('predictionsResult').style.display = 'block';
                }
            }

            async function loadResults() {
                try {
                    const response = await fetch('/api/results/list');
                    const data = await response.json();
                    
                    if (!data.results || data.results.length === 0) {
                        document.getElementById('resultsList').innerHTML = 
                            '<p style="color: orange;">No results found. Run predictions first.</p>';
                        return;
                    }
                    
                    let html = '<select id="resultsSelect" onchange="selectResults(this.value)" style="width: 100%; padding: 10px;">';
                    html += '<option value="">-- Select results --</option>';
                    
                    data.results.forEach(result => {
                        const name = result.metadata.filename;
                        html += `<option value="${name}">${name}</option>`;
                    });
                    html += '</select>';
                    
                    document.getElementById('resultsList').innerHTML = html;
                } catch (error) {
                    console.error('Error loading results:', error);
                }
            }

            function selectResults(filename) {
                selectedResults = filename;
            }

            async function loadMetrics() {
                try {
                    const response = await fetch('/api/metrics/list');
                    const data = await response.json();
                    
                    if (!data.metrics || data.metrics.length === 0) {
                        document.getElementById('metricsList').innerHTML = 
                            '<p style="color: orange;">No metrics found. Compute metrics first.</p>';
                        return;
                    }
                    
                    let html = '<select id="metricsSelect" onchange="selectMetrics(this.value)" style="width: 100%; padding: 10px;">';
                    html += '<option value="">-- Select metrics --</option>';
                    
                    data.metrics.forEach(m => {
                        const name = m.metadata.filename;
                        html += `<option value="${name}">${name}</option>`;
                    });
                    html += '</select>';
                    
                    document.getElementById('metricsList').innerHTML = html;
                } catch (error) {
                    console.error('Error loading metrics:', error);
                }
            }

            function selectMetrics(filename) {
                selectedMetrics = filename;
            }

            async function computeMetrics() {
                console.log('selectedResults:', selectedResults);
                
                if (!selectedResults) {
                    alert('Please select a result first!');
                    return;
                }
                
                // Извлекаем batch_id из названия результата
                let batchId = null;
                const parts = selectedResults.split('__');
                for (const part of parts) {
                    if (part.startsWith('batch=')) {
                        batchId = part.split('=')[1].replace('.pkl', '');
                        break;
                    }
                }
                
                if (!batchId) {
                    alert('Could not extract batch ID from result filename');
                    return;
                }
                
                // Показываем индикатор загрузки
                document.getElementById('metricsResult').innerHTML = 
                    `<div class="loading-message">
                        <div class="spinner"></div>
                        <span>Finding batch and computing metrics...</span>
                    </div>`;
                document.getElementById('metricsResult').style.display = 'block';
                
                // Находим файл батча по ID
                const batchResponse = await fetch('/api/batches/list');
                const batchData = await batchResponse.json();
                let batchFilename = null;
                
                if (batchData.batches) {
                    for (const batch of batchData.batches) {
                        const filename = batch.metadata.filename;
                        if (filename.includes(`id=${batchId}__`)) {
                            batchFilename = filename;
                            break;
                        }
                    }
                }
                
                if (!batchFilename) {
                    document.getElementById('metricsResult').innerHTML = 
                        `<span style="color: red;">Error: Batch with ID ${batchId} not found</span>`;
                    return;
                }
                
                try {
                    const response = await fetch('/api/metrics/compute', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            predictions_filename: selectedResults,
                            batch_filename: batchFilename,
                            train_batch_filename: 'batch_train.pkl',
                            save_metrics: true
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        let html = '<strong>✓ Metrics computed!</strong><br>';
                        html += `<strong>IBS Mean:</strong> ${data.metrics.ibs_mean?.toFixed(4) || 'N/A'}<br>`;
                        html += `<strong>AUPRC Mean:</strong> ${data.metrics.auprc_mean?.toFixed(4) || 'N/A'}<br>`;
                        html += `<strong>Saved as:</strong> ${data.saved_filename || 'N/A'}`;
                        
                        document.getElementById('metricsResult').innerHTML = html;
                        document.getElementById('metricsResult').style.display = 'block';
                        loadMetrics();
                    } else {
                        document.getElementById('metricsResult').innerHTML = 
                            `<span style="color: red;">Error: ${data.error || 'Unknown error'}</span>`;
                    }
                } catch (error) {
                    console.error('Error computing metrics:', error);
                    document.getElementById('metricsResult').innerHTML = 
                        `<span style="color: red;">Error: ${error.message}</span>`;
                }
            }

            async function plotBatchDistribution() {
                if (!selectedBatch) {
                    alert('Please select a batch first!');
                    return;
                }
                
                try {
                    const response = await fetch('/api/visualize/batch_distribution', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            batch_filename: selectedBatch,
                            plot_type: 'kde'
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        document.getElementById('visualizationResult').innerHTML = 
                            `<img src="data:image/png;base64,${data.image}">`;
                    } else {
                        document.getElementById('visualizationResult').innerHTML = 
                            `<span style="color: red;">Error: ${data.error || 'Unknown error'}</span>`;
                    }
                } catch (error) {
                    console.error('Error plotting distribution:', error);
                    document.getElementById('visualizationResult').innerHTML = 
                        `<span style="color: red;">Error: ${error.message}</span>`;
                }
            }

            async function plotSurvivalCurves() {
                if (!selectedResults) {
                    alert('Please select a result first!');
                    return;
                }
                
                try {
                    const response = await fetch('/api/visualize/survival_curves', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            predictions_filename: selectedResults
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        document.getElementById('visualizationResult').innerHTML = 
                            `<img src="data:image/png;base64,${data.image}">`;
                    } else {
                        document.getElementById('visualizationResult').innerHTML = 
                            `<span style="color: red;">Error: ${data.error || 'Unknown error'}</span>`;
                    }
                } catch (error) {
                    console.error('Error plotting survival curves:', error);
                    document.getElementById('visualizationResult').innerHTML = 
                        `<span style="color: red;">Error: ${error.message}</span>`;
                }
            }

            async function plotMetrics() {
                if (!selectedMetrics) {
                    alert('Please select a metrics file first!');
                    return;
                }
                
                try {
                    const response = await fetch('/api/visualize/metrics', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            metrics_filename: selectedMetrics
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        document.getElementById('visualizationResult').innerHTML = 
                            `<img src="data:image/png;base64,${data.image}">`;
                    } else {
                        document.getElementById('visualizationResult').innerHTML = 
                            `<span style="color: red;">Error: ${data.error || 'Unknown error'}</span>`;
                    }
                } catch (error) {
                    console.error('Error plotting metrics:', error);
                    document.getElementById('visualizationResult').innerHTML = 
                        `<span style="color: red;">Error: ${error.message}</span>`;
                }
            }

            // Initialize on load
            window.onload = function() {
                checkHealth();
                loadModels();
                loadBatches();
            };
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

# API ENDPOINTS

@app.route('/api/models/list', methods=['GET'])
def api_list_models():
    result, status = call_service(
        f'{STORAGE_HOST}:{STORAGE_PORT}',
        '/models/list'
    )
    return jsonify(result), status

@app.route('/api/batches/list', methods=['GET'])
def api_list_batches():
    result, status = call_service(
        f'{STORAGE_HOST}:{STORAGE_PORT}',
        '/batches/list'
    )
    return jsonify(result), status

@app.route('/api/results/list', methods=['GET'])
def api_list_results():
    result, status = call_service(
        f'{STORAGE_HOST}:{STORAGE_PORT}',
        '/results/list'
    )
    return jsonify(result), status

@app.route('/api/metrics/list', methods=['GET'])
def api_list_metrics():
    result, status = call_service(
        f'{STORAGE_HOST}:{STORAGE_PORT}',
        '/metrics/list'
    )
    return jsonify(result), status

@app.route('/api/health/storage', methods=['GET'])
def api_health_storage():
    result, status = call_service(
        f'{STORAGE_HOST}:{STORAGE_PORT}',
        '/health'
    )
    return jsonify(result), status

@app.route('/api/health/collector', methods=['GET'])
def api_health_collector():
    result, status = call_service(
        f'{COLLECTOR_HOST}:{COLLECTOR_PORT}',
        '/health'
    )
    return jsonify(result), status

@app.route('/api/health/mlservice', methods=['GET'])
def api_health_mlservice():
    result, status = call_service(
        f'{MLSERVICE_HOST}:{MLSERVICE_PORT}',
        '/health'
    )
    return jsonify(result), status

@app.route('/api/health/visualization', methods=['GET'])
def api_health_visualization():
    result, status = call_service(
        f'{VISUALIZATION_HOST}:{VISUALIZATION_PORT}',
        '/health'
    )
    return jsonify(result), status

# OTHER API ENDPOINTS

@app.route('/api/batch/create', methods=['POST'])
def api_create_batch():
    data = request.json
    result, status = call_service(
        f'{COLLECTOR_HOST}:{COLLECTOR_PORT}',
        '/batch/create',
        method='POST',
        data=data
    )
    return jsonify(result), status

@app.route('/api/batch/statistics/<filename>', methods=['GET'])
def api_batch_statistics(filename):
    result, status = call_service(
        f'{COLLECTOR_HOST}:{COLLECTOR_PORT}',
        f'/batch/statistics/{filename}'
    )
    return jsonify(result), status

@app.route('/api/batch/create_from_storage', methods=['POST'])
def api_create_batch_from_storage():
    data = request.json
    result, status = call_service(
        f'{COLLECTOR_HOST}:{COLLECTOR_PORT}',
        '/batch/create_from_storage',
        method='POST',
        data=data
    )
    return jsonify(result), status

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    result, status = call_service(
        f'{MLSERVICE_HOST}:{MLSERVICE_PORT}',
        '/predict',
        method='POST',
        data=data
    )
    return jsonify(result), status

@app.route('/api/metrics/compute', methods=['POST'])
def api_compute_metrics():
    data = request.json
    result, status = call_service(
        f'{MLSERVICE_HOST}:{MLSERVICE_PORT}',
        '/metrics/compute',
        method='POST',
        data=data
    )
    return jsonify(result), status

@app.route('/api/visualize/batch_distribution', methods=['POST'])
def api_visualize_batch():
    data = request.json
    result, status = call_service(
        f'{VISUALIZATION_HOST}:{VISUALIZATION_PORT}',
        '/plot/batch_distribution',
        method='POST',
        data=data
    )
    return jsonify(result), status

@app.route('/api/visualize/survival_curves', methods=['POST'])
def api_visualize_survival():
    data = request.json
    result, status = call_service(
        f'{VISUALIZATION_HOST}:{VISUALIZATION_PORT}',
        '/plot/survival_curves',
        method='POST',
        data=data
    )
    return jsonify(result), status

@app.route('/api/visualize/metrics', methods=['POST'])
def api_visualize_metrics():
    data = request.json
    result, status = call_service(
        f'{VISUALIZATION_HOST}:{VISUALIZATION_PORT}',
        '/plot/metrics',
        method='POST',
        data=data
    )
    return jsonify(result), status

@app.route('/health', methods=['GET'])
def health():
    """Health check for WebMaster"""
    return jsonify({
        'service': 'WebMaster',
        'status': 'healthy'
    })

if __name__ == '__main__':
    host = os.environ.get('SERVICE_HOST', '0.0.0.0')
    port = int(os.environ.get('SERVICE_PORT', WEBMASTER_PORT))
    print(config.service_name)
    app.run(host=host, port=port, debug=True)