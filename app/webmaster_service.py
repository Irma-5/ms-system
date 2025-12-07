from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
from config import *

app = Flask(__name__)
CORS(app)



def call_service(service_url, endpoint, method='GET', data=None):
    """Helper function to call other services"""
    url = f'http://{service_url}{endpoint}'
    try:
        if method == 'GET':
            response = requests.get(url)
        elif method == 'POST':
            response = requests.post(url, json=data)
        elif method == 'DELETE':
            response = requests.delete(url)
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
            <button class="button" onclick="loadBatches()">Select Batch</button>
            <div id="statsResult"></div>
        </div>

        <!-- Predictions -->
        <div class="section">
            <h2>Make Predictions</h2>
            <div class="info-box">
                <p>Select model and batch from lists above, then:</p>
                <button class="button" onclick="makePredictions()">Run Predictions</button>
            </div>
            <div id="predictionsResult" class="result"></div>
        </div>

        <!-- Metrics Computation -->
        <div class="section">
            <h2>Compute Metrics</h2>
            <button class="button" onclick="computeMetrics()">Calculate IBS & AUPRC</button>
            <div id="metricsResult" class="result"></div>
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
            let selectedTrainBatch = null;

            async function checkHealth() {
                const services = [
                    {name: 'Storage', port: 8003},
                    {name: 'Collector', port: 8001},
                    {name: 'MLService', port: 8002},
                    {name: 'Visualization', port: 8004}
                ];
                
                let html = '<ul>';
                for (let service of services) {
                    try {
                        const response = await fetch(`http://localhost:${service.port}/health`);
                        const data = await response.json();
                        html += `<li>${service.name}: ${data.status}</li>`;
                    } catch (error) {
                        html += `<li>${service.name}: offline</li>`;
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
                    
                    let html = '<select id="batchSelect" onchange="selectBatch(this.value)" style="width: 100%; padding: 10px;">';
                    html += '<option value="">-- Select a batch --</option>';
                    
                    data.batches.forEach(batch => {
                        const name = batch.metadata.filename;
                        html += `<option value="${name}">${name} (${batch.metadata.size_mb} MB)</option>`;
                    });
                    html += '</select>';
                    
                    document.getElementById('batchesList').innerHTML = html;
                } catch (error) {
                    console.error('Error loading batches:', error);
                }
            }

            function selectBatch(filename) {
                selectedBatch = filename;
            }

            async function createBatch() {
                // This is simplified - you need to provide actual data
                alert('Note: Batch creation requires data. This is a demo interface.');
            }

            async function makePredictions() {
                if (!selectedModel || !selectedBatch) {
                    alert('Please select both a model and a batch first!');
                    return;
                }

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
                            `<strong>Predictions completed!</strong><br>
                             Saved as: ${data.saved_filename}<br>
                             Samples: ${data.metadata.n_samples}`;
                        document.getElementById('predictionsResult').style.display = 'block';
                    }
                } catch (error) {
                    console.error('Error making predictions:', error);
                }
            }

            async function computeMetrics() {
                if (!selectedResults || !selectedBatch) {
                    alert('Please run predictions first!');
                    return;
                }

                // Prompt for train batch
                selectedTrainBatch = prompt('Enter train batch filename:');
                if (!selectedTrainBatch) return;

                try {
                    const response = await fetch('/api/metrics/compute', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            predictions_filename: selectedResults,
                            batch_filename: selectedBatch,
                            train_batch_filename: selectedTrainBatch,
                            save_metrics: true
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        let html = '<strong>Metrics computed!</strong><br>';
                        html += `<strong>IBS Mean:</strong> ${data.metrics.ibs_mean?.toFixed(4)}<br>`;
                        html += `<strong>AUPRC Mean:</strong> ${data.metrics.auprc_mean?.toFixed(4)}<br>`;
                        html += `<strong>Saved as:</strong> ${data.saved_filename}`;
                        
                        document.getElementById('metricsResult').innerHTML = html;
                        document.getElementById('metricsResult').style.display = 'block';
                    }
                } catch (error) {
                    console.error('Error computing metrics:', error);
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
                    }
                } catch (error) {
                    console.error('Error plotting distribution:', error);
                }
            }

            async function plotSurvivalCurves() {
                if (!selectedResults) {
                    alert('Please run predictions first!');
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
                    }
                } catch (error) {
                    console.error('Error plotting survival curves:', error);
                }
            }

            async function plotMetrics() {
                alert('Select metrics file to visualize');
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



@app.route('/api/models/list', methods=['GET'])
def api_list_models():
    """List all available models"""
    result, status = call_service(
        f'{MLSERVICE_HOST}:{MLSERVICE_PORT}',
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

    print(f"WebMaster: http://localhost:{WEBMASTER_PORT}")
    
    app.run(host=WEBMASTER_HOST, port=WEBMASTER_PORT, debug=True)
