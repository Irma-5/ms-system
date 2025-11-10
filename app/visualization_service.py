from flask import Flask, request, jsonify, send_file
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from config import *

app = Flask(__name__)

sns.set_style("whitegrid")


def plot_event_distribution(y, title="Event Distribution over Time"):

    fig, ax = plt.subplots(figsize=(9, 6))
    y_df = pd.DataFrame(y)
    events = sorted(y_df['event'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(events)))
    sns.kdeplot(data=y_df, x="duration", hue="event", bw_adjust=0.25, palette="tab10")
    
    ax.set_xlabel('Duration', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_event_histogram(y, bins=20, title="Event Distribution Histogram"):

    fig, ax = plt.subplots(figsize=(9, 6))
    y_df = pd.DataFrame(y)
    events = sorted(y_df['event'].unique())
    duration_min = y_df['duration'].min()
    duration_max = y_df['duration'].max()
    bin_edges = np.linspace(duration_min, duration_max, bins + 1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(events)))
    
    for idx, event in enumerate(events):
        event_data = y_df[y_df['event'] == event]['duration']
        ax.hist(event_data, bins=bin_edges, alpha=0.5, 
                   label=f'Event {event}', color=colors[idx])
    
    ax.set_xlabel('Duration', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_survival_curves(predictions, TIME_GRID, num=5, title="Survival Curves by Event"):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for event_idx in range(num):
        curve = predictions[event_idx]       
        ax.plot(TIME_GRID, curve, label=f'Event {event_idx + 1}')
        
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    return fig


def plot_metrics_comparison(metrics_dict, title="Metrics Comparison"):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if 'ibs' in metrics_dict and metrics_dict['ibs']:
        events = list(metrics_dict['ibs'].keys())
        values = list(metrics_dict['ibs'].values())
        
        axes[0].line(events, values, color='skyblue', alpha=0.7)
        axes[0].set_xlabel('Event', fontsize=12)
        axes[0].set_ylabel('IBS_rm', fontsize=12)
        axes[0].set_title('IBS_rm by event', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        if 'ibs_mean' in metrics_dict:
            axes[0].axhline(y=metrics_dict['ibs_mean'], color='red', 
                          linestyle='--', label=f"Mean: {metrics_dict['ibs_mean']:.4f}")
            axes[0].legend()
    
    if 'auprc' in metrics_dict and metrics_dict['auprc']:
        events = list(metrics_dict['auprc'].keys())
        values = list(metrics_dict['auprc'].values())
        
        axes[1].line(events, values, color='lightgreen', alpha=0.7)
        axes[1].set_xlabel('Event', fontsize=12)
        axes[1].set_ylabel('AUPRC Score', fontsize=12)
        axes[1].set_title('AUPRC by event', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        if 'auprc_mean' in metrics_dict:
            axes[1].axhline(y=metrics_dict['auprc_mean'], color='red', 
                          linestyle='--', label=f"Mean: {metrics_dict['auprc_mean']:.4f}")
            axes[1].legend()
    
    plt.tight_layout()
    return fig


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


@app.route('/plot/batch_distribution', methods=['POST'])
def plot_batch_distribution():
    """
    
    JSON:
    {
        "batch_filename": "batch__strategy=balanced.pkl",
        "plot_type": "kde" | "histogram",
        "bins": 20 (for histogram),
        "title": "Custom title" (optional)
    }
    """
    try:
        data = request.json
        batch_filename = data.get('batch_filename')
        plot_type = data.get('plot_type', 'kde')
        bins = data.get('bins', 20)
        title = data.get('title')
        
        response = requests.get(
            f'http://{STORAGE_HOST}:{STORAGE_PORT}/batches/load/{batch_filename}'
        )
        
        if response.status_code != 200:
            return jsonify({'error': 'Batch not found'}), 404
        
        batch_data = response.json()
        y = batch_data['y']
        
        if not title:
            title = f"Event Distribution - {batch_filename}"
        
        if plot_type == 'kde':
            fig = plot_event_distribution(y, title=title)
        elif plot_type == 'histogram':
            fig = plot_event_histogram(y, bins=bins, title=title)
        else:
            return jsonify({'error': 'Invalid plot_type. Use "kde" or "histogram"'}), 400
        
        img_base64 = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'metadata': batch_data['metadata']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/plot/survival_curves', methods=['POST'])
def plot_survival():
    """
    JSON:
    {
        "predictions_filename": "results__model=cox__batch=b1.pkl",
        "title": "Custom title" (optional)
    }
    """
    try:
        data = request.json
        predictions_filename = data.get('predictions_filename')
        title = data.get('title')
        
        response = requests.get(
            f'http://{STORAGE_HOST}:{STORAGE_PORT}/results/load/{predictions_filename}'
        )
        
        if response.status_code != 200:
            return jsonify({'error': 'Predictions not found'}), 404
        
        pred_data = response.json()
        predictions = np.array(pred_data['predictions'])
        TIME_GRID = np.array(pred_data['metadata']['TIME_GRID'])
        
        if not title:
            title = f"Survival Curves - {predictions_filename}"
        
        fig = plot_survival_curves(predictions, TIME_GRID, title=title)
        img_base64 = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'metadata': pred_data['metadata']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/plot/metrics', methods=['POST'])
def plot_metrics():
    """
    JSON:
    {
        "metrics_filename": "metrics__model=cox__batch=b1.json",
        "title": "Custom title" (optional)
    }
    """
    try:
        data = request.json
        metrics_filename = data.get('metrics_filename')
        title = data.get('title')
        response = requests.get(
            f'http://{STORAGE_HOST}:{STORAGE_PORT}/metrics/load/{metrics_filename}'
        )
        
        if response.status_code != 200:
            return jsonify({'error': 'Metrics not found'}), 404
        metrics_data = response.json()
        metrics = metrics_data['metrics']
        
        if not title:
            title = f"Metrics Comparison - {metrics_filename}"
        
        fig = plot_metrics_comparison(metrics, title=title)
        
        img_base64 = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'metadata': metrics_data['metadata']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/plot/custom', methods=['POST'])
def plot_custom():
    """
    JSON:
    {
        "data": {...},
        "plot_config": {
            "type": "line" | "bar" | "scatter",
            "x": [...],
            "y": [...],
            "title": "...",
            "xlabel": "...",
            "ylabel": "..."
        }
    }
    """
    try:
        data = request.json
        plot_config = data.get('plot_config', {})
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        plot_type = plot_config.get('type', 'line')
        x = np.array(plot_config.get('x', []))
        y = np.array(plot_config.get('y', []))
        
        if plot_type == 'line':
            ax.plot(x, y, linewidth=2)
        elif plot_type == 'bar':
            ax.bar(range(len(y)), y)
        elif plot_type == 'scatter':
            ax.scatter(x, y, alpha=0.6)
        
        ax.set_title(plot_config.get('title', 'Custom Plot'), fontsize=14)
        ax.set_xlabel(plot_config.get('xlabel', 'X'), fontsize=12)
        ax.set_ylabel(plot_config.get('ylabel', 'Y'), fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Convert to base64
        img_base64 = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'image': img_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'service': 'Visualization',
        'status': 'healthy' # TODO придумать что тут 
    })


if __name__ == '__main__':
    app.run(host=VISUALIZATION_HOST, port=VISUALIZATION_PORT, debug=True)
