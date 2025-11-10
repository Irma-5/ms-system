"""
Example: Using the microservices system with OvR and MetaModel

This script demonstrates:
1. How to save your trained model
2. How to create batches
3. How to make predictions
4. How to compute metrics
5. How to visualize results
"""

import requests
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
import copy

# Assuming these are from your code
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier


# ============= YOUR MODEL CLASSES =============

class OvR:
    def __init__(self, estimator, mode, early_threshold=1.0):
        self.estimator = estimator
        self.mode = mode
        self.y_train = None
        self.model = None
        self.events = None
        self.TIME_GRID = None
        self.thrsh = early_threshold
    
    def fit(self, X_, y_):
        # Your fitting logic here
        pass
    
    def predict(self, X_):
        # Your prediction logic here
        pass


class MetaModel:
    def __init__(self, estimator, mode='weighted'):
        self.ovr = estimator
        self.meta_model = RandomForestClassifier(n_jobs=-1, random_state=42)
        self.mode = mode
        self.train = None
    
    def fit(self, X_, y_):
        # Your fitting logic here
        pass
    
    def predict(self, X_):
        # Your prediction logic here
        pass


# ============= CONFIGURATION =============

BASE_URL = 'http://localhost:8000/api'
STORAGE_URL = 'http://localhost:8003'


# ============= 1. SAVE TRAINED MODEL =============

def save_model_to_storage(model, model_name, metadata=None):
    """
    Save trained model to storage service
    
    Args:
        model: Your trained MetaModel instance
        model_name: Name for the model (e.g., "cox_metamodel")
        metadata: Additional metadata dict
    """
    if metadata is None:
        metadata = {}
    
    metadata['model_name'] = model_name
    
    # Pickle the model
    model_bytes = pickle.dumps(model)
    
    # Send to storage
    response = requests.post(
        f'{STORAGE_URL}/models/save',
        json={
            'model_data': str(model_bytes),
            'metadata': metadata
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Model saved: {result['filename']}")
        return result['filename']
    else:
        print(f"❌ Error saving model: {response.json()}")
        return None


# ============= 2. CREATE AND SAVE BATCHES =============

def create_and_save_batch(X, y, batch_name, strategy='balanced', batch_size=None):
    """
    Create a batch and save it to storage
    
    Args:
        X: Features dataframe/array
        y: DataFrame with 'event' and 'duration' columns
        batch_name: Name for the batch
        strategy: 'balanced', 'imbalanced', or 'random'
        batch_size: Size of batch (None = use all data)
    """
    if batch_size is None:
        batch_size = len(X)
    
    # Convert to lists for JSON
    X_list = X.tolist() if hasattr(X, 'tolist') else X
    y_dict = {
        'event': y['event'].tolist() if hasattr(y['event'], 'tolist') else list(y['event']),
        'duration': y['duration'].tolist() if hasattr(y['duration'], 'tolist') else list(y['duration'])
    }
    
    # Create batch
    response = requests.post(
        f'{BASE_URL}/batch/create',
        json={
            'X': X_list,
            'y': y_dict,
            'batch_size': batch_size,
            'strategy': strategy,
            'metadata': {
                'name': batch_name,
                'original_size': len(X)
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Batch created: {result['filename']}")
        print(f"   Event distribution: {result.get('event_distribution', {})}")
        return result['filename']
    else:
        print(f"❌ Error creating batch: {response.json()}")
        return None


# ============= 3. MAKE PREDICTIONS =============

def make_predictions(model_filename, batch_filename):
    """
    Make predictions using saved model and batch
    
    Args:
        model_filename: Name of saved model file
        batch_filename: Name of saved batch file
    
    Returns:
        results_filename: Name of saved results file
    """
    response = requests.post(
        f'{BASE_URL}/predict',
        json={
            'model_filename': model_filename,
            'batch_filename': batch_filename,
            'time_grid_size': 100,
            'save_results': True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Predictions completed: {result.get('saved_filename')}")
        return result.get('saved_filename')
    else:
        print(f"❌ Error making predictions: {response.json()}")
        return None


# ============= 4. COMPUTE METRICS =============

def compute_metrics(predictions_filename, test_batch_filename, train_batch_filename):
    """
    Compute IBS and AUPRC metrics
    
    Args:
        predictions_filename: Name of predictions file
        test_batch_filename: Name of test batch file
        train_batch_filename: Name of train batch file
    
    Returns:
        metrics: Dict with metrics
    """
    response = requests.post(
        f'{BASE_URL}/metrics/compute',
        json={
            'predictions_filename': predictions_filename,
            'batch_filename': test_batch_filename,
            'train_batch_filename': train_batch_filename,
            'save_metrics': True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Metrics computed:")
        print(f"   IBS mean: {result['metrics'].get('ibs_mean'):.4f}")
        print(f"   AUPRC mean: {result['metrics'].get('auprc_mean'):.4f}")
        print(f"   Saved as: {result.get('saved_filename')}")
        return result['metrics']
    else:
        print(f"❌ Error computing metrics: {response.json()}")
        return None


# ============= 5. VISUALIZATIONS =============

def plot_batch_distribution(batch_filename, plot_type='kde'):
    """
    Plot event distribution for a batch
    
    Args:
        batch_filename: Name of batch file
        plot_type: 'kde' or 'histogram'
    
    Returns:
        image_base64: Base64-encoded image
    """
    response = requests.post(
        f'{BASE_URL}/visualize/batch_distribution',
        json={
            'batch_filename': batch_filename,
            'plot_type': plot_type
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Plot created (base64 image available)")
        return result['image']
    else:
        print(f"❌ Error creating plot: {response.json()}")
        return None


def plot_survival_curves(predictions_filename):
    """
    Plot survival curves from predictions
    
    Args:
        predictions_filename: Name of predictions file
    
    Returns:
        image_base64: Base64-encoded image
    """
    response = requests.post(
        f'{BASE_URL}/visualize/survival_curves',
        json={
            'predictions_filename': predictions_filename
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Survival curves plot created")
        return result['image']
    else:
        print(f"❌ Error creating plot: {response.json()}")
        return None


def plot_metrics_comparison(metrics_filename):
    """
    Plot metrics comparison
    
    Args:
        metrics_filename: Name of metrics file
    
    Returns:
        image_base64: Base64-encoded image
    """
    response = requests.post(
        f'{BASE_URL}/visualize/metrics',
        json={
            'metrics_filename': metrics_filename
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Metrics plot created")
        return result['image']
    else:
        print(f"❌ Error creating plot: {response.json()}")
        return None


# ============= COMPLETE WORKFLOW EXAMPLE =============

def complete_workflow_example():
    """
    Complete example workflow from training to visualization
    """
    print("=" * 60)
    print("COMPLETE WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Step 0: Load your data (from your code)
    # y_large, x_large, _, _, _ = bal280_sample()
    # sc = Scaler()
    # sc.fit([x_large])
    # x_large = sc.transform(x_large)
    # y_large = pd.DataFrame(y_large)
    # y_large, dct = transform_events(y_large)
    
    # For this example, let's assume you have:
    # x_train, x_test, y_train, y_test (already split and preprocessed)
    
    print("\n📊 Assuming you have preprocessed data...")
    print("   x_train, x_test, y_train, y_test")
    
    # Step 1: Train your model (from your code)
    print("\n🤖 Step 1: Training model...")
    # early_threshold = 0.9
    # model = MetaModel(
    #     OvR(CoxPHFitter(penalizer=0.02), mode='single', early_threshold=early_threshold),
    #     mode='early'
    # )
    # model.fit(x_train, y_train)
    print("   (Assuming model is trained)")
    
    # Step 2: Save model
    print("\n💾 Step 2: Saving model to storage...")
    # model_filename = save_model_to_storage(
    #     model,
    #     model_name='cox_metamodel_early',
    #     metadata={
    #         'mode': 'early',
    #         'early_threshold': 0.9,
    #         'penalizer': 0.02
    #     }
    # )
    print("   (Skipping - need trained model)")
    model_filename = "model__example.pkl"
    
    # Step 3: Create batches
    print("\n📦 Step 3: Creating test batch...")
    # test_batch_filename = create_and_save_batch(
    #     x_test,
    #     y_test,
    #     batch_name='test_balanced',
    #     strategy='balanced',
    #     batch_size=1000
    # )
    print("   (Skipping - need data)")
    test_batch_filename = "batch__test.pkl"
    
    print("\n📦 Creating train batch (for metrics)...")
    # train_batch_filename = create_and_save_batch(
    #     x_train,
    #     y_train,
    #     batch_name='train',
    #     strategy='random'
    # )
    print("   (Skipping - need data)")
    train_batch_filename = "batch__train.pkl"
    
    # Step 4: Get batch statistics
    print("\n📈 Step 4: Getting batch statistics...")
    # response = requests.get(f'{BASE_URL}/batch/statistics/{test_batch_filename}')
    # if response.status_code == 200:
    #     stats = response.json()['statistics']
    #     print(f"   Total samples: {stats['total_samples']}")
    #     print(f"   Event counts: {stats['event_counts']}")
    print("   (Skipping - need batch)")
    
    # Step 5: Visualize batch distribution
    print("\n📊 Step 5: Visualizing batch distribution...")
    # plot_batch_distribution(test_batch_filename, plot_type='kde')
    print("   (Skipping - need batch)")
    
    # Step 6: Make predictions
    print("\n🎯 Step 6: Making predictions...")
    # predictions_filename = make_predictions(model_filename, test_batch_filename)
    print("   (Skipping - need model and batch)")
    predictions_filename = "results__example.pkl"
    
    # Step 7: Compute metrics
    print("\n📊 Step 7: Computing metrics...")
    # metrics = compute_metrics(predictions_filename, test_batch_filename, train_batch_filename)
    print("   (Skipping - need predictions)")
    
    # Step 8: Visualize survival curves
    print("\n📉 Step 8: Visualizing survival curves...")
    # plot_survival_curves(predictions_filename)
    print("   (Skipping - need predictions)")
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE!")
    print("=" * 60)
    print("\nTo use this example:")
    print("1. Make sure all services are running (./start_all.sh)")
    print("2. Load and preprocess your data")
    print("3. Train your model")
    print("4. Uncomment the function calls above")
    print("5. Run this script")


# ============= USAGE EXAMPLES =============

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Survival Analysis Microservices - Usage Example        ║
    ╚══════════════════════════════════════════════════════════╝
    
    This script shows how to use the microservices system.
    
    Before running:
    1. Start all services: ./start_all.sh
    2. Have your data ready (X, y with 'event' and 'duration')
    3. Train your model (OvR + MetaModel)
    
    Then you can:
    - Save models: save_model_to_storage(model, 'model_name')
    - Create batches: create_and_save_batch(X, y, 'batch_name')
    - Make predictions: make_predictions(model_file, batch_file)
    - Compute metrics: compute_metrics(pred_file, test_batch, train_batch)
    - Visualize: plot_batch_distribution(), plot_survival_curves()
    
    Or run the complete workflow:
    """)
    
    # Uncomment to run complete workflow (after preparing data)
    # complete_workflow_example()
    
    print("\n⚠️  Remember to:")
    print("   - Start all services first")
    print("   - Prepare your data")
    print("   - Train your model")
    print("   - Then uncomment and run the workflow!")
