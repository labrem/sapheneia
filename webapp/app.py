"""
Sapheneia TimesFM Web Application

A Flask-based web application for TimesFM forecasting with a professional interface.
Supports both localhost deployment and GCP Cloud Run deployment.

Features:
- File upload for CSV data
- Interactive parameter configuration
- Real-time forecasting with TimesFM
- Professional visualizations
- Downloadable results
- Support for covariates and quantile forecasting
"""

import os
import sys
import json
import io
import base64
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import Sapheneia TimesFM modules
from model import TimesFMModel
from data import DataProcessor
from forecast import Forecaster
from visualization import Visualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure directories exist
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global variables for model management
current_model = None
current_forecaster = None
current_visualizer = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_model(backend='cpu', context_len=100, horizon_len=24, checkpoint=None, local_path=None):
    """Initialize TimesFM model with given parameters."""
    global current_model, current_forecaster, current_visualizer
    
    try:
        logger.info(f"Initializing model with backend={backend}, context={context_len}, horizon={horizon_len}")
        
        model_wrapper = TimesFMModel(
            backend=backend,
            context_len=context_len,
            horizon_len=horizon_len,
            checkpoint=checkpoint,
            local_model_path=local_path
        )
        
        timesfm_model = model_wrapper.load_model()
        current_model = model_wrapper
        current_forecaster = Forecaster(timesfm_model)
        current_visualizer = Visualizer(style="professional")
        
        logger.info("Model initialized successfully")
        return True, "Model initialized successfully"
        
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        return False, f"Model initialization failed: {str(e)}"


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    num_periods = 150
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(weeks=i) for i in range(num_periods)]
    
    # Generate synthetic financial data
    btc_base = 25000
    trend = np.linspace(0, 0.6, num_periods)
    volatility = np.random.normal(0, 0.08, num_periods)
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(num_periods) / 52)
    btc_price = btc_base * np.exp(trend + volatility + seasonal)
    
    # Correlated assets
    eth_price = btc_price * 0.06 * (1 + np.random.normal(0, 0.05, num_periods))
    sp500_price = 3500 * np.exp(trend * 0.3 + np.random.normal(0, 0.02, num_periods))
    vix_price = np.clip(20 - 2 * np.diff(np.log(btc_price), prepend=0) * 10 + 
                       np.random.normal(0, 3, num_periods), 10, 80)
    
    # Categorical features
    quarters = [(pd.Timestamp(d).month - 1) // 3 + 1 for d in dates]
    market_regime = ['bull' if p > btc_base else 'bear' for p in btc_price]
    
    sample_df = pd.DataFrame({
        'date': dates,
        'btc_price': btc_price,
        'eth_price': eth_price,
        'sp500_price': sp500_price,
        'vix_index': vix_price,
        'quarter': quarters,
        'market_regime': market_regime,
        'asset_category': 'cryptocurrency',
        'base_volatility': 0.08
    })
    
    return sample_df


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/model/init', methods=['POST'])
def api_init_model():
    """Initialize TimesFM model via API."""
    try:
        data = request.get_json()
        
        backend = data.get('backend', 'cpu')
        context_len = int(data.get('context_len', 100))
        horizon_len = int(data.get('horizon_len', 24))
        checkpoint = data.get('checkpoint')
        local_path = data.get('local_path')
        
        # Use default checkpoint if none specified
        if not checkpoint and not local_path:
            checkpoint = "google/timesfm-2.0-500m-pytorch"
        
        success, message = init_model(backend, context_len, horizon_len, checkpoint, local_path)
        
        return jsonify({
            'success': success,
            'message': message,
            'model_info': current_model.get_model_info() if success else None
        })
        
    except Exception as e:
        logger.error(f"API model init error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/data/upload', methods=['POST'])
def api_upload_data():
    """Upload and process CSV data."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type. Only CSV files allowed.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and analyze data
        df = pd.read_csv(filepath)
        df_info = {
            'filename': filename,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'head': df.head().to_dict('records'),
            'null_counts': df.isnull().sum().to_dict()
        }
        
        # Check for date column
        has_date = 'date' in df.columns
        if has_date:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df_info['date_range'] = {
                'start': str(df['date'].min().date()) if not df['date'].isnull().all() else None,
                'end': str(df['date'].max().date()) if not df['date'].isnull().all() else None,
                'periods': len(df)
            }
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'data_info': df_info,
            'has_date_column': has_date
        })
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return jsonify({'success': False, 'message': f'Upload failed: {str(e)}'}), 500


@app.route('/api/data/sample')
def api_get_sample():
    """Get sample data for demonstration."""
    try:
        sample_df = create_sample_data()
        
        # Save sample data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"sample_data_{timestamp}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        sample_df.to_csv(filepath, index=False)
        
        # Return data info
        df_info = {
            'filename': filename,
            'shape': sample_df.shape,
            'columns': sample_df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
            'head': sample_df.head().to_dict('records'),
            'date_range': {
                'start': str(sample_df['date'].min().date()),
                'end': str(sample_df['date'].max().date()),
                'periods': len(sample_df)
            }
        }
        
        return jsonify({
            'success': True,
            'message': 'Sample data generated',
            'data_info': df_info,
            'has_date_column': True
        })
        
    except Exception as e:
        logger.error(f"Sample data error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    """Perform forecasting with uploaded data."""
    try:
        if not current_model or not current_forecaster:
            return jsonify({'success': False, 'message': 'Model not initialized'}), 400
        
        data = request.get_json()
        filename = data.get('filename')
        data_definition = data.get('data_definition', {})
        use_covariates = data.get('use_covariates', False)
        use_quantiles = data.get('use_quantiles', False)
        context_len = int(data.get('context_len', 100))
        horizon_len = int(data.get('horizon_len', 24))
        
        if not filename:
            return jsonify({'success': False, 'message': 'No data file specified'}), 400
        
        # Load data
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'Data file not found'}), 400
        
        # Process data
        data_processor = DataProcessor()
        processed_data = data_processor.load_csv_data(filepath, data_definition)
        
        # Check data sufficiency
        total_required = context_len + horizon_len
        if len(processed_data) < total_required:
            return jsonify({
                'success': False, 
                'message': f'Insufficient data. Need {total_required} periods, have {len(processed_data)}'
            }), 400
        
        # Prepare forecast data
        target_column = None
        for col, dtype in data_definition.items():
            if dtype == 'target':
                target_column = col
                break
        
        if not target_column:
            return jsonify({'success': False, 'message': 'No target column specified'}), 400
        
        target_inputs, covariates = data_processor.prepare_forecast_data(
            processed_data, context_len, horizon_len, target_column
        )
        
        # Perform forecasting
        results = {}
        
        # Basic forecasting
        point_forecast, _ = current_forecaster.forecast_basic(target_inputs, freq=0)
        results['point_forecast'] = point_forecast[0].tolist()
        
        # Quantile forecasting if requested
        if use_quantiles:
            try:
                _, quantile_forecast = current_forecaster.forecast_with_quantiles(target_inputs, freq=0)
                if quantile_forecast is not None:
                    results['quantile_forecast'] = quantile_forecast[0].tolist()
            except:
                pass
        
        # Covariates forecasting if requested
        if use_covariates and any(covariates.values()):
            try:
                enhanced_forecast, linear_forecast = current_forecaster.forecast_with_covariates(
                    inputs=target_inputs,
                    dynamic_numerical_covariates=covariates.get('dynamic_numerical_covariates'),
                    dynamic_categorical_covariates=covariates.get('dynamic_categorical_covariates'),
                    static_numerical_covariates=covariates.get('static_numerical_covariates'),
                    static_categorical_covariates=covariates.get('static_categorical_covariates'),
                    freq=0
                )
                results['enhanced_forecast'] = enhanced_forecast[0].tolist()
                results['linear_forecast'] = linear_forecast[0].tolist()
            except Exception as e:
                logger.warning(f"Covariates forecasting failed: {str(e)}")
        
        # Generate prediction intervals
        try:
            intervals = current_forecaster.generate_prediction_intervals(
                inputs=target_inputs,
                freq=0,
                covariates=covariates if use_covariates else None,
                num_bootstrap_samples=50,
                confidence_levels=[0.5, 0.8, 0.95]
            )
            
            results['prediction_intervals'] = {}
            for key, values in intervals.items():
                results['prediction_intervals'][key] = values.tolist()
        except Exception as e:
            logger.warning(f"Prediction intervals failed: {str(e)}")
        
        # Prepare visualization data
        historical_data = target_inputs
        dates_historical = processed_data['date'].iloc[:context_len].tolist()
        dates_future = processed_data['date'].iloc[context_len:context_len + horizon_len].tolist() \
                      if len(processed_data) > context_len + horizon_len else \
                      [dates_historical[-1] + timedelta(weeks=i+1) for i in range(horizon_len)]
        
        # Convert dates to strings for JSON serialization
        visualization_data = {
            'historical_data': historical_data,
            'dates_historical': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in dates_historical],
            'dates_future': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in dates_future],
            'target_name': target_column
        }
        
        return jsonify({
            'success': True,
            'message': 'Forecasting completed successfully',
            'results': results,
            'visualization_data': visualization_data,
            'forecast_summary': {
                'methods_used': list(results.keys()),
                'context_length': context_len,
                'horizon_length': horizon_len,
                'target_column': target_column,
                'covariates_used': use_covariates and any(covariates.values())
            }
        })
        
    except Exception as e:
        logger.error(f"Forecasting error: {str(e)}")
        return jsonify({'success': False, 'message': f'Forecasting failed: {str(e)}'}), 500


@app.route('/api/visualize', methods=['POST'])
def api_visualize():
    """Generate visualization and return as base64 image."""
    try:
        data = request.get_json()
        viz_data = data.get('visualization_data', {})
        results = data.get('results', {})
        
        if not current_visualizer:
            return jsonify({'success': False, 'message': 'Visualizer not initialized'}), 400
        
        # Extract data
        historical_data = viz_data.get('historical_data', [])
        dates_historical = [pd.to_datetime(d) for d in viz_data.get('dates_historical', [])]
        dates_future = [pd.to_datetime(d) for d in viz_data.get('dates_future', [])]
        target_name = viz_data.get('target_name', 'Value')
        
        # Choose best forecast
        if 'enhanced_forecast' in results:
            forecast = results['enhanced_forecast']
            title = f"{target_name} Forecast with Covariates Enhancement"
        elif 'point_forecast' in results:
            forecast = results['point_forecast']
            title = f"{target_name} Forecast (TimesFM)"
        else:
            return jsonify({'success': False, 'message': 'No forecast data available'}), 400
        
        # Prepare intervals
        intervals = results.get('prediction_intervals', {})
        
        # Generate plot
        fig = current_visualizer.plot_forecast_with_intervals(
            historical_data=historical_data,
            forecast=forecast,
            intervals=intervals if intervals else None,
            dates_historical=dates_historical,
            dates_future=dates_future,
            title=title,
            target_name=target_name
        )
        
        # Convert to base64
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'message': 'Visualization generated successfully',
            'image': img_base64
        })
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return jsonify({'success': False, 'message': f'Visualization failed: {str(e)}'}), 500


@app.route('/health')
def health_check():
    """Health check endpoint for deployment."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': current_model is not None
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'success': False, 'message': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    return jsonify({'success': False, 'message': 'Internal server error.'}), 500


if __name__ == '__main__':
    # Configuration for different environments
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Initialize default model for local development
    if debug:
        logger.info("Development mode - initializing default model")
        init_model(backend='cpu', context_len=100, horizon_len=24, 
                  checkpoint="google/timesfm-2.0-500m-pytorch")
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=debug)