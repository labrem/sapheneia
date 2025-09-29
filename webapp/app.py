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
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import Sapheneia TimesFM modules
from model import TimesFMModel, initialize_timesfm_model
from data import DataProcessor, prepare_visualization_data
from forecast import Forecaster, run_forecast, process_quantile_bands

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


def init_model(backend='cpu', context_len=64, horizon_len=24, checkpoint=None, local_path=None):
    """Initialize TimesFM model with given parameters using centralized function."""
    global current_model, current_forecaster, current_visualizer
    
    try:
        logger.info(f"Initializing model with backend={backend}, context={context_len}, horizon={horizon_len}")
        
        # Use centralized model initialization
        current_model, current_forecaster, current_visualizer = initialize_timesfm_model(
            backend=backend,
            context_len=context_len,
            horizon_len=horizon_len,
            checkpoint=checkpoint,
            local_model_path=local_path
        )
        
        logger.info("Model initialized successfully")
        return True, "Model initialized successfully"
        
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        return False, f"Model initialization failed: {str(e)}"


# Sample data generation removed as per requirements


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
        context_len = int(data.get('context_len', 64))
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
        
        try:
            file.save(filepath)
            logger.info(f"File saved successfully: {filepath}")
            
            # Load and analyze data
            df = pd.read_csv(filepath)
            logger.info(f"CSV loaded successfully with shape: {df.shape}")
            
            # Convert data to JSON-serializable format
            df_head = df.head()
            head_records = []
            for _, row in df_head.iterrows():
                record = {}
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        record[col] = None
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        record[col] = str(value)
                    elif isinstance(value, (np.integer, np.floating)):
                        record[col] = float(value)
                    else:
                        record[col] = str(value)
                head_records.append(record)
            
            df_info = {
                'filename': filename,
                'shape': list(df.shape),  # Convert tuple to list
                'columns': df.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'head': head_records,
                'null_counts': {col: int(count) for col, count in df.isnull().sum().items()}
            }
            
            # Check for date column
            has_date = 'date' in df.columns
            if has_date:
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df_info['date_range'] = {
                        'start': str(df['date'].min().date()) if not df['date'].isnull().all() else None,
                        'end': str(df['date'].max().date()) if not df['date'].isnull().all() else None,
                        'periods': len(df)
                    }
                except Exception as date_error:
                    logger.warning(f"Date parsing failed: {date_error}")
                    has_date = False
            
            # Check if this looks like forecast output data instead of time series data
            forecast_output_indicators = [
                'period', 'point_forecast', 'quantile_forecast', 'forecast', 
                'prediction', 'forecast_lower', 'forecast_upper', 'quantile'
            ]
            
            column_names_lower = [col.lower() for col in df.columns]
            is_forecast_output = any(indicator in ' '.join(column_names_lower) for indicator in forecast_output_indicators)
            
            if is_forecast_output:
                logger.warning("Detected forecast output data instead of time series data")
                return jsonify({
                    'success': False, 
                    'message': 'This appears to be forecast output data, not time series input data. Please upload your original time series data with a "date" column and numeric value columns.',
                    'is_forecast_output': True,
                    'suggested_columns': ['date', 'value', 'price', 'amount', 'count', 'sales', 'revenue']
                }), 400
            
            logger.info(f"Data analysis completed. Has date column: {has_date}")
            
            # Create response
            response_data = {
                'success': True,
                'message': 'File uploaded successfully',
                'data_info': df_info,
                'has_date_column': has_date
            }
            
            logger.info(f"Sending response with keys: {list(response_data.keys())}")
            return jsonify(response_data)
            
        except Exception as processing_error:
            logger.error(f"File processing error: {processing_error}")
            if os.path.exists(filepath):
                os.remove(filepath)
            raise processing_error
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return jsonify({'success': False, 'message': f'Upload failed: {str(e)}'}), 500


@app.route('/api/sample_data', methods=['POST'])
def api_sample_data():
    """Generate sample time series data for testing."""
    try:
        data = request.get_json()
        data_type = data.get('type', 'financial')
        periods = int(data.get('periods', 100))
        
        # Generate sample data
        dates = pd.date_range(start='2020-01-01', periods=periods, freq='D')
        
        if data_type == 'financial':
            # Generate financial time series (like stock prices)
            np.random.seed(42)
            base_price = 100
            returns = np.random.normal(0.001, 0.02, periods)  # Daily returns
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            sample_data = pd.DataFrame({
                'date': dates,
                'price': prices,
                'volume': np.random.randint(1000, 10000, periods),
                'volatility': np.random.uniform(0.1, 0.3, periods)
            })
            
        elif data_type == 'sales':
            # Generate sales data
            np.random.seed(42)
            trend = np.linspace(100, 150, periods)
            seasonal = 20 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
            noise = np.random.normal(0, 5, periods)
            sales = trend + seasonal + noise
            
            sample_data = pd.DataFrame({
                'date': dates,
                'sales': sales,
                'customers': np.random.randint(50, 200, periods),
                'marketing_spend': np.random.uniform(1000, 5000, periods)
            })
            
        else:
            # Generate generic time series
            np.random.seed(42)
            trend = np.linspace(0, 100, periods)
            seasonal = 10 * np.sin(2 * np.pi * np.arange(periods) / 30)
            noise = np.random.normal(0, 2, periods)
            values = trend + seasonal + noise
            
            sample_data = pd.DataFrame({
                'date': dates,
                'value': values,
                'category': np.random.choice(['A', 'B', 'C'], periods),
                'score': np.random.uniform(0, 100, periods)
            })
        
        # Save sample data
        filename = f"sample_{data_type}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        sample_data.to_csv(filepath, index=False)
        
        # Return data info
        df_info = {
            'filename': filename,
            'shape': list(sample_data.shape),
            'columns': sample_data.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in sample_data.dtypes.items()},
            'head': sample_data.head().to_dict('records'),
            'null_counts': {col: int(count) for col, count in sample_data.isnull().sum().items()},
            'date_range': {
                'start': str(sample_data['date'].min().date()),
                'end': str(sample_data['date'].max().date()),
                'periods': len(sample_data)
            }
        }
        
        return jsonify({
            'success': True,
            'message': f'Sample {data_type} data generated successfully',
            'data_info': df_info,
            'has_date_column': True
        })
        
    except Exception as e:
        logger.error(f"Sample data generation error: {str(e)}")
        return jsonify({'success': False, 'message': f'Sample data generation failed: {str(e)}'}), 500


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
        context_len = int(data.get('context_len', 64))
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
        
        # Perform forecasting using centralized function
        try:
            # Ensure target_inputs is in the correct format (list of lists)
            if isinstance(target_inputs[0], (int, float)):
                target_inputs_formatted = [target_inputs]
            else:
                target_inputs_formatted = target_inputs
            
            results = run_forecast(
                forecaster=current_forecaster,
                target_inputs=target_inputs_formatted,
                covariates=covariates if use_covariates and any(covariates.values()) else None,
                use_covariates=use_covariates and any(covariates.values()),
                freq=0
            )
            
            # Check for NaN values before JSON serialization
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    if np.any(np.isnan(value)):
                        logger.error(f"❌ NaN values detected in {key}: {np.isnan(value).sum()} out of {value.size}")
                        return jsonify({
                            'success': False, 
                            'message': f'Forecasting failed: Invalid values (NaN) detected in {key}. This may be due to insufficient data or model issues.'
                        }), 500
                    
                    if key == 'quantile_forecast':
                        # For quantiles, keep the full array structure
                        if value.ndim == 3:
                            results[key] = value[0].tolist()  # (1, horizon, quantiles) -> (horizon, quantiles)
                        else:
                            results[key] = value.tolist()
                    elif value.ndim > 1:
                        results[key] = value[0].tolist()  # Take first series if batch
                    else:
                        results[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    # Check for NaN in lists/tuples
                    if any(isinstance(x, float) and np.isnan(x) for x in value):
                        logger.error(f"❌ NaN values detected in {key} list")
                        return jsonify({
                            'success': False, 
                            'message': f'Forecasting failed: Invalid values (NaN) detected in {key}. This may be due to insufficient data or model issues.'
                        }), 500
            
            logger.info(f"✅ Centralized forecasting completed. Methods: {list(results.keys())}")
            logger.info(f"Results structure: {[(k, type(v), len(v) if hasattr(v, '__len__') else 'N/A') for k, v in results.items()]}")
            if 'quantile_forecast' in results:
                shape_quantile = len(results['quantile_forecast']) if hasattr(results['quantile_forecast'], '__len__') else 'N/A'
                logger.info(f"Quantile forecast shape: {shape_quantile}")
            else:
                logger.warning("No quantile_forecast in results!")
            
        except Exception as e:
            logger.error(f"Centralized forecasting failed: {str(e)}")
            return jsonify({'success': False, 'message': f'Forecasting failed: {str(e)}'}), 500
        
        # Prepare visualization data using centralized function
        visualization_data = prepare_visualization_data(
            processed_data=processed_data,
            target_inputs=target_inputs,
            target_column=target_column,
            context_len=context_len,
            horizon_len=horizon_len
        )
        
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
        selected_indices = data.get('quantile_indices', [])
        
        if not current_visualizer:
            return jsonify({'success': False, 'message': 'Visualizer not initialized'}), 400
        
        # Extract data
        historical_data = viz_data.get('historical_data', [])
        dates_historical = [pd.to_datetime(d) for d in viz_data.get('dates_historical', [])]
        dates_future = [pd.to_datetime(d) for d in viz_data.get('dates_future', [])]
        actual_future = viz_data.get('actual_future', [])
        target_name = viz_data.get('target_name', 'Value')
        
        # Debug logging
        logger.info(f"Visualization data - historical_data length: {len(historical_data)}")
        logger.info(f"Visualization data - dates_historical length: {len(dates_historical)}")
        logger.info(f"Visualization data - dates_future length: {len(dates_future)}")
        logger.info(f"Visualization data - target_name: {target_name}")
        logger.info(f"Results keys: {list(results.keys())}")
        
        # Choose best forecast
        if 'point_forecast' in results:
            forecast = results['point_forecast']
            if results.get('method') == 'covariates_enhanced':
                title = f"{target_name} Forecast with Covariates Enhancement"
            else:
                title = f"{target_name} Forecast (TimesFM)"
        else:
            return jsonify({'success': False, 'message': 'No forecast data available'}), 400
        
        # Process quantile bands using centralized function
        intervals = {}
        used_quantile_intervals = False
        quantile_shape = None
        
        logger.info(f"Available results keys: {list(results.keys())}")
        if 'quantile_forecast' in results:
            try:
                quantiles = np.array(results['quantile_forecast'])
                quantile_shape = list(quantiles.shape)
                logger.info(f"Quantile forecast shape received for viz: {quantile_shape}")
                
                # Use centralized quantile processing
                intervals = process_quantile_bands(
                    quantile_forecast=quantiles,
                    selected_indices=selected_indices if selected_indices else None
                )
                
                used_quantile_intervals = len(intervals) > 0
                logger.info(f"✅ Processed quantile bands using centralized function. Bands: {len(intervals)//3}")
                
            except Exception as e:
                logger.warning(f"Quantile band processing failed: {e}")
                intervals = {}
                used_quantile_intervals = False
        else:
            logger.warning("No quantile_forecast found in results - quantile intervals will not be displayed")
        
        # Generate plot
        try:
            logger.info(f"Generating plot with forecast length: {len(forecast)}")
            logger.info(f"Historical data type: {type(historical_data)}, length: {len(historical_data) if hasattr(historical_data, '__len__') else 'N/A'}")
            logger.info(f"Forecast type: {type(forecast)}, length: {len(forecast) if hasattr(forecast, '__len__') else 'N/A'}")
            logger.info(f"Intervals keys: {list(intervals.keys()) if intervals else 'None'}")
            logger.info(f"Dates historical length: {len(dates_historical)}")
            logger.info(f"Dates future length: {len(dates_future)}")

            fig = current_visualizer.plot_forecast_with_intervals(
                historical_data=historical_data,
                forecast=forecast,
                intervals=intervals if intervals else None,
                actual_future=actual_future if actual_future else None,
                dates_historical=dates_historical,
                dates_future=dates_future,
                title=title,
                target_name=target_name,
                show_figure=False
            )
            logger.info("Interactive plot generated successfully")
        except Exception as plot_error:
            logger.error(f"Plot generation failed: {str(plot_error)}")
            import traceback
            traceback.print_exc()
            raise plot_error

        figure_payload = json.loads(fig.to_json())
        plot_config = {
            'responsive': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }

        return jsonify({
            'success': True,
            'message': 'Visualization generated successfully',
            'figure': figure_payload,
            'config': plot_config,
            'used_quantile_intervals': used_quantile_intervals,
            'quantile_shape': quantile_shape
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
        init_model(backend='cpu', context_len=64, horizon_len=24, 
                   checkpoint="google/timesfm-2.0-500m-pytorch")
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=debug)
