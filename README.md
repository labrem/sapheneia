# Sapheneia TimesFM

A comprehensive TimesFM (Google's Time Series Foundation Model) implementation for financial forecasting and time series analysis with advanced covariates support.

## 🚀 Quick Start

The fastest way to get started is with our automated setup script:

```bash
# Clone the repository
git clone https://github.com/labrem/sapheneia.git
cd sapheneia

# Make script executable and run
chmod +x setup.sh
./setup.sh
```

This will:
- Install UV package manager
- Set up Python 3.11 virtual environment  
- Install TimesFM with appropriate backend (JAX/PyTorch)
- Install all dependencies including Jupyter support
- Create sample data for testing
- Verify the installation

## 📦 What's Included

### Core Components

- **`/src`** - Complete Python library for TimesFM
  - `model.py` - Model initialization and configuration
  - `data.py` - CSV loading and data processing
  - `forecast.py` - Forecasting with covariates support
  - `visualization.py` - Professional plotting and analysis

- **`/notebooks`** - Comprehensive demo and research notebooks
  - `sapheneia_timesfm_demo.ipynb` - Complete working example
  - Individual researcher notebooks (Marcelo, Lucas, Matt, Matthieu)

- **`/webapp`** - Professional web application
  - Flask-based interface for TimesFM forecasting
  - Support for localhost deployment
  - Interactive parameter configuration
  - File upload and real-time visualization

### Key Features

✅ **Model Loading**: Both HuggingFace checkpoints and local model paths  
✅ **Quantile Forecasting**: Using `experimental_quantile_forecast` (Marcelo's approach)  
✅ **Covariates Support**: Dynamic/static, numerical/categorical variables  
✅ **Professional Visualization**: Publication-quality plots with prediction intervals  
✅ **Bootstrap Intervals**: Uncertainty quantification through sampling  
✅ **Web Interface**: Complete webapp for non-technical users  

## 🛠️ Setup Options

### Local Development Only
For research and notebook usage:
```bash
./setup.sh --local-only
```

### Web Application Only  
For webapp deployment:
```bash
./setup.sh --webapp-only
```


## 📊 Usage Examples

### 1. Python Library Usage

```python
from src.model import TimesFMModel
from src.data import DataProcessor  
from src.forecast import Forecaster
from src.visualization import Visualizer

# Initialize model
model_wrapper = TimesFMModel(
    backend="cpu",
    context_len=64,
    horizon_len=24,
    checkpoint="google/timesfm-2.0-500m-pytorch"
)
timesfm_model = model_wrapper.load_model()

# Process data
processor = DataProcessor()
data = processor.load_csv_data("data.csv", data_definition)
inputs, covariates = processor.prepare_forecast_data(data, 64, 24)

# Generate forecasts
forecaster = Forecaster(timesfm_model)
point_forecast, quantiles = forecaster.forecast_with_quantiles(inputs)

# Create visualization
visualizer = Visualizer()
fig = visualizer.plot_forecast_with_intervals(
    historical_data=inputs,
    forecast=point_forecast[0],
    intervals=quantiles
)
```

### 2. Jupyter Notebook

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter
uv run jupyter notebook

# Open the demo notebook
# notebooks/sapheneia_timesfm_demo.ipynb
```

### 3. Web Application

```bash
# Activate environment
source .venv/bin/activate

# Start webapp
cd webapp
python app.py

# Open browser to http://localhost:8080
```

## 🔧 Configuration

### Data Format

Your CSV file must have:
- A `date` column as the first column
- One or more time series columns
- A data definition JSON specifying column types:

```json
{
  "target_price": "target",
  "covariate_1": "dynamic_numerical",
  "covariate_2": "dynamic_categorical", 
  "category": "static_categorical",
  "base_value": "static_numerical"
}
```

### Model Parameters

Key configuration options:

- **Context Length**: 100-2048 (input time series length)
- **Horizon Length**: 1-128 (forecast periods)
- **Backend**: "cpu", "gpu", "tpu"
- **Checkpoint**: HuggingFace repo ID or local path
- **Covariates**: Enable/disable exogenous variables
- **Quantiles**: Enable experimental quantile forecasting

## 🌐 Web Application Features

- **File Upload**: CSV data with automatic column detection
- **Interactive Configuration**: Model and forecasting parameters
- **Real-time Visualization**: Professional charts with prediction intervals
- **Multiple Forecast Methods**: Basic, quantiles, and covariates-enhanced
- **Downloadable Results**: Charts and data tables
- **Sample Data Generation**: Built-in synthetic financial data


## 📁 Project Structure

```
sapheneia/
├── src/                        # Core Python library
│   ├── model.py               # TimesFM model management
│   ├── data.py                # Data processing utilities  
│   ├── forecast.py            # Forecasting functionality
│   └── visualization.py       # Professional plotting
├── notebooks/                  # Jupyter notebooks
│   ├── sapheneia_timesfm_demo.ipynb  # Complete demo
│   └── [researcher folders]/  # Individual research
├── webapp/                     # Web application
│   ├── app.py                 # Flask application
│   ├── templates/             # HTML templates
│   ├── static/                # CSS/JS assets
│   └── requirements.txt       # Web dependencies
├── data/                      # Sample and user data
├── setup.sh                   # Automated setup script
├── CLAUDE.md                 # Development instructions
└── README.md                 # This file
```

## 🔍 Troubleshooting

### Installation Issues

```bash
# Check UV installation
uv --version

# Verify virtual environment
source .venv/bin/activate
which python

# Test TimesFM import
python -c "import timesfm; print('TimesFM ready!')"
```

### Model Loading Issues

- **Apple Silicon**: Use PyTorch version (`google/timesfm-2.0-500m-pytorch`)
- **x86_64**: Use JAX version (`google/timesfm-2.0-500m-jax`) 
- **Memory Issues**: Reduce `context_len` and `per_core_batch_size`
- **CUDA Issues**: Set backend to "cpu" if GPU problems occur

### Data Format Issues

- Ensure CSV has `date` column first
- Check data definition matches column types
- Verify sufficient data length (context_len + horizon_len)
- Confirm numeric columns contain valid numbers

## 🤝 Contributing

This is a research repository for advancing TimesFM capabilities. Each researcher maintains their own notebook directory:

- `notebooks/marcelo/` - Marcelo's experiments and covariates research
- `notebooks/lucas/` - Lucas's financial forecasting demos  
- `notebooks/matt/` - Matt's TimesFM analysis
- `notebooks/matthieu/` - Matthieu's research notebooks

## 📚 References

- [TimesFM Paper](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [TimesFM GitHub](https://github.com/google-research/timesfm)
- [Google Research TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)

## 📜 License

This project builds upon Google's TimesFM foundation model. Please refer to the original TimesFM license and terms of use.

## 🆘 Support

For issues and questions:
1. Check this README and CLAUDE.md
2. Review the demo notebook
3. Test with sample data first
4. Check logs in the webapp console

---

**Sapheneia TimesFM** - Professional Time Series Forecasting with Google's Foundation Model