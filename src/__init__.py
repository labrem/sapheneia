"""
Sapheneia TimesFM Library

A comprehensive TimesFM (Google's Time Series Foundation Model) implementation 
for financial forecasting and time series analysis with covariates support.

This package provides:
- Model initialization and configuration
- Data processing and validation
- Forecasting with optional covariates
- Professional visualization
- Quantile forecasting capabilities
"""

__version__ = "1.0.0"
__author__ = "Sapheneia Research Team"

from .model import TimesFMModel
from .data import DataProcessor
from .forecast import Forecaster
from .visualization import Visualizer

__all__ = [
    "TimesFMModel",
    "DataProcessor", 
    "Forecaster",
    "Visualizer"
]