"""
TimesFM Forecasting Module

This module provides comprehensive forecasting capabilities using Google's TimesFM
foundation model, including point forecasts, quantile forecasts, and covariates support.

Key Features:
- Point forecasting with TimesFM
- Experimental quantile forecasting (following Marcelo's approach)
- Covariates-based forecasting with dynamic and static variables
- Bootstrap-based prediction intervals generation
- Comprehensive forecast validation and error handling
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
import timesfm

logger = logging.getLogger(__name__)


class Forecaster:
    """
    High-level interface for TimesFM forecasting operations.
    
    This class provides methods for different types of forecasting operations,
    including basic point forecasts, quantile forecasts, and covariates-enhanced
    forecasting with proper data validation and error handling.
    
    Example:
        >>> forecaster = Forecaster(model)
        >>> point_forecast, quantiles = forecaster.forecast_with_quantiles(
        ...     inputs=[1,2,3,4,5], 
        ...     freq=0
        ... )
    """
    
    def __init__(self, model: timesfm.TimesFm):
        """
        Initialize the Forecaster with a loaded TimesFM model.
        
        Args:
            model: Initialized TimesFM model instance
        """
        self.model = model
        self.capabilities = self._detect_capabilities()
        logger.info(f"Forecaster initialized with capabilities: {list(self.capabilities.keys())}")
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect available forecasting capabilities of the model."""
        return {
            'basic_forecasting': True,
            'quantile_forecasting': hasattr(self.model, 'experimental_quantile_forecast'),
            'covariates_support': hasattr(self.model, 'forecast_with_covariates')
        }
    
    def forecast_basic(
        self,
        inputs: Union[List[float], List[List[float]]],
        freq: Union[int, List[int]] = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform basic point forecasting using TimesFM.
        
        Args:
            inputs: Input time series data (single series or batch)
            freq: Frequency indicator(s) [0], [1], or [2]
            
        Returns:
            Tuple of (forecast_array, metadata)
        """
        logger.info("Performing basic TimesFM forecasting...")
        
        # Normalize inputs format
        if isinstance(inputs[0], (int, float)):
            inputs = [inputs]  # Single series
            
        if isinstance(freq, int):
            freq = [freq] * len(inputs)
        
        try:
            # Perform forecasting
            forecast, metadata = self.model.forecast(inputs=inputs, freq=freq)
            forecast_array = np.array(forecast)
            
            logger.info(f"✅ Basic forecast completed. Shape: {forecast_array.shape}")
            
            return forecast_array, {
                'method': 'basic_timesfm',
                'input_series': len(inputs),
                'forecast_length': forecast_array.shape[-1],
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Basic forecasting failed: {str(e)}")
            raise
    
    def forecast(
        self,
        inputs: Union[List[float], List[List[float]]],
        freq: Union[int, List[int]] = 0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform TimesFM forecasting and return both point and quantile forecasts.
        
        In the installed TimesFM version used by the notebooks, `TimesFm.forecast`
        returns a tuple: (point_forecast, experimental_quantile_forecast).
        We rely on that directly here for robustness.
        """
        logger.info("Performing TimesFM forecasting with built-in quantiles...")

        # Normalize inputs format
        if isinstance(inputs[0], (int, float)):
            inputs_norm = [inputs]
        else:
            inputs_norm = inputs

        if isinstance(freq, int):
            freq_norm = [freq] * len(inputs_norm)
        else:
            freq_norm = freq

        try:
            # Many TimesFM builds return (point, quantiles)
            point, maybe_quantiles = self.model.forecast(inputs=inputs_norm, freq=freq_norm)
            point_array = np.array(point)

            quantile_array: Optional[np.ndarray] = None
            try:
                quantile_array = np.array(maybe_quantiles)
            except Exception:
                quantile_array = None

            logger.info(f"✅ Forecast completed. point shape: {point_array.shape}, quantiles: {None if quantile_array is None else quantile_array.shape}")
            return point_array, quantile_array

        except Exception as e:
            logger.error(f"❌ Forecast failed: {str(e)}")
            raise
    
    def forecast_with_covariates(
        self,
        inputs: Union[List[float], List[List[float]]],
        dynamic_numerical_covariates: Optional[Dict[str, List[List[float]]]] = None,
        dynamic_categorical_covariates: Optional[Dict[str, List[List[str]]]] = None,
        static_numerical_covariates: Optional[Dict[str, List[float]]] = None,
        static_categorical_covariates: Optional[Dict[str, List[str]]] = None,
        freq: Union[int, List[int]] = 0,
        xreg_mode: str = "xreg + timesfm",
        ridge: float = 0.0,
        normalize_xreg_target_per_input: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform forecasting with covariates support and quantile predictions.
        
        This method uses TimesFM's forecast_with_covariates functionality to
        incorporate exogenous variables and always attempts to get quantiles.
        
        Args:
            inputs: Input time series data
            dynamic_numerical_covariates: Dynamic numerical covariates dict
            dynamic_categorical_covariates: Dynamic categorical covariates dict  
            static_numerical_covariates: Static numerical covariates dict
            static_categorical_covariates: Static categorical covariates dict
            freq: Frequency indicator(s)
            xreg_mode: Covariate integration mode ("xreg + timesfm" or "timesfm + xreg")
            ridge: Ridge regression parameter for covariates
            normalize_xreg_target_per_input: Whether to normalize covariates
            
        Returns:
            Tuple of (enhanced_forecast, quantile_forecast)
        """
        if not self.capabilities['covariates_support']:
            raise ValueError("Model does not support covariates forecasting")
        
        logger.info(f"Performing covariates-enhanced forecasting (mode: {xreg_mode})...")
        
        # Normalize inputs format
        if isinstance(inputs[0], (int, float)):
            inputs = [inputs]
            
        if isinstance(freq, int):
            freq = [freq] * len(inputs)
        
        # Validate covariates
        self._validate_covariates(
            inputs, 
            dynamic_numerical_covariates,
            dynamic_categorical_covariates,
            static_numerical_covariates,
            static_categorical_covariates
        )
        
        try:
            # Perform covariates forecasting
            enhanced_forecast, linear_forecast = self.model.forecast_with_covariates(
                inputs=inputs,
                dynamic_numerical_covariates=dynamic_numerical_covariates or {},
                dynamic_categorical_covariates=dynamic_categorical_covariates or {},
                static_numerical_covariates=static_numerical_covariates or {},
                static_categorical_covariates=static_categorical_covariates or {},
                freq=freq,
                xreg_mode=xreg_mode,
                ridge=ridge,
                normalize_xreg_target_per_input=normalize_xreg_target_per_input
            )
            
            enhanced_array = np.array(enhanced_forecast)
            
            logger.info(f"✅ Covariates forecasting completed.")
            logger.info(f"  Enhanced forecast shape: {enhanced_array.shape}")
            
            # Try to get quantiles for the enhanced forecast
            quantile_forecast = None
            if self.capabilities['quantile_forecasting']:
                try:
                    logger.info("Getting quantiles for covariates-enhanced forecast...")
                    quantile_result = self.model.experimental_quantile_forecast(
                        inputs=inputs,
                        freq=freq
                    )
                    quantile_forecast = np.array(quantile_result)
                    logger.info(f"✅ Quantile forecast for covariates completed. Shape: {quantile_forecast.shape}")
                except Exception as e:
                    logger.warning(f"⚠️ Quantile forecasting with covariates failed: {str(e)}")
            
            return enhanced_array, quantile_forecast
            
        except Exception as e:
            logger.error(f"❌ Covariates forecasting failed: {str(e)}")
            raise
    
    def _validate_covariates(
        self,
        inputs: List[List[float]],
        dynamic_numerical: Optional[Dict],
        dynamic_categorical: Optional[Dict],
        static_numerical: Optional[Dict],
        static_categorical: Optional[Dict]
    ) -> None:
        """Validate covariates data structure and compatibility."""
        logger.info("Validating covariates data structure...")
        
        num_series = len(inputs)
        
        # Validate dynamic covariates
        for cov_name, cov_dict in [
            ("dynamic_numerical", dynamic_numerical),
            ("dynamic_categorical", dynamic_categorical)
        ]:
            if not cov_dict:
                continue
                
            for var_name, var_data in cov_dict.items():
                if len(var_data) != num_series:
                    raise ValueError(f"{cov_name} covariate '{var_name}' must have {num_series} series, got {len(var_data)}")
                
                # Check that dynamic covariates have context + horizon length
                expected_length = len(inputs[0]) + getattr(self.model.hparams, 'horizon_len', 24)
                for i, series in enumerate(var_data):
                    if len(series) < expected_length:
                        logger.warning(f"{cov_name} covariate '{var_name}' series {i} may be too short: {len(series)} < {expected_length}")
        
        # Validate static covariates  
        for cov_name, cov_dict in [
            ("static_numerical", static_numerical),
            ("static_categorical", static_categorical)
        ]:
            if not cov_dict:
                continue
                
            for var_name, var_data in cov_dict.items():
                if len(var_data) != num_series:
                    raise ValueError(f"{cov_name} covariate '{var_name}' must have {num_series} values, got {len(var_data)}")
        
        logger.info("✅ Covariates validation passed")
    
    def get_forecast_summary(
        self, 
        forecast: np.ndarray,
        intervals: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of forecast results.
        
        Args:
            forecast: Point forecast array
            intervals: Optional prediction intervals dictionary
            
        Returns:
            Dictionary containing forecast summary statistics
        """
        summary = {
            'forecast_shape': forecast.shape,
            'forecast_statistics': {
                'min': float(np.min(forecast)),
                'max': float(np.max(forecast)),
                'mean': float(np.mean(forecast)),
                'std': float(np.std(forecast)),
                'median': float(np.median(forecast))
            },
            'has_intervals': intervals is not None
        }
        
        if intervals:
            summary['interval_widths'] = {}
            for key, values in intervals.items():
                if key.startswith('lower_') or key.startswith('upper_'):
                    continue
                if key.endswith('_forecast'):
                    continue
                    
                # Calculate interval widths for confidence levels
                conf_level = key.split('_')[-1]
                if f'lower_{conf_level}' in intervals and f'upper_{conf_level}' in intervals:
                    width = np.mean(intervals[f'upper_{conf_level}'] - intervals[f'lower_{conf_level}'])
                    summary['interval_widths'][f'{conf_level}%_interval'] = float(width)
        
        return summary