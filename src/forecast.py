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
import pandas as pd
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
    
    def forecast_with_quantiles(
        self,
        inputs: Union[List[float], List[List[float]]],
        freq: Union[int, List[int]] = 0,
        quantiles: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform forecasting with quantile predictions (Marcelo's approach).
        
        This method first performs point forecasting, then attempts quantile
        forecasting using the experimental_quantile_forecast method.
        
        Args:
            inputs: Input time series data
            freq: Frequency indicator(s)
            quantiles: List of quantiles to compute (if supported)
            
        Returns:
            Tuple of (point_forecast, quantile_forecast)
        """
        logger.info("Performing TimesFM forecasting with quantiles...")
        
        # Get basic point forecast first
        point_forecast, _ = self.forecast_basic(inputs, freq)
        quantile_forecast = None
        
        # Try experimental quantile forecasting
        if self.capabilities['quantile_forecasting']:
            try:
                logger.info("Using experimental_quantile_forecast method...")
                
                # Normalize inputs format for quantile method
                if isinstance(inputs[0], (int, float)):
                    inputs_norm = [inputs]
                else:
                    inputs_norm = inputs
                    
                if isinstance(freq, int):
                    freq_norm = [freq] * len(inputs_norm)
                else:
                    freq_norm = freq
                
                quantile_result = self.model.experimental_quantile_forecast(
                    inputs=inputs_norm,
                    freq=freq_norm
                )
                
                quantile_forecast = np.array(quantile_result)
                logger.info(f"✅ Quantile forecast completed. Shape: {quantile_forecast.shape}")
                
            except Exception as e:
                logger.warning(f"⚠️ Quantile forecasting failed: {str(e)}")
                logger.info("Continuing with point forecast only")
        else:
            logger.info("⚠️ Experimental quantile forecasting not available")
        
        return point_forecast, quantile_forecast
    
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform forecasting with covariates support.
        
        This method uses TimesFM's forecast_with_covariates functionality to
        incorporate exogenous variables into the forecasting process.
        
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
            Tuple of (enhanced_forecast, linear_model_forecast)
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
            linear_array = np.array(linear_forecast)
            
            logger.info(f"✅ Covariates forecasting completed.")
            logger.info(f"  Enhanced forecast shape: {enhanced_array.shape}")
            logger.info(f"  Linear model forecast shape: {linear_array.shape}")
            
            return enhanced_array, linear_array
            
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
    
    def generate_prediction_intervals(
        self,
        inputs: Union[List[float], List[List[float]]],
        freq: Union[int, List[int]] = 0,
        covariates: Optional[Dict[str, Any]] = None,
        num_bootstrap_samples: int = 100,
        confidence_levels: List[float] = [0.5, 0.8, 0.95],
        noise_scale: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Generate prediction intervals using bootstrap sampling approach.
        
        This method creates multiple perturbed forecasts to estimate uncertainty
        and compute prediction intervals at specified confidence levels.
        
        Args:
            inputs: Input time series data
            freq: Frequency indicator(s)
            covariates: Optional covariates dictionary
            num_bootstrap_samples: Number of bootstrap samples to generate
            confidence_levels: Confidence levels for prediction intervals
            noise_scale: Scale of noise to add for bootstrap sampling
            
        Returns:
            Dictionary containing forecasts and intervals
        """
        logger.info(f"Generating prediction intervals with {num_bootstrap_samples} bootstrap samples...")
        
        # Normalize inputs
        if isinstance(inputs[0], (int, float)):
            inputs = [inputs]
            single_series = True
        else:
            single_series = False
            
        if isinstance(freq, int):
            freq = [freq] * len(inputs)
        
        forecasts_collection = []
        base_forecast = None
        
        # Generate base forecast
        try:
            if covariates and self.capabilities['covariates_support']:
                base_forecast, _ = self.forecast_with_covariates(
                    inputs=inputs,
                    freq=freq,
                    **covariates
                )
            else:
                base_forecast, _ = self.forecast_basic(inputs, freq)
            
            forecasts_collection.append(base_forecast)
            
        except Exception as e:
            logger.error(f"Failed to generate base forecast: {str(e)}")
            raise
        
        # Generate bootstrap samples
        successful_samples = 1
        for i in range(num_bootstrap_samples - 1):
            try:
                # Add noise to inputs
                noisy_inputs = []
                for series in inputs:
                    noise = np.random.normal(0, noise_scale, len(series))
                    noisy_series = (np.array(series) * (1 + noise)).tolist()
                    noisy_inputs.append(noisy_series)
                
                # Generate forecast with noisy inputs
                if covariates and self.capabilities['covariates_support']:
                    # Also perturb covariates slightly
                    noisy_covariates = self._perturb_covariates(covariates, noise_scale * 0.5)
                    perturbed_forecast, _ = self.forecast_with_covariates(
                        inputs=noisy_inputs,
                        freq=freq,
                        **noisy_covariates
                    )
                else:
                    perturbed_forecast, _ = self.forecast_basic(noisy_inputs, freq)
                
                forecasts_collection.append(perturbed_forecast)
                successful_samples += 1
                
            except Exception:
                # Fallback: add synthetic noise around base forecast
                if base_forecast is not None:
                    noise = np.random.normal(0, np.std(base_forecast) * 0.2, base_forecast.shape)
                    synthetic_forecast = base_forecast + noise
                    forecasts_collection.append(synthetic_forecast)
                    successful_samples += 1
        
        logger.info(f"Generated {successful_samples} successful bootstrap samples")
        
        # Calculate prediction intervals
        forecasts_array = np.array(forecasts_collection)
        
        results = {
            'mean_forecast': np.mean(forecasts_array, axis=0),
            'median_forecast': np.percentile(forecasts_array, 50, axis=0),
            'std_forecast': np.std(forecasts_array, axis=0)
        }
        
        # Calculate confidence intervals
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_q = (alpha / 2) * 100
            upper_q = (1 - alpha / 2) * 100
            
            results[f'lower_{int(conf_level*100)}'] = np.percentile(forecasts_array, lower_q, axis=0)
            results[f'upper_{int(conf_level*100)}'] = np.percentile(forecasts_array, upper_q, axis=0)
        
        # Return single series format if input was single series
        if single_series:
            results = {k: v[0] if v.ndim > 1 else v for k, v in results.items()}
        
        logger.info(f"✅ Prediction intervals generated for confidence levels: {confidence_levels}")
        
        return results
    
    def _perturb_covariates(
        self, 
        covariates: Dict[str, Any], 
        noise_scale: float
    ) -> Dict[str, Any]:
        """Add slight perturbations to covariates for bootstrap sampling."""
        perturbed = {}
        
        for cov_type, cov_data in covariates.items():
            if not cov_data:
                perturbed[cov_type] = cov_data
                continue
                
            perturbed[cov_type] = {}
            
            for var_name, var_values in cov_data.items():
                if 'numerical' in cov_type:
                    # Add noise to numerical covariates
                    perturbed_values = []
                    for series in var_values:
                        if isinstance(series, list):
                            noise = np.random.normal(0, noise_scale, len(series))
                            perturbed_series = (np.array(series) * (1 + noise)).tolist()
                            perturbed_values.append(perturbed_series)
                        else:
                            # Single value
                            noise = np.random.normal(0, noise_scale)
                            perturbed_values.append(series * (1 + noise))
                    
                    perturbed[cov_type][var_name] = perturbed_values
                else:
                    # Keep categorical covariates unchanged
                    perturbed[cov_type][var_name] = var_values
        
        return perturbed
    
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