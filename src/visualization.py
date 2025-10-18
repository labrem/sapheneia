"""
Professional Visualization Module for TimesFM Forecasting

This module provides comprehensive visualization capabilities for TimesFM forecasting,
including professional-grade plots with prediction intervals, covariates displays,
and publication-ready styling.

Key Features:
- Professional forecast visualizations with seamless connections
- Prediction intervals with customizable confidence levels
- Covariates subplots integration
- Sapheneia-style professional formatting
- Interactive and static plot options
- Export capabilities for presentations and publications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Set professional style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Professional visualization class for TimesFM forecasting results.
    
    This class provides methods to create publication-quality visualizations
    of forecasting results, including prediction intervals, covariates analysis,
    and comprehensive time series plots.
    
    Example:
        >>> viz = Visualizer()
        >>> fig = viz.plot_forecast_with_intervals(
        ...     historical_data=historical,
        ...     forecast=point_forecast,
        ...     intervals=prediction_intervals,
        ...     title="Bitcoin Price Forecast"
        ... )
    """
    
    def __init__(self, style: str = "professional"):
        """
        Initialize the Visualizer with specified styling.
        
        Args:
            style: Visualization style ("professional", "minimal", "presentation")
        """
        self.style = style
        self._setup_style()
        logger.info(f"Visualizer initialized with '{style}' style")
    
    def _setup_style(self) -> None:
        """Set up the visualization style and parameters."""
        if self.style == "professional":
            # Sapheneia professional style
            self.colors = {
                'historical': '#1f77b4',
                'forecast': '#d62728', 
                'actual': '#2ca02c',
                'interval_80': '#ffb366',
                'interval_50': '#ff7f0e',
                'grid': '#e0e0e0',
                'background': '#fafafa'
            }
            self.figsize = (16, 12)
            
        elif self.style == "minimal":
            # Clean minimal style
            self.colors = {
                'historical': '#2E86AB',
                'forecast': '#A23B72',
                'actual': '#F18F01',
                'interval_80': '#C73E1D',
                'interval_50': '#F18F01',
                'grid': '#f0f0f0',
                'background': 'white'
            }
            self.figsize = (14, 10)
            
        else:  # presentation
            # High contrast for presentations
            self.colors = {
                'historical': '#003f5c',
                'forecast': '#ff6361',
                'actual': '#58508d',
                'interval_80': '#ffa600',
                'interval_50': '#ff6361',
                'grid': '#e8e8e8',
                'background': 'white'
            }
            self.figsize = (18, 14)
    
    def plot_forecast_with_intervals(
        self,
        historical_data: Union[List[float], np.ndarray],
        forecast: Union[List[float], np.ndarray], 
        intervals: Optional[Dict[str, np.ndarray]] = None,
        actual_future: Optional[Union[List[float], np.ndarray]] = None,
        dates_historical: Optional[List[Union[str, datetime]]] = None,
        dates_future: Optional[List[Union[str, datetime]]] = None,
        title: str = "TimesFM Forecast with Prediction Intervals",
        target_name: str = "Value",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a professional forecast visualization with prediction intervals.
        
        Args:
            historical_data: Historical time series data
            forecast: Point forecast values
            intervals: Dictionary containing prediction intervals
            actual_future: Optional actual future values for comparison
            dates_historical: Optional dates for historical data
            dates_future: Optional dates for forecast period
            title: Plot title
            target_name: Name of the target variable
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        logger.info(f"Creating forecast visualization: {title}")
        
        # Convert to numpy arrays
        historical_data = np.array(historical_data)
        forecast = np.array(forecast)
        if actual_future is not None:
            actual_future = np.array(actual_future)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_facecolor(self.colors['background'])
        
        # Setup time axis
        if dates_historical is None:
            historical_x = np.arange(len(historical_data))
        else:
            historical_x = pd.to_datetime(dates_historical)
        
        if dates_future is None:
            future_x = np.arange(len(historical_data), len(historical_data) + len(forecast))
        else:
            future_x = pd.to_datetime(dates_future)
        
        # Plot historical data
        ax.plot(historical_x, historical_data, 
               color=self.colors['historical'], linewidth=2.5, 
               label='Historical Data', zorder=5)
        
        # Create seamless connection for forecast
        if dates_historical is None:
            connection_x = [len(historical_data) - 1] + list(future_x)
        else:
            connection_x = [historical_x[-1]] + list(future_x)
        
        connection_forecast = [historical_data[-1]] + list(forecast)
        
        # Plot quantile intervals if available
        if intervals:
            # Handle different types of intervals
            if 'lower_80' in intervals and 'upper_80' in intervals:
                # Traditional confidence intervals
                interval_lower = [historical_data[-1]] + list(intervals['lower_80'])
                interval_upper = [historical_data[-1]] + list(intervals['upper_80'])
                
                ax.fill_between(connection_x, interval_lower, interval_upper, 
                               alpha=0.3, color=self.colors['interval_80'], 
                               label='80% Quantile Interval', zorder=1)
                               
                # Add 50% interval if available
                if 'lower_50' in intervals and 'upper_50' in intervals:
                    interval_lower_50 = [historical_data[-1]] + list(intervals['lower_50'])
                    interval_upper_50 = [historical_data[-1]] + list(intervals['upper_50'])
                    
                    ax.fill_between(connection_x, interval_lower_50, interval_upper_50, 
                                   alpha=0.5, color=self.colors['interval_50'], 
                                   label='50% Quantile Interval', zorder=2)
            
            else:
                # Check for generic confidence levels
                conf_levels = []
                for key in intervals.keys():
                    if key.startswith('lower_'):
                        conf_level = key.split('_')[1]
                        if f'upper_{conf_level}' in intervals:
                            conf_levels.append(int(conf_level))
                
                conf_levels.sort(reverse=True)  # Largest first for layering
                
                for conf_level in conf_levels:
                    lower_key = f'lower_{conf_level}'
                    upper_key = f'upper_{conf_level}'
                    
                    if lower_key in intervals and upper_key in intervals:
                        # Create seamless intervals
                        interval_lower = [historical_data[-1]] + list(intervals[lower_key])
                        interval_upper = [historical_data[-1]] + list(intervals[upper_key])
                        
                        alpha = 0.3 if conf_level == max(conf_levels) else 0.5
                        color = self.colors['interval_80'] if conf_level >= 80 else self.colors['interval_50']
                        
                        ax.fill_between(connection_x, interval_lower, interval_upper, 
                                       alpha=alpha, color=color, 
                                       label=f'{conf_level}% Quantile Interval', zorder=1)
            
            # Handle quantile bands (new format)
            quantile_bands = {}
            for key in intervals.keys():
                if key.startswith('quantile_band_') and key.endswith('_lower'):
                    band_name = key.replace('quantile_band_', '').replace('_lower', '')
                    upper_key = f'quantile_band_{band_name}_upper'
                    if upper_key in intervals:
                        quantile_bands[band_name] = {
                            'lower': intervals[key],
                            'upper': intervals[upper_key]
                        }
            
            if quantile_bands:
                # Define colors for different bands
                band_colors = ['#ff9999', '#99ccff', '#99ff99', '#ffcc99', '#cc99ff', '#ffff99']
                
                logger.info(f"Processing {len(quantile_bands)} quantile bands")
                logger.info(f"Connection_x length: {len(connection_x)}, Forecast length: {len(forecast)}")
                
                for i, (band_name, band_data) in enumerate(sorted(quantile_bands.items())):
                    color = band_colors[i % len(band_colors)]
                    alpha = 0.3 + (0.2 * (1 - i / max(1, len(quantile_bands) - 1)))  # Vary alpha
                    
                    # Ensure quantile band data matches forecast length
                    lower_values = band_data['lower']
                    upper_values = band_data['upper']
                    
                    logger.info(f"Band {band_name}: lower length={len(lower_values)}, upper length={len(upper_values)}")
                    
                    # Truncate or pad to match forecast length
                    if len(lower_values) > len(forecast):
                        lower_values = lower_values[:len(forecast)]
                        upper_values = upper_values[:len(forecast)]
                        logger.info(f"Truncated band {band_name} to forecast length")
                    elif len(lower_values) < len(forecast):
                        # Pad with last value if too short
                        last_lower = lower_values[-1] if lower_values else 0
                        last_upper = upper_values[-1] if upper_values else 0
                        lower_values = list(lower_values) + [last_lower] * (len(forecast) - len(lower_values))
                        upper_values = list(upper_values) + [last_upper] * (len(forecast) - len(upper_values))
                        logger.info(f"Padded band {band_name} to forecast length")
                    
                    interval_lower = [historical_data[-1]] + list(lower_values)
                    interval_upper = [historical_data[-1]] + list(upper_values)
                    
                    logger.info(f"Final interval lengths: lower={len(interval_lower)}, upper={len(interval_upper)}, connection_x={len(connection_x)}")
                    
                    label_key = f'quantile_band_{band_name}_label'
                    label_text = intervals.get(label_key, f'Quantile Band {int(band_name)+1}')
                    
                    ax.fill_between(connection_x, interval_lower, interval_upper, 
                                   alpha=alpha, color=color, 
                                   label=label_text, zorder=1)
        
        # Plot forecast line
        ax.plot(connection_x, connection_forecast, 
               color=self.colors['forecast'], linestyle='--', linewidth=2.5,
               label='Point Forecast', zorder=4)
        
        # Plot actual future data if available
        if actual_future is not None:
            actual_connection = [historical_data[-1]] + list(actual_future)
            ax.plot(connection_x, actual_connection, 
                   color=self.colors['actual'], linewidth=3, 
                   marker='o', markersize=6, markeredgecolor='white',
                   markeredgewidth=1, label='Actual Future', zorder=6)
        
        # Add forecast start line
        forecast_start = historical_x[-1] if dates_historical else len(historical_data) - 1
        ax.axvline(x=forecast_start, color='gray', linestyle=':', 
                  alpha=0.7, linewidth=1.5, label='Forecast Start')
        
        # Styling
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel(target_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['grid'])
        
        # Legend
        legend = ax.legend(loc='upper left', fontsize=12, frameon=True,
                          fancybox=True, shadow=True, framealpha=0.95)
        legend.get_frame().set_facecolor('white')
        
        # Format axes
        ax.tick_params(labelsize=12)
        
        # Format dates if using datetime
        if dates_historical is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', va='bottom', 
                fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Plot saved to: {save_path}")
        
        logger.info("✅ Forecast visualization completed")
        return fig
    
    def plot_forecast_with_covariates(
        self,
        historical_data: Union[List[float], np.ndarray],
        forecast: Union[List[float], np.ndarray],
        covariates_data: Dict[str, Dict[str, Union[List[float], float, str]]],
        intervals: Optional[Dict[str, np.ndarray]] = None,
        actual_future: Optional[Union[List[float], np.ndarray]] = None,
        dates_historical: Optional[List[Union[str, datetime]]] = None,
        dates_future: Optional[List[Union[str, datetime]]] = None,
        title: str = "TimesFM Forecast with Covariates Analysis",
        target_name: str = "Target Value",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive visualization with main forecast and covariates subplots.
        
        Args:
            historical_data: Historical time series data
            forecast: Point forecast values
            covariates_data: Dictionary containing covariates information
            intervals: Optional prediction intervals
            actual_future: Optional actual future values
            dates_historical: Optional historical dates
            dates_future: Optional future dates
            title: Main plot title
            target_name: Name of target variable
            save_path: Optional save path
            
        Returns:
            Matplotlib Figure object
        """
        logger.info(f"Creating comprehensive forecast with covariates: {title}")
        
        # Count covariates for subplot layout
        num_covariates = len([k for k, v in covariates_data.items() 
                             if isinstance(v, dict) and 'historical' in v])
        
        # Create subplot layout
        if num_covariates == 0:
            return self.plot_forecast_with_intervals(
                historical_data, forecast, intervals, actual_future,
                dates_historical, dates_future, title, target_name, save_path
            )
        
        # Determine grid layout
        if num_covariates <= 2:
            rows, cols = 2, 2
            height_ratios = [3, 1]
        elif num_covariates <= 4:
            rows, cols = 3, 2  
            height_ratios = [3, 1, 1]
        else:
            rows, cols = 4, 2
            height_ratios = [3, 1, 1, 1]
        
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(rows, cols, height_ratios=height_ratios, 
                             hspace=0.35, wspace=0.25)
        
        # Main forecast plot (top row, full width)
        ax_main = fig.add_subplot(gs[0, :])
        
        # Convert data
        historical_data = np.array(historical_data)
        forecast = np.array(forecast)
        
        # Setup time axes
        if dates_historical is None:
            historical_x = np.arange(len(historical_data))
            future_x = np.arange(len(historical_data), len(historical_data) + len(forecast))
        else:
            historical_x = pd.to_datetime(dates_historical)
            future_x = pd.to_datetime(dates_future) if dates_future else None
        
        # Plot main forecast (similar to single plot method)
        ax_main.set_facecolor(self.colors['background'])
        ax_main.plot(historical_x, historical_data, 
                    color=self.colors['historical'], linewidth=2.5, 
                    label='Historical Data', zorder=5)
        
        # Forecast with seamless connection
        if dates_historical is None:
            connection_x = [len(historical_data) - 1] + list(future_x)
        else:
            connection_x = [historical_x[-1]] + list(future_x)
        connection_forecast = [historical_data[-1]] + list(forecast)
        
        # Plot intervals if available
        if intervals:
            for key in intervals.keys():
                if key.startswith('lower_'):
                    conf_level = key.split('_')[1]
                    upper_key = f'upper_{conf_level}'
                    if upper_key in intervals:
                        interval_lower = [historical_data[-1]] + list(intervals[key])
                        interval_upper = [historical_data[-1]] + list(intervals[upper_key])
                        
                        alpha = 0.3 if int(conf_level) >= 80 else 0.5
                        color = self.colors['interval_80'] if int(conf_level) >= 80 else self.colors['interval_50']
                        
                        ax_main.fill_between(connection_x, interval_lower, interval_upper,
                                           alpha=alpha, color=color, 
                                           label=f'{conf_level}% Prediction Interval')
        
        ax_main.plot(connection_x, connection_forecast, 
                    color=self.colors['forecast'], linestyle='--', linewidth=2.5,
                    label='Point Forecast', zorder=4)
        
        # Plot actual future if available
        if actual_future is not None:
            actual_future = np.array(actual_future)
            actual_connection = [historical_data[-1]] + list(actual_future)
            ax_main.plot(connection_x, actual_connection, 
                        color=self.colors['actual'], linewidth=3,
                        marker='o', markersize=6, markeredgecolor='white',
                        markeredgewidth=1, label='Actual Future', zorder=6)
        
        # Forecast start line
        forecast_start = historical_x[-1] if dates_historical else len(historical_data) - 1
        ax_main.axvline(x=forecast_start, color='gray', linestyle=':', 
                       alpha=0.7, linewidth=1.5, label='Forecast Start')
        
        # Main plot styling
        ax_main.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax_main.set_ylabel(target_name, fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3, color=self.colors['grid'])
        ax_main.tick_params(labelsize=12)
        legend = ax_main.legend(loc='upper left', fontsize=12, frameon=True)
        legend.get_frame().set_facecolor('white')
        
        # Create covariate subplots
        covariate_colors = ['#9467bd', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']
        
        plot_idx = 0
        for cov_name, cov_data in covariates_data.items():
            if not isinstance(cov_data, dict) or 'historical' not in cov_data:
                continue
                
            if plot_idx >= (rows - 1) * cols:  # Don't exceed subplot capacity
                break
                
            # Calculate subplot position
            row = 1 + plot_idx // cols
            col = plot_idx % cols
            ax_cov = fig.add_subplot(gs[row, col])
            
            color = covariate_colors[plot_idx % len(covariate_colors)]
            
            # Plot historical covariate data
            ax_cov.plot(historical_x, cov_data['historical'],
                       color=color, linewidth=2.5, alpha=0.8, label='Historical')
            
            # Plot future covariate data if available
            if 'future' in cov_data and future_x is not None:
                combined_data = list(cov_data['historical']) + list(cov_data['future'])
                if dates_historical is None:
                    combined_x = np.arange(len(combined_data))
                else:
                    combined_x = list(historical_x) + list(future_x)
                
                future_start_idx = len(cov_data['historical']) - 1
                ax_cov.plot(combined_x[future_start_idx:], combined_data[future_start_idx:],
                           color=color, linewidth=2.5, linestyle='--', alpha=0.9,
                           marker='s', markersize=4, label='Future')
            
            # Forecast start line
            ax_cov.axvline(x=forecast_start, color='gray', linestyle=':', alpha=0.5)
            
            # Styling
            ax_cov.set_title(f'{cov_name.replace("_", " ").title()}', 
                           fontsize=12, fontweight='bold')
            ax_cov.set_ylabel('Value', fontsize=10)
            ax_cov.grid(True, alpha=0.3, color=self.colors['grid'])
            ax_cov.tick_params(labelsize=9)
            ax_cov.legend(fontsize=8, loc='upper left')
            ax_cov.set_facecolor(self.colors['background'])
            
            plot_idx += 1
        
        # Format x-axis for dates
        if dates_historical is not None:
            for ax in fig.get_axes():
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Overall title and timestamp
        fig.suptitle('TimesFM Comprehensive Forecasting Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', va='bottom', 
                fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Comprehensive plot saved to: {save_path}")
        
        logger.info("✅ Comprehensive forecast visualization completed")
        return fig
    
    def plot_forecast_comparison(
        self,
        forecasts_dict: Dict[str, np.ndarray],
        historical_data: Union[List[float], np.ndarray],
        actual_future: Optional[Union[List[float], np.ndarray]] = None,
        title: str = "Forecast Methods Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare multiple forecasting methods in a single plot.
        
        Args:
            forecasts_dict: Dictionary of {method_name: forecast_array}
            historical_data: Historical data for context
            actual_future: Optional actual future values
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib Figure object
        """
        logger.info(f"Creating forecast comparison plot: {title}")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_facecolor(self.colors['background'])
        
        historical_data = np.array(historical_data)
        historical_x = np.arange(len(historical_data))
        
        # Plot historical data
        ax.plot(historical_x, historical_data, 
               color=self.colors['historical'], linewidth=2.5,
               label='Historical Data', zorder=5)
        
        # Plot different forecasts
        forecast_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
        
        for i, (method, forecast) in enumerate(forecasts_dict.items()):
            forecast = np.array(forecast)
            future_x = np.arange(len(historical_data), len(historical_data) + len(forecast))
            
            # Seamless connection
            connection_x = [len(historical_data) - 1] + list(future_x)
            connection_forecast = [historical_data[-1]] + list(forecast)
            
            color = forecast_colors[i % len(forecast_colors)]
            linestyle = '--' if i == 0 else '-.'
            
            ax.plot(connection_x, connection_forecast,
                   color=color, linestyle=linestyle, linewidth=2.5,
                   label=f'{method} Forecast', zorder=3)
        
        # Plot actual future if available
        if actual_future is not None:
            actual_future = np.array(actual_future)
            future_x = np.arange(len(historical_data), len(historical_data) + len(actual_future))
            connection_x = [len(historical_data) - 1] + list(future_x)
            actual_connection = [historical_data[-1]] + list(actual_future)
            
            ax.plot(connection_x, actual_connection,
                   color=self.colors['actual'], linewidth=3,
                   marker='o', markersize=6, markeredgecolor='white',
                   markeredgewidth=1, label='Actual Future', zorder=6)
        
        # Forecast start line
        ax.axvline(x=len(historical_data) - 1, color='gray', linestyle=':', 
                  alpha=0.7, linewidth=1.5, label='Forecast Start')
        
        # Styling
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Value', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(labelsize=12)
        
        # Legend
        legend = ax.legend(loc='upper left', fontsize=12, frameon=True)
        legend.get_frame().set_facecolor('white')
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', va='bottom', 
                fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Comparison plot saved to: {save_path}")
        
        logger.info("✅ Forecast comparison visualization completed")
        return fig