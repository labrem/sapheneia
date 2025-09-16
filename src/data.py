"""
Data Processing and Validation Module

This module provides comprehensive data processing capabilities for TimesFM,
including CSV loading, covariate preparation, and data validation.

Key Features:
- CSV data loading with flexible column configuration
- Automatic data type inference and conversion
- Covariates data preparation and validation
- Data structure formatting for TimesFM input requirements
- Support for dynamic and static, numerical and categorical covariates
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data loading, processing, and validation for TimesFM forecasting.
    
    This class provides methods to load CSV data, process covariates according
    to TimesFM requirements, and validate data structures before forecasting.
    
    Example:
        >>> processor = DataProcessor()
        >>> data = processor.load_csv_data("data.csv", data_definition)
        >>> inputs, covariates = processor.prepare_forecast_data(
        ...     data, context_len=100, horizon_len=24
        ... )
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        self.data = None
        self.data_definition = None
        self.processed_data = None
        
    def load_csv_data(
        self, 
        csv_file_path: str, 
        data_definition: Union[str, Dict[str, str]]
    ) -> pd.DataFrame:
        """
        Load CSV data with proper column type conversion based on data definition.
        
        Args:
            csv_file_path: Path to the CSV file
            data_definition: Either JSON file path or dictionary defining column types
            
        Returns:
            Loaded and processed DataFrame
            
        Raises:
            FileNotFoundError: If CSV or JSON file not found
            ValueError: If data definition is invalid
        """
        logger.info(f"Loading CSV data from: {csv_file_path}")
        
        # Load data definition
        if isinstance(data_definition, str):
            with open(data_definition, 'r') as f:
                self.data_definition = json.load(f)
        else:
            self.data_definition = data_definition.copy()
        
        logger.info(f"Data definition: {self.data_definition}")
        
        # Load CSV
        try:
            self.data = pd.read_csv(csv_file_path)
            logger.info(f"Loaded CSV with shape: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        # Validate that 'date' column exists
        if 'date' not in self.data.columns:
            raise ValueError("CSV file must contain a 'date' column as the first column")
        
        # Convert date column
        self.data['date'] = pd.to_datetime(self.data['date'])
        logger.info(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        
        # Apply data type conversions based on definition
        self._apply_data_types()
        
        # Validate data definition
        self._validate_data_definition()
        
        logger.info("✅ CSV data loaded and processed successfully")
        return self.data.copy()
    
    def _apply_data_types(self) -> None:
        """Apply appropriate data types based on the data definition."""
        logger.info("Applying data type conversions...")
        
        for column, data_type in self.data_definition.items():
            if column not in self.data.columns:
                logger.warning(f"Column '{column}' in data definition not found in CSV")
                continue
            
            try:
                if data_type in ['target', 'dynamic_numerical', 'static_numerical']:
                    # Convert to float
                    self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
                    logger.info(f"Converted '{column}' to numerical (float)")
                    
                elif data_type in ['dynamic_categorical', 'static_categorical']:
                    # Convert to string
                    self.data[column] = self.data[column].astype(str)
                    logger.info(f"Converted '{column}' to categorical (string)")
                    
                else:
                    logger.warning(f"Unknown data type '{data_type}' for column '{column}'")
                    
            except Exception as e:
                logger.error(f"Failed to convert column '{column}': {str(e)}")
                raise
    
    def _validate_data_definition(self) -> None:
        """Validate the data definition against the loaded data."""
        logger.info("Validating data definition...")
        
        # Check for required data types
        target_columns = [col for col, dtype in self.data_definition.items() if dtype == 'target']
        if not target_columns:
            raise ValueError("Data definition must contain at least one 'target' column")
        
        if len(target_columns) > 1:
            logger.warning(f"Multiple target columns found: {target_columns}. Using first one for univariate forecasting.")
        
        # Validate column existence
        missing_columns = set(self.data_definition.keys()) - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"Columns defined in data_definition but missing from CSV: {missing_columns}")
        
        # Check for data quality issues
        for column in target_columns:
            if self.data[column].isnull().any():
                null_count = self.data[column].isnull().sum()
                logger.warning(f"Target column '{column}' has {null_count} null values")
        
        logger.info("✅ Data definition validation passed")
    
    def prepare_forecast_data(
        self,
        data: pd.DataFrame,
        context_len: int,
        horizon_len: int,
        target_column: Optional[str] = None
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Prepare data for TimesFM forecasting with covariates.
        
        Args:
            data: Input DataFrame
            context_len: Length of context window for forecasting
            horizon_len: Length of forecast horizon
            target_column: Target column name (auto-detected if None)
            
        Returns:
            Tuple of (target_inputs, covariates_dict)
            
        Raises:
            ValueError: If insufficient data or invalid configuration
        """
        logger.info(f"Preparing forecast data (context: {context_len}, horizon: {horizon_len})")
        
        # Auto-detect target column if not specified
        if target_column is None:
            target_columns = [col for col, dtype in self.data_definition.items() if dtype == 'target']
            if not target_columns:
                raise ValueError("No target column found in data definition")
            target_column = target_columns[0]
            logger.info(f"Using target column: {target_column}")
        
        # Validate data length
        total_required = context_len + horizon_len
        if len(data) < total_required:
            raise ValueError(f"Insufficient data: need {total_required} points, have {len(data)}")
        
        # Prepare target inputs (context only)
        target_series = data[target_column].values
        target_inputs = target_series[:context_len].tolist()
        
        # Prepare covariates
        covariates = self._prepare_covariates(data, context_len, horizon_len)
        
        logger.info(f"✅ Prepared forecast data:")
        logger.info(f"  Target inputs length: {len(target_inputs)}")
        logger.info(f"  Target range: {min(target_inputs):.2f} - {max(target_inputs):.2f}")
        logger.info(f"  Covariates: {list(covariates.keys())}")
        
        return target_inputs, covariates
    
    def _prepare_covariates(
        self, 
        data: pd.DataFrame, 
        context_len: int, 
        horizon_len: int
    ) -> Dict[str, Dict[str, List]]:
        """
        Prepare covariates data structure for TimesFM.
        
        Args:
            data: Input DataFrame
            context_len: Context window length
            horizon_len: Forecast horizon length
            
        Returns:
            Dictionary containing organized covariates
        """
        covariates = {
            'dynamic_numerical_covariates': {},
            'dynamic_categorical_covariates': {},
            'static_numerical_covariates': {},
            'static_categorical_covariates': {}
        }
        
        total_len = context_len + horizon_len
        
        for column, data_type in self.data_definition.items():
            if column == 'date' or data_type == 'target':
                continue
            
            if data_type == 'dynamic_numerical':
                # Dynamic numerical: need context + horizon values
                if len(data) < total_len:
                    logger.warning(f"Insufficient data for dynamic covariate '{column}': need {total_len}, have {len(data)}")
                    continue
                
                values = data[column].values[:total_len].tolist()
                covariates['dynamic_numerical_covariates'][column] = [values]
                logger.info(f"Added dynamic numerical covariate '{column}': {len(values)} values")
                
            elif data_type == 'dynamic_categorical':
                # Dynamic categorical: need context + horizon values
                if len(data) < total_len:
                    logger.warning(f"Insufficient data for dynamic covariate '{column}': need {total_len}, have {len(data)}")
                    continue
                
                values = data[column].values[:total_len].tolist()
                covariates['dynamic_categorical_covariates'][column] = [values]
                logger.info(f"Added dynamic categorical covariate '{column}': {len(values)} values")
                
            elif data_type == 'static_numerical':
                # Static numerical: single value per time series
                value = float(data[column].iloc[0])
                covariates['static_numerical_covariates'][column] = [value]
                logger.info(f"Added static numerical covariate '{column}': {value}")
                
            elif data_type == 'static_categorical':
                # Static categorical: single value per time series
                value = str(data[column].iloc[0])
                covariates['static_categorical_covariates'][column] = [value]
                logger.info(f"Added static categorical covariate '{column}': {value}")
        
        # Remove empty covariate types
        covariates = {k: v for k, v in covariates.items() if v}
        
        return covariates
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dictionary containing data summary statistics
        """
        if self.data is None:
            return {"status": "No data loaded"}
        
        summary = {
            "status": "loaded",
            "shape": self.data.shape,
            "date_range": {
                "start": str(self.data['date'].min().date()),
                "end": str(self.data['date'].max().date()),
                "total_periods": len(self.data)
            },
            "columns": list(self.data.columns),
            "data_definition": self.data_definition
        }
        
        # Add column-specific statistics
        column_stats = {}
        for column in self.data.columns:
            if column == 'date':
                continue
                
            col_data = self.data[column]
            data_type = self.data_definition.get(column, 'unknown')
            
            if data_type in ['target', 'dynamic_numerical', 'static_numerical']:
                column_stats[column] = {
                    "type": data_type,
                    "dtype": str(col_data.dtype),
                    "min": float(col_data.min()) if not col_data.isnull().all() else None,
                    "max": float(col_data.max()) if not col_data.isnull().all() else None,
                    "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
                    "null_count": int(col_data.isnull().sum())
                }
            else:
                column_stats[column] = {
                    "type": data_type,
                    "dtype": str(col_data.dtype),
                    "unique_values": int(col_data.nunique()),
                    "null_count": int(col_data.isnull().sum()),
                    "sample_values": col_data.dropna().unique()[:5].tolist()
                }
        
        summary["column_statistics"] = column_stats
        return summary
    
    def validate_forecast_inputs(
        self, 
        inputs: List[float], 
        covariates: Dict[str, Any], 
        context_len: int, 
        horizon_len: int
    ) -> bool:
        """
        Validate that forecast inputs are properly formatted for TimesFM.
        
        Args:
            inputs: Target time series inputs
            covariates: Covariates dictionary
            context_len: Expected context length
            horizon_len: Expected horizon length
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating forecast inputs...")
        
        # Validate inputs length
        if len(inputs) != context_len:
            raise ValueError(f"Input length {len(inputs)} doesn't match context_len {context_len}")
        
        # Validate inputs are numeric
        if not all(isinstance(x, (int, float)) and not np.isnan(x) for x in inputs):
            raise ValueError("All inputs must be numeric and non-NaN")
        
        # Validate covariates structure
        total_len = context_len + horizon_len
        
        for cov_type, cov_dict in covariates.items():
            if cov_type in ['dynamic_numerical_covariates', 'dynamic_categorical_covariates']:
                for name, values_list in cov_dict.items():
                    if len(values_list) != 1:
                        raise ValueError(f"Dynamic covariate '{name}' must have exactly 1 time series")
                    if len(values_list[0]) != total_len:
                        raise ValueError(f"Dynamic covariate '{name}' must have {total_len} values, got {len(values_list[0])}")
            
            elif cov_type in ['static_numerical_covariates', 'static_categorical_covariates']:
                for name, values_list in cov_dict.items():
                    if len(values_list) != 1:
                        raise ValueError(f"Static covariate '{name}' must have exactly 1 value")
        
        logger.info("✅ Forecast inputs validation passed")
        return True
    
    def create_sample_data_definition(self, output_path: str) -> None:
        """
        Create a sample data definition JSON file.
        
        Args:
            output_path: Path where to save the sample JSON file
        """
        sample_definition = {
            "btc_price": "target",
            "eth_price": "dynamic_numerical", 
            "vix_index": "dynamic_numerical",
            "sp500_price": "dynamic_numerical",
            "quarter": "dynamic_categorical",
            "asset_category": "static_categorical",
            "base_price": "static_numerical"
        }
        
        with open(output_path, 'w') as f:
            json.dump(sample_definition, f, indent=2)
        
        logger.info(f"Sample data definition saved to: {output_path}")
        print(f"Sample data definition structure:")
        print(json.dumps(sample_definition, indent=2))