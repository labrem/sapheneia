# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sapheneia is a TimesFM (Google's Time Series Foundation Model) research and experimentation repository focused on financial forecasting and time series analysis with covariates support. The project uses TimesFM for making predictions with exogenous variables and includes comprehensive utilities for financial data analysis.

## Environment Setup

The repository uses UV package manager for Python dependency management and virtual environments. Always use the automated setup script:

```bash
# Make script executable and run
chmod +x setup_environment.sh
./setup_environment.sh
```

This script automatically:
- Detects system architecture (Apple Silicon, ARM64, x86_64)
- Installs UV package manager if needed
- Sets up Python 3.10/3.11 virtual environment
- Installs TimesFM with appropriate backend (PAX for x86_64, PyTorch for ARM/Apple Silicon)
- Installs JAX/JAXlib for covariates functionality
- Installs Jupyter notebook support
- Verifies installation completeness

## Common Commands

### Environment Management
```bash
# Activate virtual environment
source .venv/bin/activate

# Run Python scripts with UV
uv run python your_script.py

# Start Jupyter notebook server
uv run jupyter notebook

# Test TimesFM installation
uv run python -c "import timesfm; print('TimesFM ready!')"
```

### Development Workflow
```bash
# Install additional dependencies
uv pip install package_name

# Update dependencies
uv pip install --upgrade package_name

# Run specific notebook
uv run jupyter notebook notebooks/marcelo/timesfm_covariates.ipynb
```

## Code Architecture

### Directory Structure
- `notebooks/` - Individual researcher notebooks organized by author
  - `marcelo/` - Marcelo's TimesFM experiments and covariates research
  - `lucas/` - Lucas's financial forecasting demos
  - `matt/` - Matt's TimesFM financial analysis
  - `matthieu/` - Matthieu's research notebooks
- `data/` - Dataset storage and intermediate results
- `setup_environment.sh` - Automated environment setup script
- `pyproject.toml` - Project dependencies and metadata
- `uv.lock` - Locked dependency versions

### Key Components

#### TimesFM Integration
The project uses Google's TimesFM 2.0-500m model with two possible backends:
- **PAX version** (preferred for x86_64): `google/timesfm-2.0-500m-jax`
- **PyTorch version** (ARM/Apple Silicon): `google/timesfm-2.0-500m-pytorch`

Model initialization follows this pattern:
```python
model = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",  # or "gpu"
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=50,  # Must match checkpoint
        use_positional_embedding=False,
        context_len=2048,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    ),
)
```

#### Covariates Support
The project implements advanced covariates functionality with TimesFM:
- **Static covariates**: Per time series (e.g., category, base_price)
- **Dynamic covariates**: Per timestamp (e.g., weekday, temperature)
- **Numerical and categorical** covariate types supported
- **XReg modes**: "timesfm + xreg" or "xreg + timesfm" strategies

#### Financial Forecasting Utilities
Located in `notebooks/lucas/forecast_utils.py` and `notebooks/matt/forecast_utils.py`:
- Synthetic financial data generation with realistic correlations
- Professional visualization with prediction intervals
- Bootstrap sampling for uncertainty quantification
- Multi-asset forecasting (Bitcoin, Ethereum, S&P 500, VIX)

### Dependencies
Core dependencies managed in `pyproject.toml`:
- `jax>=0.7.0` and `jaxlib>=0.7.0` - JAX framework for covariates
- `timesfm>=1.3.0` - Google's TimesFM model
- `jupyter>=1.1.1` - Notebook environment
- `matplotlib>=3.10.5`, `seaborn>=0.13.2` - Visualization

## Model Configuration Guidelines

### TimesFM Parameters
- **num_layers**: Must be 50 for the 2.0-500m checkpoint
- **context_len**: Use 512-2048 depending on use case
- **horizon_len**: Forecast length (typically 4-128 periods)
- **backend**: "cpu" for development, "gpu"/"tpu" for production

### Covariates Configuration
When using `forecast_with_covariates()`:
- Dynamic covariates must cover context + horizon periods
- Use `xreg_mode="xreg + timesfm"` for best results
- Set `normalize_xreg_target_per_input=True` for stability
- Apply ridge regression (`ridge=0.0` to small positive values)

## Testing and Verification

The setup script includes comprehensive verification:
- TimesFM import and initialization tests
- API method availability checks
- Jupyter components verification
- Basic forecasting functionality test

Run manual verification:
```bash
uv run python -c "
import timesfm
import pandas as pd
import numpy as np
hparams = timesfm.TimesFmHparams(backend='cpu', per_core_batch_size=1, horizon_len=24)
print('TimesFM verification successful!')
"
```

## Project Context

This is a research repository focused on advancing TimesFM capabilities for financial forecasting. Each researcher maintains their own notebook directory with experiments ranging from basic TimesFM usage to advanced covariates integration and professional financial analysis workflows.

The automated environment setup ensures consistent development environments across different system architectures, with appropriate TimesFM backend selection and full covariates support through JAX integration.