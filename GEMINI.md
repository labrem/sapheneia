
# GEMINI Project Context

This document provides an overview of the Sapheneia TimesFM project for the Gemini AI agent.

## Project Overview

This is a Python-based project for financial forecasting using Google's TimesFM (Time Series Foundation Model). The project is structured as a comprehensive toolkit that includes a core library, research notebooks, and a web application.

**Main Technologies:**

*   **Core:** Python, Google TimesFM
*   **ML/Scientific:** JAX, PyTorch, Pandas, NumPy, Scikit-learn
*   **Web:** Flask
*   **Visualization:** Matplotlib, Seaborn
*   **Package Management:** uv

**Architecture:**

*   `src/`: A Python library containing the core logic for:
    *   `model.py`: Loading and configuring the TimesFM model.
    *   `data.py`: Processing and validating input data and covariates.
    *   `forecast.py`: Performing forecasts (point, quantile, with covariates).
    *   `visualization.py`: Generating professional plots.
*   `notebooks/`: Jupyter notebooks for research, demos, and experimentation.
*   `webapp/`: A Flask-based web application providing a UI for the forecasting tools.
*   `data/`: Directory for storing CSV data files.

## Building and Running

The project uses a `setup.sh` script to automate the setup process.

**Key Commands:**

*   **Full Setup (Local + Webapp):**
    ```bash
    ./setup.sh
    ```
*   **Activate Virtual Environment:**
    ```bash
    source .venv/bin/activate
    ```
*   **Run Jupyter Notebook:**
    ```bash
    uv run jupyter notebook
    ```
*   **Run the Web Application:**
    ```bash
    cd webapp
    python app.py
    ```

## Development Conventions

*   **Package Management:** The project uses `uv` for package management. Dependencies are listed in `pyproject.toml` and `webapp/requirements.txt`.
*   **Code Style:** The Python code is well-structured with type hints and docstrings. It follows a modular approach, separating concerns into different files.
*   **Logging:** The project uses the `logging` module to provide informative output during execution.
*   **Testing:** While no dedicated test files are present in the `src` directory, the `setup.sh` script includes a verification step that performs a basic test of the TimesFM model.
*   **Notebooks:** The `notebooks` directory is organized by researcher, suggesting a collaborative research environment.
*   **Web Application:** The web application is a standard Flask application with templates, static assets, and a main `app.py` file.
