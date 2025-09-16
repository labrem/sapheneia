/**
 * Sapheneia TimesFM Web Application JavaScript
 * Handles user interactions, API calls, and dynamic content updates
 */

class SapheneiaTimesFM {
    constructor() {
        this.modelInitialized = false;
        this.currentData = null;
        this.currentResults = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.setupFormValidation();
    }

    bindEvents() {
        // Model configuration
        document.getElementById('modelConfigForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.initializeModel();
        });

        document.getElementById('checkpoint').addEventListener('change', (e) => {
            this.toggleLocalPathField(e.target.value === 'local');
        });

        // Data upload
        document.getElementById('uploadBtn').addEventListener('click', () => {
            this.uploadData();
        });

        document.getElementById('sampleBtn').addEventListener('click', () => {
            this.generateSampleData();
        });

        // Forecasting
        document.getElementById('forecastConfigForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.runForecast();
        });

        // Download chart
        document.getElementById('downloadChart').addEventListener('click', () => {
            this.downloadChart();
        });
    }

    setupFormValidation() {
        // Add real-time validation for numeric inputs
        const numericInputs = ['contextLen', 'horizonLen'];
        numericInputs.forEach(id => {
            const input = document.getElementById(id);
            input.addEventListener('input', (e) => {
                this.validateNumericInput(e.target);
            });
        });
    }

    validateNumericInput(input) {
        const value = parseInt(input.value);
        const min = parseInt(input.min);
        const max = parseInt(input.max);

        if (isNaN(value) || value < min || value > max) {
            input.classList.add('is-invalid');
            return false;
        } else {
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
            return true;
        }
    }

    toggleLocalPathField(show) {
        const localPathRow = document.getElementById('localPathRow');
        localPathRow.style.display = show ? 'block' : 'none';
    }

    showAlert(type, title, message) {
        const alertContainer = document.getElementById('alertContainer');
        const alertId = 'alert_' + Date.now();
        
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show slide-in" role="alert" id="${alertId}">
                <strong>${title}</strong> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        alertContainer.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }

    showLoading(title = 'Processing...', subtitle = 'Please wait while we process your request.') {
        document.getElementById('loadingText').textContent = title;
        document.getElementById('loadingSubtext').textContent = subtitle;
        
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }

    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
    }

    updateModelStatus(status, info = null) {
        const statusElement = document.getElementById('modelStatus');
        
        if (status === 'initializing') {
            statusElement.innerHTML = '<span class="badge bg-warning">Initializing...</span>';
        } else if (status === 'ready') {
            statusElement.innerHTML = '<span class="badge bg-success">Ready</span>';
            this.modelInitialized = true;
            this.updateForecastButtonState();
        } else if (status === 'error') {
            statusElement.innerHTML = '<span class="badge bg-danger">Error</span>';
            this.modelInitialized = false;
        } else {
            statusElement.innerHTML = '<span class="badge bg-secondary">Not Initialized</span>';
            this.modelInitialized = false;
        }
    }

    async initializeModel() {
        const form = document.getElementById('modelConfigForm');
        const formData = new FormData(form);
        
        const config = {
            backend: document.getElementById('backend').value,
            context_len: parseInt(document.getElementById('contextLen').value),
            horizon_len: parseInt(document.getElementById('horizonLen').value)
        };

        const checkpointValue = document.getElementById('checkpoint').value;
        if (checkpointValue === 'local') {
            config.local_path = document.getElementById('localPath').value;
            if (!config.local_path) {
                this.showAlert('danger', 'Error', 'Please provide a local model path.');
                return;
            }
        } else {
            config.checkpoint = checkpointValue;
        }

        this.updateModelStatus('initializing');
        this.showLoading('Initializing TimesFM Model', 'This may take a few minutes on first run...');

        try {
            const response = await fetch('/api/model/init', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            const result = await response.json();
            
            if (result.success) {
                this.updateModelStatus('ready', result.model_info);
                this.showAlert('success', 'Success', 'TimesFM model initialized successfully!');
                
                // Update UI with model capabilities
                if (result.model_info && result.model_info.capabilities) {
                    this.updateCapabilitiesUI(result.model_info.capabilities);
                }
            } else {
                this.updateModelStatus('error');
                this.showAlert('danger', 'Initialization Failed', result.message || 'Unknown error occurred');
            }
        } catch (error) {
            this.updateModelStatus('error');
            this.showAlert('danger', 'Network Error', 'Failed to communicate with server: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    updateCapabilitiesUI(capabilities) {
        // Update quantiles checkbox based on capability
        const quantilesCheckbox = document.getElementById('useQuantiles');
        if (!capabilities.quantile_forecasting) {
            quantilesCheckbox.disabled = true;
            quantilesCheckbox.checked = false;
            quantilesCheckbox.parentElement.title = 'Quantile forecasting not available for this model';
        }
    }

    async uploadData() {
        const fileInput = document.getElementById('dataFile');
        const file = fileInput.files[0];

        if (!file) {
            this.showAlert('warning', 'No File Selected', 'Please select a CSV file to upload.');
            return;
        }

        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showAlert('danger', 'Invalid File Type', 'Please upload a CSV file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        this.showLoading('Uploading Data', 'Processing your CSV file...');

        try {
            const response = await fetch('/api/data/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.currentData = result.data_info;
                this.showAlert('success', 'Upload Successful', 'Data uploaded and processed successfully!');
                this.displayDataInfo(result.data_info);
                this.generateDataDefinition(result.data_info.columns);
                this.updateForecastButtonState();
            } else {
                this.showAlert('danger', 'Upload Failed', result.message || 'Upload failed');
            }
        } catch (error) {
            this.showAlert('danger', 'Network Error', 'Failed to upload file: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async generateSampleData() {
        this.showLoading('Generating Sample Data', 'Creating synthetic financial data...');

        try {
            const response = await fetch('/api/data/sample');
            const result = await response.json();

            if (result.success) {
                this.currentData = result.data_info;
                this.showAlert('success', 'Sample Data Generated', 'Synthetic financial data created successfully!');
                this.displayDataInfo(result.data_info);
                this.generateDataDefinition(result.data_info.columns, true); // true for sample data
                this.updateForecastButtonState();
            } else {
                this.showAlert('danger', 'Generation Failed', result.message || 'Failed to generate sample data');
            }
        } catch (error) {
            this.showAlert('danger', 'Network Error', 'Failed to generate sample data: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayDataInfo(dataInfo) {
        const dataInfoDiv = document.getElementById('dataInfo');
        const dataDetailsDiv = document.getElementById('dataDetails');

        let html = `
            <div class="row">
                <div class="col-md-6">
                    <table class="table table-sm data-info-table">
                        <tr><th>Filename</th><td>${dataInfo.filename}</td></tr>
                        <tr><th>Shape</th><td>${dataInfo.shape[0]} rows × ${dataInfo.shape[1]} columns</td></tr>
                        ${dataInfo.date_range ? `
                            <tr><th>Date Range</th><td>${dataInfo.date_range.start} to ${dataInfo.date_range.end}</td></tr>
                            <tr><th>Total Periods</th><td>${dataInfo.date_range.periods}</td></tr>
                        ` : ''}
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Data Preview</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>
                                    ${dataInfo.columns.map(col => `<th>${col}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${dataInfo.head.slice(0, 3).map(row => `
                                    <tr>
                                        ${dataInfo.columns.map(col => `<td>${this.formatValue(row[col])}</td>`).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;

        dataDetailsDiv.innerHTML = html;
        dataInfoDiv.style.display = 'block';
        dataInfoDiv.classList.add('fade-in');
    }

    formatValue(value) {
        if (value == null) return 'null';
        if (typeof value === 'number') {
            return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
        }
        return String(value).substring(0, 20) + (String(value).length > 20 ? '...' : '');
    }

    generateDataDefinition(columns, isSampleData = false) {
        const definitionDiv = document.getElementById('dataDefinition');
        const columnsDiv = document.getElementById('columnDefinitions');

        // Default definitions for sample data
        const sampleDefaults = {
            'btc_price': 'target',
            'eth_price': 'dynamic_numerical',
            'sp500_price': 'dynamic_numerical',
            'vix_index': 'dynamic_numerical',
            'quarter': 'dynamic_categorical',
            'market_regime': 'dynamic_categorical',
            'asset_category': 'static_categorical',
            'base_volatility': 'static_numerical'
        };

        let html = '<div class="row">';

        columns.forEach((col, index) => {
            if (col === 'date') return; // Skip date column

            const defaultValue = isSampleData ? (sampleDefaults[col] || 'dynamic_numerical') : 'target';
            
            html += `
                <div class="col-md-6 col-lg-4">
                    <div class="column-definition">
                        <div class="column-name">${col}</div>
                        <select class="form-select form-select-sm mt-2" id="def_${col}">
                            <option value="target" ${defaultValue === 'target' ? 'selected' : ''}>Target (main forecast variable)</option>
                            <option value="dynamic_numerical" ${defaultValue === 'dynamic_numerical' ? 'selected' : ''}>Dynamic Numerical</option>
                            <option value="dynamic_categorical" ${defaultValue === 'dynamic_categorical' ? 'selected' : ''}>Dynamic Categorical</option>
                            <option value="static_numerical" ${defaultValue === 'static_numerical' ? 'selected' : ''}>Static Numerical</option>
                            <option value="static_categorical" ${defaultValue === 'static_categorical' ? 'selected' : ''}>Static Categorical</option>
                        </select>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        columnsDiv.innerHTML = html;
        definitionDiv.style.display = 'block';
        definitionDiv.classList.add('fade-in');
    }

    getDataDefinition() {
        const definition = {};
        const columns = this.currentData.columns.filter(col => col !== 'date');

        columns.forEach(col => {
            const select = document.getElementById(`def_${col}`);
            if (select) {
                definition[col] = select.value;
            }
        });

        return definition;
    }

    updateForecastButtonState() {
        const forecastBtn = document.getElementById('forecastBtn');
        const canForecast = this.modelInitialized && this.currentData;
        
        forecastBtn.disabled = !canForecast;
        
        if (canForecast) {
            forecastBtn.classList.remove('btn-secondary');
            forecastBtn.classList.add('btn-warning');
        } else {
            forecastBtn.classList.add('btn-secondary');
            forecastBtn.classList.remove('btn-warning');
        }
    }

    async runForecast() {
        if (!this.modelInitialized || !this.currentData) {
            this.showAlert('warning', 'Cannot Forecast', 'Please initialize model and upload data first.');
            return;
        }

        const dataDefinition = this.getDataDefinition();
        
        // Validate that at least one target column is defined
        const hasTarget = Object.values(dataDefinition).includes('target');
        if (!hasTarget) {
            this.showAlert('warning', 'No Target Variable', 'Please define at least one column as the target variable.');
            return;
        }

        const config = {
            filename: this.currentData.filename,
            data_definition: dataDefinition,
            use_covariates: document.getElementById('useCovariates').checked,
            use_quantiles: document.getElementById('useQuantiles').checked,
            context_len: parseInt(document.getElementById('contextLen').value),
            horizon_len: parseInt(document.getElementById('horizonLen').value)
        };

        this.showLoading('Running Forecast', 'TimesFM is analyzing your data and generating forecasts...');

        try {
            const response = await fetch('/api/forecast', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            const result = await response.json();

            if (result.success) {
                this.currentResults = result;
                this.showAlert('success', 'Forecast Complete', 'Forecasting completed successfully!');
                await this.displayResults(result);
                this.generateVisualization(result);
            } else {
                this.showAlert('danger', 'Forecast Failed', result.message || 'Forecasting failed');
            }
        } catch (error) {
            this.showAlert('danger', 'Network Error', 'Forecasting failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async displayResults(result) {
        const resultsCard = document.getElementById('resultsCard');
        resultsCard.style.display = 'block';
        resultsCard.classList.add('fade-in');

        // Populate summary tab
        this.displaySummary(result.forecast_summary, result.results);
        
        // Populate data tab
        this.displayDataTable(result.results);
    }

    displaySummary(summary, results) {
        const summaryDiv = document.getElementById('forecastSummary');

        const methodsCount = summary.methods_used.length;
        const mainForecast = results.enhanced_forecast || results.point_forecast;
        const avgForecast = mainForecast ? (mainForecast.reduce((a, b) => a + b, 0) / mainForecast.length) : 0;

        let html = `
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="summary-card bg-primary">
                        <div class="summary-value">${summary.context_length}</div>
                        <div class="summary-label">Context Length</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="summary-card bg-success">
                        <div class="summary-value">${summary.horizon_length}</div>
                        <div class="summary-label">Forecast Periods</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="summary-card bg-info">
                        <div class="summary-value">${methodsCount}</div>
                        <div class="summary-label">Methods Used</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="summary-card bg-warning">
                        <div class="summary-value">$${avgForecast.toLocaleString(undefined, {maximumFractionDigits: 0})}</div>
                        <div class="summary-label">Avg. Forecast</div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <h6>Configuration</h6>
                    <table class="table table-sm">
                        <tr><th>Target Variable</th><td>${summary.target_column}</td></tr>
                        <tr><th>Covariates Used</th><td>${summary.covariates_used ? 'Yes' : 'No'}</td></tr>
                        <tr><th>Methods</th><td>${summary.methods_used.join(', ')}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Forecast Methods</h6>
                    <ul class="list-group list-group-flush">
        `;

        summary.methods_used.forEach(method => {
            const methodName = method.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
            const forecast = results[method];
            const minVal = Math.min(...forecast);
            const maxVal = Math.max(...forecast);
            
            html += `
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    ${methodName}
                    <span class="badge bg-secondary">$${minVal.toLocaleString()} - $${maxVal.toLocaleString()}</span>
                </li>
            `;
        });

        html += `
                    </ul>
                </div>
            </div>
        `;

        summaryDiv.innerHTML = html;
    }

    displayDataTable(results) {
        const dataDiv = document.getElementById('forecastData');
        
        // Create table with forecast results
        const forecastLength = results.point_forecast ? results.point_forecast.length : 
                              results.enhanced_forecast ? results.enhanced_forecast.length : 0;

        if (forecastLength === 0) {
            dataDiv.innerHTML = '<p>No forecast data available.</p>';
            return;
        }

        let html = `
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Period</th>
        `;

        Object.keys(results).forEach(method => {
            if (method !== 'prediction_intervals') {
                const methodName = method.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                html += `<th>${methodName}</th>`;
            }
        });

        html += `</tr></thead><tbody>`;

        for (let i = 0; i < forecastLength; i++) {
            html += `<tr><td>${i + 1}</td>`;
            
            Object.entries(results).forEach(([method, values]) => {
                if (method !== 'prediction_intervals' && Array.isArray(values)) {
                    const value = values[i];
                    html += `<td>$${value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>`;
                }
            });
            
            html += '</tr>';
        }

        html += '</tbody></table></div>';
        dataDiv.innerHTML = html;
    }

    async generateVisualization(result) {
        this.showLoading('Generating Chart', 'Creating professional forecast visualization...');

        try {
            const response = await fetch('/api/visualize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    visualization_data: result.visualization_data,
                    results: result.results
                })
            });

            const vizResult = await response.json();

            if (vizResult.success) {
                const chartImg = document.getElementById('forecastChart');
                chartImg.src = 'data:image/png;base64,' + vizResult.image;
                chartImg.style.display = 'block';
            } else {
                this.showAlert('warning', 'Visualization Failed', vizResult.message || 'Failed to generate chart');
            }
        } catch (error) {
            this.showAlert('danger', 'Chart Error', 'Failed to generate visualization: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    downloadChart() {
        const chartImg = document.getElementById('forecastChart');
        if (!chartImg.src) {
            this.showAlert('warning', 'No Chart', 'No chart available for download.');
            return;
        }

        const link = document.createElement('a');
        link.download = `sapheneia_forecast_${new Date().toISOString().slice(0, 10)}.png`;
        link.href = chartImg.src;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        this.showAlert('success', 'Download Started', 'Chart download has started.');
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.sapheneiaApp = new SapheneiaTimesFM();
});