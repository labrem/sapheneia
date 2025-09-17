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

        document.getElementById('modelSource').addEventListener('change', (e) => {
            this.toggleModelSourceFields(e.target.value);
        });

        // Data upload
        document.getElementById('uploadBtn').addEventListener('click', () => {
            this.uploadData();
        });

        // Sample data generation removed as per requirements

        // Forecasting
        document.getElementById('forecastConfigForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.runForecast();
        });

        // Quantile selector is now always visible, no toggle needed

        // Download chart
        document.getElementById('downloadChart').addEventListener('click', () => {
            this.downloadChart();
        });

        // Download data
        document.getElementById('downloadData').addEventListener('click', () => {
            this.downloadData();
        });

        // Refresh plot
        document.getElementById('refreshPlot').addEventListener('click', () => {
            if (this.currentResults) {
                this.generateVisualization(this.currentResults);
            } else {
                this.showAlert('warning', 'No Results', 'Run a forecast first.');
            }
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

    toggleModelSourceFields(source) {
        const huggingfaceRow = document.getElementById('huggingfaceRow');
        const localPathRow = document.getElementById('localPathRow');
        
        if (source === 'local') {
            huggingfaceRow.style.display = 'none';
            localPathRow.style.display = 'block';
        } else {
            huggingfaceRow.style.display = 'block';
            localPathRow.style.display = 'none';
        }
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
        const loadingTextElement = document.getElementById('loadingText');
        const loadingSubtextElement = document.getElementById('loadingSubtext');
        const modalElement = document.getElementById('loadingModal');
        
        if (!loadingTextElement || !loadingSubtextElement || !modalElement) {
            console.error('Loading modal elements not found:', {
                loadingText: !!loadingTextElement,
                loadingSubtext: !!loadingSubtextElement, 
                loadingModal: !!modalElement
            });
            return;
        }
        
        loadingTextElement.textContent = title;
        loadingSubtextElement.textContent = subtitle;
        
        const modal = new bootstrap.Modal(modalElement);
        modal.show();
    }

    hideLoading() {
        console.log('hideLoading() called');
        
        // Multiple approaches to ensure modal is hidden
        const modalElement = document.getElementById('loadingModal');
        const modal = bootstrap.Modal.getInstance(modalElement);
        
        console.log('Modal instance:', modal);
        console.log('Modal element classes:', modalElement?.classList.toString());
        
        // Approach 1: Try Bootstrap's hide method
        if (modal) {
            modal.hide();
            console.log('Called Bootstrap modal.hide()');
        }
        
        // Approach 2: Force hide with a slight delay to ensure Bootstrap completes
        setTimeout(() => {
            console.log('Force hiding modal after timeout...');
            console.log('Modal element before force hide:', modalElement);
            console.log('Modal is visible?', modalElement?.offsetParent !== null);
            
            if (modalElement) {
                // Remove Bootstrap classes and attributes more aggressively
                modalElement.classList.remove('show', 'd-block');
                modalElement.classList.add('d-none');
                modalElement.setAttribute('aria-hidden', 'true');
                modalElement.removeAttribute('aria-modal');
                modalElement.removeAttribute('role');
                modalElement.style.display = 'none !important';
                
                // Also hide the modal dialog
                const modalDialog = modalElement.querySelector('.modal-dialog');
                if (modalDialog) {
                    modalDialog.style.display = 'none';
                }
                
                // Clean up body classes and attributes
                document.body.classList.remove('modal-open');
                document.body.style.overflow = '';
                document.body.style.paddingRight = '';
                
                // Remove backdrop more precisely - only actual modal backdrops
                const backdrops = document.querySelectorAll('.modal-backdrop');
                console.log('Found modal backdrops:', backdrops.length);
                backdrops.forEach((element, index) => {
                    console.log(`Removing backdrop ${index}:`, element);
                    element.remove();
                });
                
                // Double-check for any remaining modal-related elements (but don't remove them)
                const remainingModals = document.querySelectorAll('.modal.show, .modal.d-block');
                remainingModals.forEach(m => {
                    // Only hide modals, don't remove them from DOM
                    if (m !== modalElement) { // Don't double-process our target modal
                        m.classList.remove('show', 'd-block');
                        m.classList.add('d-none');
                        m.style.display = 'none';
                    }
                });
                
                console.log('Modal force hidden - classes now:', modalElement.classList.toString());
                console.log('Modal visible after force hide?', modalElement.offsetParent !== null);
                console.log('Body classes after cleanup:', document.body.classList.toString());
            }
        }, 100);
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

        const modelSource = document.getElementById('modelSource').value;
        if (modelSource === 'local') {
            config.local_path = document.getElementById('localPath').value;
            if (!config.local_path) {
                this.showAlert('danger', 'Error', 'Please provide a local model path.');
                return;
            }
        } else {
            config.checkpoint = document.getElementById('checkpoint').value;
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
        // Quantile selector is now always visible and enabled
        // No need to update any checkboxes since they're always shown
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
            console.log('Starting file upload...');
            const response = await fetch('/api/data/upload', {
                method: 'POST',
                body: formData
            });

            console.log('Response received:', response.status, response.statusText);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Get response text first to debug potential JSON parsing issues
            const responseText = await response.text();
            console.log('Raw response text:', responseText.substring(0, 200) + '...');

            let result;
            try {
                result = JSON.parse(responseText);
                console.log('Parsed result:', result);
            } catch (jsonError) {
                console.error('JSON parsing failed:', jsonError);
                console.error('Response text that failed to parse:', responseText);
                throw new Error(`Failed to parse JSON response: ${jsonError.message}`);
            }

            if (result.success) {
                console.log('Processing successful result...');
                try {
                    this.currentData = result.data_info;
                    console.log('Set currentData:', this.currentData);
                    
                    this.showAlert('success', 'Upload Successful', 'Data uploaded and processed successfully!');
                    console.log('Showed success alert');
                    
                    this.displayDataInfo(result.data_info);
                    console.log('Displayed data info');
                    
                    this.generateDataDefinition(result.data_info.columns);
                    console.log('Generated data definition');
                    
                    this.updateForecastButtonState();
                    console.log('Updated forecast button state');
                    
                } catch (processingError) {
                    console.error('Error processing successful result:', processingError);
                    this.showAlert('danger', 'Processing Error', `Error processing upload result: ${processingError.message}`);
                }
            } else {
                console.log('Upload failed, showing error:', result.message);
                this.showAlert('danger', 'Upload Failed', result.message || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert('danger', 'Network Error', 'Failed to upload file: ' + error.message);
        } finally {
            console.log('Upload process completed');
            this.hideLoading();
        }
    }

    // Sample data generation removed as per requirements

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

        let html = '<div class="row">';

        columns.forEach((col, index) => {
            if (col === 'date') return; // Skip date column

            // Default to target for first non-date column, others as dynamic_numerical
            const defaultValue = index === 1 ? 'target' : 'dynamic_numerical';
            
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
            context_len: parseInt(document.getElementById('contextLen').value),
            horizon_len: parseInt(document.getElementById('horizonLen').value)
        };
        
        // Attach user-selected quantile ticks (always on; default [1,9])
        let ticks = Array.from(document.querySelectorAll('.quantile-tick'))
            .filter(cb => cb.checked)
            .map(cb => parseInt(cb.value))
            .sort((a,b) => a - b);
        if (ticks.length < 2) {
            ticks = [1, 9];
        }
        config.quantile_indices = ticks;

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
                    results: result.results,
                    quantile_indices: (function(){
                        let t = Array.from(document.querySelectorAll('.quantile-tick'))
                            .filter(cb => cb.checked)
                            .map(cb => parseInt(cb.value))
                            .sort((a,b) => a-b);
                        if (t.length < 2) t = [1,9];
                        return t;
                    })()
                })
            });

            const vizResult = await response.json();

            if (vizResult.success) {
                const chartImg = document.getElementById('forecastChart');
                // Scroll to results after the image has fully rendered
                try {
                    chartImg.onload = () => {
                        const resultsCard = document.getElementById('resultsCard');
                        if (resultsCard) {
                            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                        // Clean up the onload handler to avoid duplicate triggers
                        chartImg.onload = null;
                    };
                } catch (e) {
                    // Fallback: delayed scroll
                    setTimeout(() => {
                        const resultsCard = document.getElementById('resultsCard');
                        if (resultsCard) {
                            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                    }, 150);
                }
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

    downloadData() {
        if (!this.currentResults || !this.currentResults.results) {
            this.showAlert('warning', 'No Data', 'No forecast data available for download.');
            return;
        }

        const results = this.currentResults.results;
        const forecastLength = results.point_forecast ? results.point_forecast.length : 
                              results.enhanced_forecast ? results.enhanced_forecast.length : 0;

        if (forecastLength === 0) {
            this.showAlert('warning', 'No Data', 'No forecast data available for download.');
            return;
        }

        // Create CSV content
        let csvContent = 'Period,Point_Forecast';
        
        // Add quantile columns
        const quantileKeys = Object.keys(results).filter(key => 
            key.includes('quantile_band_') && key.endsWith('_lower')
        );
        
        // Sort quantile bands by band number
        const sortedQuantileKeys = quantileKeys.sort((a, b) => {
            const aNum = parseInt(a.match(/quantile_band_(\d+)_lower/)[1]);
            const bNum = parseInt(b.match(/quantile_band_(\d+)_lower/)[1]);
            return aNum - bNum;
        });

        // Add column headers for quantile bands
        sortedQuantileKeys.forEach(key => {
            const bandNum = key.match(/quantile_band_(\d+)_lower/)[1];
            csvContent += `,Quantile_Band_${parseInt(bandNum)+1}_Lower,Quantile_Band_${parseInt(bandNum)+1}_Upper`;
        });

        // Add other forecast methods
        Object.keys(results).forEach(method => {
            if (!method.includes('quantile_band_') && method !== 'point_forecast' && 
                method !== 'enhanced_forecast' && Array.isArray(results[method])) {
                const methodName = method.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                csvContent += `,${methodName}`;
            }
        });

        csvContent += '\n';

        // Add data rows
        for (let i = 0; i < forecastLength; i++) {
            csvContent += `${i + 1}`;
            
            // Point forecast
            const pointForecast = results.enhanced_forecast || results.point_forecast;
            csvContent += `,${pointForecast[i]}`;
            
            // Quantile bands
            sortedQuantileKeys.forEach(key => {
                const upperKey = key.replace('_lower', '_upper');
                const lowerValues = results[key];
                const upperValues = results[upperKey];
                
                if (lowerValues && upperValues && lowerValues[i] !== undefined && upperValues[i] !== undefined) {
                    csvContent += `,${lowerValues[i]},${upperValues[i]}`;
                } else {
                    csvContent += `,,`;
                }
            });
            
            // Other forecast methods
            Object.entries(results).forEach(([method, values]) => {
                if (!method.includes('quantile_band_') && method !== 'point_forecast' && 
                    method !== 'enhanced_forecast' && Array.isArray(values) && values[i] !== undefined) {
                    csvContent += `,${values[i]}`;
                }
            });
            
            csvContent += '\n';
        }

        // Create and download file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `sapheneia_forecast_data_${new Date().toISOString().slice(0, 10)}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        this.showAlert('success', 'Download Started', 'Forecast data download has started.');
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.sapheneiaApp = new SapheneiaTimesFM();
});