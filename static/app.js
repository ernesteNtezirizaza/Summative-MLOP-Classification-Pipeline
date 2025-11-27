// API Base URL - use relative path for production compatibility
const API_BASE = window.location.origin;

// Global state
let currentPage = 'home';
let selectedFiles = [];
let uploadedFile = null;

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    checkModelStatus();
    setupFileInputs();
    loadUptime();
    setInterval(loadUptime, 60000); // Update every minute
});

// Navigation
function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const page = btn.getAttribute('data-page');
            switchPage(page);
        });
    });
}

function switchPage(page) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    
    // Show selected page
    document.getElementById(page).classList.add('active');
    
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-page') === page) {
            btn.classList.add('active');
        }
    });
    
    currentPage = page;
    
    // Load page-specific data
    if (page === 'visualizations') {
        loadVisualizations();
    } else if (page === 'retrain') {
        loadRetrainStats();
        loadRecentSessions();
    } else if (page === 'uptime') {
        loadUptime();
    }
}

// Model Status
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        const statusIndicator = document.getElementById('statusIndicator');
        if (data.model_loaded) {
            statusIndicator.textContent = '✅ Model Loaded';
            statusIndicator.className = 'status-indicator success';
        } else {
            statusIndicator.textContent = '❌ Model Not Loaded';
            statusIndicator.className = 'status-indicator error';
        }
    } catch (error) {
        const statusIndicator = document.getElementById('statusIndicator');
        statusIndicator.textContent = '❌ Connection Error';
        statusIndicator.className = 'status-indicator error';
    }
}

// File Inputs
function setupFileInputs() {
    // Single image input
    const imageInput = document.getElementById('imageInput');
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            uploadedFile = file;
            document.getElementById('fileName').textContent = file.name;
            displayImagePreview(file);
            document.getElementById('predictBtn').disabled = false;
        }
    });
    
    // Multiple image input
    const multiImageInput = document.getElementById('multiImageInput');
    multiImageInput.addEventListener('change', function(e) {
        selectedFiles = Array.from(e.target.files);
        document.getElementById('fileCount').textContent = `${selectedFiles.length} file(s) selected`;
        displayUploadedFiles(selectedFiles);
        document.getElementById('saveFilesBtn').disabled = selectedFiles.length === 0;
    });
    
    // Sliders
    const epochsSlider = document.getElementById('epochs');
    const fineTuneSlider = document.getElementById('fineTuneEpochs');
    
    epochsSlider.addEventListener('input', function() {
        document.getElementById('epochsValue').textContent = this.value;
    });
    
    fineTuneSlider.addEventListener('input', function() {
        document.getElementById('fineTuneEpochsValue').textContent = this.value;
    });
}

// Image Preview
function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('imagePreview');
        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
    };
    reader.readAsDataURL(file);
}

// Display Uploaded Files with class selection for each file
function displayUploadedFiles(files) {
    const container = document.getElementById('uploadedFiles');
    if (files.length === 0) {
        container.innerHTML = '';
        return;
    }
    
    let html = '<h3>Selected Files - ⚠️ IMPORTANT: Assign Class for Each File:</h3><div class="uploaded-files">';
    files.forEach((file, index) => {
        // Rotate default class to make it more obvious that selection is needed
        const defaultClasses = ['glioma', 'meningioma', 'notumor', 'pituitary'];
        const defaultClass = defaultClasses[index % defaultClasses.length];
        
        html += `
            <div class="uploaded-file-item" style="display: flex; align-items: center; gap: 10px; margin: 10px 0; padding: 10px; border: 2px solid #4CAF50; border-radius: 5px; background-color: #f9f9f9;">
                <div style="flex: 1;">
                    <strong>${file.name}</strong> (${formatFileSize(file.size)})
                </div>
                <label style="font-weight: bold; color: #333;">Class:</label>
                <select class="form-control file-class-select" data-file-index="${index}" style="width: 200px; padding: 8px; border: 2px solid #2196F3; border-radius: 4px; font-size: 14px;" required>
                    <option value="glioma" ${defaultClass === 'glioma' ? 'selected' : ''}>Glioma</option>
                    <option value="meningioma" ${defaultClass === 'meningioma' ? 'selected' : ''}>Meningioma</option>
                    <option value="notumor" ${defaultClass === 'notumor' ? 'selected' : ''}>No Tumor</option>
                    <option value="pituitary" ${defaultClass === 'pituitary' ? 'selected' : ''}>Pituitary</option>
                </select>
            </div>
        `;
    });
    html += '</div>';
    html += '<div class="result-message warning" style="margin-top: 10px; background-color: #fff3cd; border: 2px solid #ffc107; padding: 10px; border-radius: 5px;">';
    html += '⚠️ <strong>Important:</strong> Please verify and change the class selection for each file if needed. Files will be saved with their assigned classes.';
    html += '</div>';
    container.innerHTML = html;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// Prediction
async function makePrediction() {
    if (!uploadedFile) {
        showMessage('predictionOutput', 'Please select an image first.', 'error');
        return;
    }
    
    const output = document.getElementById('predictionOutput');
    output.innerHTML = '<div class="loading">Predicting...</div>';
    
    try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayPredictionResult(result);
        
    } catch (error) {
        output.innerHTML = `<div class="result-message error">Prediction error: ${error.message}</div>`;
    }
}

function displayPredictionResult(result) {
    const output = document.getElementById('predictionOutput');
    
    let html = `
        <div class="metric-card">
            <div class="metric-label">Predicted Class</div>
            <div class="metric-value">${result.predicted_class.toUpperCase()}</div>
            <div style="margin-top: 0.5rem; color: #666;">${(result.confidence * 100).toFixed(2)}% confidence</div>
        </div>
        
        <h3 style="margin-top: 1.5rem;">Class Probabilities</h3>
    `;
    
    // Sort probabilities
    const sortedProbs = Object.entries(result.probabilities)
        .sort((a, b) => b[1] - a[1]);
    
    // Create bar chart data
    const chartData = [{
        x: sortedProbs.map(p => p[0]),
        y: sortedProbs.map(p => p[1] * 100),
        type: 'bar',
        marker: {
            color: sortedProbs.map(p => `rgba(31, 119, 180, ${0.5 + p[1] * 0.5})`)
        }
    }];
    
    const chartLayout = {
        title: 'Prediction Probabilities',
        xaxis: { title: 'Class' },
        yaxis: { title: 'Probability (%)' },
        margin: { l: 50, r: 50, t: 50, b: 50 }
    };
    
    html += '<div id="probChart" style="margin: 1rem 0;"></div>';
    html += '<table class="probability-table"><thead><tr><th>Class</th><th>Probability</th></tr></thead><tbody>';
    
    sortedProbs.forEach(([class_name, prob]) => {
        html += `<tr>
            <td>${class_name}</td>
            <td>
                ${(prob * 100).toFixed(2)}%
                <div class="probability-bar" style="width: ${prob * 100}%"></div>
            </td>
        </tr>`;
    });
    
    html += '</tbody></table>';
    output.innerHTML = html;
    
    // Render Plotly chart
    Plotly.newPlot('probChart', chartData, chartLayout, {responsive: true});
}

// Visualizations
async function loadVisualizations() {
    const container = document.getElementById('visualizationsContent');
    container.innerHTML = '<div class="loading">Loading visualizations...</div>';
    
    try {
        // Try to load feature data
        const response = await fetch(`${API_BASE}/visualizations/data`);
        
        if (!response.ok) {
            throw new Error('Failed to load visualization data');
        }
        
        const data = await response.json();
        
        if (!data.features || data.features.length === 0) {
            container.innerHTML = `
                <div class="result-message info">
                    Feature CSV file not found. Please run feature extraction first.
                </div>
            `;
            return;
        }
        
        displayVisualizations(data);
        
    } catch (error) {
        container.innerHTML = `
            <div class="result-message error">
                Error loading visualizations: ${error.message}
            </div>
        `;
    }
}

function displayVisualizations(data) {
    const container = document.getElementById('visualizationsContent');
    let html = `
        <div class="result-message success">
            Loaded features from ${data.source}
        </div>
        <div class="result-message info">
            Total samples: ${data.total_samples}
        </div>
    `;
    
    // Feature 1: Mean Intensity
    if (data.has_mean_intensity) {
        html += `
            <div class="visualization-section">
                <h3>Feature 1: Mean Intensity Distribution by Tumor Class</h3>
                <p><strong>Interpretation</strong>: This visualization shows the average pixel intensity across different tumor types.</p>
                <div id="chart1"></div>
            </div>
        `;
    }
    
    // Feature 2: Standard Deviation
    if (data.has_std_intensity) {
        html += `
            <div class="visualization-section">
                <h3>Feature 2: Intensity Variability (Standard Deviation) by Class</h3>
                <p><strong>Interpretation</strong>: Standard deviation measures texture variability in the images.</p>
                <div id="chart2"></div>
            </div>
        `;
    }
    
    // Feature 3: Gradient Mean
    if (data.has_gradient_mean) {
        html += `
            <div class="visualization-section">
                <h3>Feature 3: Edge Strength (Gradient Mean) by Class</h3>
                <p><strong>Interpretation</strong>: Gradient magnitude indicates edge strength and boundaries in images.</p>
                <div id="chart3"></div>
            </div>
        `;
    }
    
    // Class Distribution
    if (data.class_distribution) {
        html += `
            <div class="visualization-section">
                <h3>Class Distribution</h3>
                <div id="chart4"></div>
            </div>
        `;
    }
    
    container.innerHTML = html;
    
    // Render charts
    if (data.has_mean_intensity) {
        renderBoxPlot('chart1', data.mean_intensity_data, 'Mean Intensity by Tumor Class', 'class', 'mean_intensity');
    }
    
    if (data.has_std_intensity) {
        renderViolinPlot('chart2', data.std_intensity_data, 'Intensity Standard Deviation by Tumor Class', 'class', 'std_intensity');
    }
    
    if (data.has_gradient_mean) {
        renderScatterPlot('chart3', data.gradient_data, 'Mean Intensity vs Gradient Mean', 'mean_intensity', 'gradient_mean');
    }
    
    if (data.class_distribution) {
        renderPieChart('chart4', data.class_distribution, 'Class Distribution');
    }
}

function renderBoxPlot(containerId, data, title, xCol, yCol) {
    const traces = [];
    const classes = [...new Set(data.map(d => d[xCol]))];
    
    classes.forEach(cls => {
        const values = data.filter(d => d[xCol] === cls).map(d => d[yCol]);
        traces.push({
            y: values,
            type: 'box',
            name: cls,
            boxpoints: 'outliers'
        });
    });
    
    Plotly.newPlot(containerId, traces, {
        title: title,
        yaxis: { title: yCol },
        margin: { l: 50, r: 50, t: 50, b: 50 }
    }, {responsive: true});
}

function renderViolinPlot(containerId, data, title, xCol, yCol) {
    const traces = [];
    const classes = [...new Set(data.map(d => d[xCol]))];
    
    classes.forEach(cls => {
        const values = data.filter(d => d[xCol] === cls).map(d => d[yCol]);
        traces.push({
            y: values,
            type: 'violin',
            name: cls,
            box: { visible: true }
        });
    });
    
    Plotly.newPlot(containerId, traces, {
        title: title,
        yaxis: { title: yCol },
        margin: { l: 50, r: 50, t: 50, b: 50 }
    }, {responsive: true});
}

function renderScatterPlot(containerId, data, title, xCol, yCol) {
    const traces = [];
    const classes = [...new Set(data.map(d => d.class))];
    
    classes.forEach(cls => {
        const filtered = data.filter(d => d.class === cls);
        traces.push({
            x: filtered.map(d => d[xCol]),
            y: filtered.map(d => d[yCol]),
            mode: 'markers',
            type: 'scatter',
            name: cls
        });
    });
    
    Plotly.newPlot(containerId, traces, {
        title: title,
        xaxis: { title: xCol },
        yaxis: { title: yCol },
        margin: { l: 50, r: 50, t: 50, b: 50 }
    }, {responsive: true});
}

function renderPieChart(containerId, distribution, title) {
    const trace = {
        labels: Object.keys(distribution),
        values: Object.values(distribution),
        type: 'pie'
    };
    
    Plotly.newPlot(containerId, [trace], {
        title: title,
        margin: { l: 50, r: 50, t: 50, b: 50 }
    }, {responsive: true});
}

// Save Files with per-file class assignment
async function saveFiles() {
    if (selectedFiles.length === 0) {
        showMessage('uploadResult', 'Please select files first.', 'error');
        return;
    }
    
    const resultDiv = document.getElementById('uploadResult');
    resultDiv.innerHTML = '<div class="loading">Saving files with their assigned classes...</div>';
    
    try {
        // Get class assignments for each file
        const fileClassAssignments = [];
        const classSelects = document.querySelectorAll('.file-class-select');
        
        if (classSelects.length === 0) {
            showMessage('uploadResult', 'Error: Class selection dropdowns not found. Please refresh and try again.', 'error');
            return;
        }
        
        classSelects.forEach((select, index) => {
            if (index < selectedFiles.length) {
                const selectedClass = select.value;
                if (!selectedClass) {
                    showMessage('uploadResult', `Error: No class selected for file ${selectedFiles[index].name}. Please select a class.`, 'error');
                    return;
                }
                fileClassAssignments.push({
                    file: selectedFiles[index],
                    class_name: selectedClass
                });
            }
        });
        
        // Validate that we have assignments for all files
        if (fileClassAssignments.length !== selectedFiles.length) {
            showMessage('uploadResult', 'Error: Class assignment mismatch. Please ensure all files have a class selected.', 'error');
            return;
        }
        
        // Show summary of class distribution
        const classCounts = {};
        fileClassAssignments.forEach(assignment => {
            classCounts[assignment.class_name] = (classCounts[assignment.class_name] || 0) + 1;
        });
        const classSummary = Object.entries(classCounts).map(([cls, count]) => `${count} ${cls}`).join(', ');
        resultDiv.innerHTML = `<div class="loading">Saving ${selectedFiles.length} file(s) with classes: ${classSummary}...</div>`;
        
        // Upload each file with its assigned class
        let successCount = 0;
        let errorCount = 0;
        const errors = [];
        
        for (const assignment of fileClassAssignments) {
            try {
                const formData = new FormData();
                formData.append('files', assignment.file);
                formData.append('class_name', assignment.class_name);
                
                const response = await fetch(`${API_BASE}/retrain`, {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`Failed to save ${assignment.file.name}`);
                }
                
                successCount++;
            } catch (error) {
                errorCount++;
                errors.push(`${assignment.file.name}: ${error.message}`);
            }
        }
        
        // Show results with class distribution
        let resultHtml = '';
        if (successCount > 0) {
            // Calculate class distribution from successfully saved files
            const savedClassCounts = {};
            fileClassAssignments.slice(0, successCount).forEach(assignment => {
                savedClassCounts[assignment.class_name] = (savedClassCounts[assignment.class_name] || 0) + 1;
            });
            const classDistribution = Object.entries(savedClassCounts)
                .map(([cls, count]) => `<strong>${count}</strong> ${cls}`)
                .join(', ');
            
            resultHtml += `
                <div class="result-message success">
                    ✅ Successfully saved ${successCount} file(s) to database!
                </div>
                <div class="result-message info" style="margin-top: 10px;">
                    📊 Class distribution: ${classDistribution}
                </div>
            `;
        }
        if (errorCount > 0) {
            resultHtml += `
                <div class="result-message error">
                    ❌ Failed to save ${errorCount} file(s). Errors: ${errors.join(', ')}
                </div>
            `;
        }
        resultHtml += `
            <div class="result-message info" style="margin-top: 10px;">
                Files are ready for retraining. Go to 'Retrain Model' to trigger retraining.
            </div>
        `;
        
        resultDiv.innerHTML = resultHtml;
        
        // Clear selection
        selectedFiles = [];
        document.getElementById('multiImageInput').value = '';
        document.getElementById('fileCount').textContent = '';
        document.getElementById('uploadedFiles').innerHTML = '';
        document.getElementById('saveFilesBtn').disabled = true;
        
    } catch (error) {
        resultDiv.innerHTML = `<div class="result-message error">Error saving files: ${error.message}</div>`;
    }
}

// Retrain Stats
async function loadRetrainStats() {
    try {
        const response = await fetch(`${API_BASE}/database/stats`);
        const stats = await response.json();
        
        document.getElementById('totalUploaded').textContent = stats.total_uploaded_images || 0;
        document.getElementById('processed').textContent = stats.processed_images || 0;
        document.getElementById('trainingSessions').textContent = stats.total_training_sessions || 0;
        document.getElementById('completed').textContent = stats.completed_sessions || 0;
        
    } catch (error) {
        console.error('Error loading retrain stats:', error);
    }
}

async function loadRecentSessions() {
    try {
        const response = await fetch(`${API_BASE}/retrain/sessions`);
        const sessions = await response.json();
        
        const container = document.getElementById('recentSessions');
        
        if (!sessions || sessions.length === 0) {
            container.innerHTML = '<div class="result-message info">No training sessions yet.</div>';
            return;
        }
        
        let html = '<h3>Recent Training Sessions</h3><table class="sessions-table"><thead><tr>';
        html += '<th>ID</th><th>Status</th><th>Epochs</th><th>Accuracy</th><th>Images Used</th>';
        html += '</tr></thead><tbody>';
        
        sessions.forEach(session => {
            // Format timestamp properly (already adjusted by +2 hours in API)
            let timestampStr = '-';
            if (session.session_timestamp) {
                try {
                    // Handle ISO format or SQLite format
                    let date;
                    if (session.session_timestamp.includes('T')) {
                        // ISO format
                        date = new Date(session.session_timestamp);
                    } else {
                        // SQLite format: YYYY-MM-DD HH:MM:SS
                        date = new Date(session.session_timestamp.replace(' ', 'T'));
                    }
                    // Format with proper timezone (timestamp already has +2 hours from API)
                    timestampStr = date.toLocaleString('en-US', {
                        year: 'numeric',
                        month: '2-digit',
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        hour12: false
                    });
                } catch (e) {
                    timestampStr = session.session_timestamp;
                }
            }
            
            html += `<tr>
                <td>${session.id}</td>
                <td>${session.status}</td>
                <td>${session.epochs || '-'}</td>
                <td>${session.final_accuracy ? (session.final_accuracy * 100).toFixed(2) + '%' : '-'}</td>
                <td>${session.images_used || '-'}</td>
            </tr>`;
        });
        
        html += '</tbody></table>';
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
}

// Trigger Retraining
async function triggerRetraining() {
    const epochs = document.getElementById('epochs').value;
    const fineTuneEpochs = document.getElementById('fineTuneEpochs').value;
    const resultDiv = document.getElementById('retrainResult');
    
    resultDiv.innerHTML = '<div class="loading">Retraining model... This may take a while. Please wait...</div>';
    
    // Create abort controller for timeout handling
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minutes timeout
    
    try {
        const response = await fetch(`${API_BASE}/retrain/trigger`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                epochs: parseInt(epochs),
                fine_tune_epochs: parseInt(fineTuneEpochs)
            }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        // Check if response is ok
        if (!response.ok) {
            let errorMessage = 'Retraining failed';
            try {
                const errorData = await response.json();
                errorMessage = errorData.message || errorData.detail || errorMessage;
            } catch (e) {
                // If response is not JSON, try to get text
                try {
                    const errorText = await response.text();
                    errorMessage = errorText || errorMessage;
                } catch (e2) {
                    errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                }
            }
            throw new Error(errorMessage);
        }
        
        // Try to parse JSON response
        let result;
        try {
            const text = await response.text();
            if (!text || text.trim() === '') {
                throw new Error('Empty response from server');
            }
            result = JSON.parse(text);
        } catch (parseError) {
            throw new Error(`Failed to parse response: ${parseError.message}. The server may have timed out or encountered an error.`);
        }
        
        if (result.status === 'success') {
            resultDiv.innerHTML = `
                <div class="result-message success">
                    ✅ Model retraining completed successfully!
                </div>
            `;
        } else {
            throw new Error(result.message || 'Retraining failed');
        }
        
        // Reload stats
        loadRetrainStats();
        loadRecentSessions();
        checkModelStatus();
        
    } catch (error) {
        clearTimeout(timeoutId);
        let errorMessage = error.message;
        if (error.name === 'AbortError' || error.name === 'TimeoutError') {
            errorMessage = 'Retraining request timed out. The process may still be running. Please check the server logs or try again later.';
        } else if (error.message.includes('JSON') || error.message.includes('parse')) {
            errorMessage = 'Server response error. The retraining process may have encountered an issue. Please check the server logs.';
        }
        resultDiv.innerHTML = `<div class="result-message error">Retraining error: ${errorMessage}</div>`;
        console.error('Retraining error:', error);
    }
}

// Uptime
async function loadUptime() {
    try {
        const response = await fetch(`${API_BASE}/uptime`);
        const data = await response.json();
        
        document.getElementById('uptimeValue').textContent = data.uptime_formatted || '-';
        document.getElementById('totalRequests').textContent = data.total_requests || 0;
        document.getElementById('modelStatusValue').textContent = data.model_loaded ? '🟢 Online' : '🔴 Offline';
        
        // Update model info
        const modelInfo = document.getElementById('modelInfo');
        if (data.model_loaded) {
            modelInfo.innerHTML = `
                <div class="info-item">✅ Model is loaded and ready</div>
                <div class="info-item">Model Path: models/brain_tumor_model.h5</div>
            `;
        } else {
            modelInfo.innerHTML = '<div class="info-item">❌ Model is not loaded</div>';
        }
        
        // Update performance info
        const performanceInfo = document.getElementById('performanceInfo');
        performanceInfo.innerHTML = `
            <div class="info-item">Request handling: Active</div>
            <div class="info-item">Session started: ${new Date(data.timestamp).toLocaleString()}</div>
        `;
        
    } catch (error) {
        console.error('Error loading uptime:', error);
    }
}

// Utility
function showMessage(containerId, message, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = `<div class="result-message ${type}">${message}</div>`;
}

