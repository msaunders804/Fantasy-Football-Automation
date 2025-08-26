// Main JavaScript for Fantasy Football ML Draft System
// Common functions used across multiple pages

// Global variables
let systemStatus = {};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    checkSystemStatus();
    setupCommonEventListeners();
}

// System Status Management
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        systemStatus = data;
        
        updateSystemStatusBanner(data);
        
        // Broadcast status to other functions if needed
        if (window.onSystemStatusUpdate) {
            window.onSystemStatusUpdate(data);
        }
        
    } catch (error) {
        console.error('Error checking system status:', error);
        updateSystemStatusBanner({ error: true });
    }
}

function updateSystemStatusBanner(status) {
    const banner = document.getElementById('system-status-banner');
    const indicator = document.getElementById('status-indicator');
    const message = document.getElementById('status-message');
    const details = document.getElementById('status-details');
    
    if (!banner) return; // Not all pages have status banner
    
    let statusClass = 'success';
    let statusMessage = 'System Ready';
    let statusDetails = '';
    
    if (status.error) {
        statusClass = 'error';
        statusMessage = 'System Error';
        statusDetails = 'Unable to connect to system';
    } else {
        // Determine overall status
        const issues = [];
        
        if (!status.system_initialized) {
            issues.push('System not initialized');
        }
        if (!status.models_trained) {
            issues.push('Models not trained');
        }
        if (!status.projections_ready) {
            issues.push('Projections not ready');
        }
        if (!status.claude_api_available) {
            issues.push('AI Advisor not configured');
        }
        
        if (issues.length === 0) {
            statusClass = 'success';
            statusMessage = 'All Systems Ready';
            statusDetails = 'ML models trained, projections ready, AI advisor available';
        } else if (issues.length <= 2) {
            statusClass = 'warning';
            statusMessage = 'Partial Setup';
            statusDetails = issues.join(', ');
        } else {
            statusClass = 'error';
            statusMessage = 'Setup Required';
            statusDetails = `${issues.length} issues: ${issues[0]}...`;
        }
    }
    
    // Update banner appearance
    banner.className = `system-status-banner ${statusClass}`;
    indicator.className = `status-dot ${statusClass === 'success' ? '' : statusClass}`;
    message.textContent = statusMessage;
    details.textContent = statusDetails;
    
    // Show banner
    banner.style.display = 'flex';
}

// Common Event Listeners
function setupCommonEventListeners() {
    // Handle navigation active states
    updateActiveNavigation();
    
    // Auto-refresh status every 30 seconds
    setInterval(checkSystemStatus, 30000);
}

function updateActiveNavigation() {
    const currentPath = window.location.pathname;
    const navTabs = document.querySelectorAll('.nav-tab');
    
    navTabs.forEach(tab => {
        tab.classList.remove('active');
        if (tab.getAttribute('href') === currentPath || 
            (currentPath === '/' && tab.getAttribute('href') === '/')) {
            tab.classList.add('active');
        }
    });
}

// Common API Functions
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`API call failed for ${endpoint}:`, error);
        throw error;
    }
}

// Training Functions
async function trainModels() {
    const btn = document.getElementById('train-models-btn');
    const statusDiv = document.getElementById('training-status');
    
    if (!btn || !statusDiv) return;
    
    setLoadingState(btn, 'Training...', true);
    showLoadingMessage(statusDiv, 'Training ML models...');
    
    try {
        const data = await apiCall('/api/train-models', { method: 'POST' });
        
        if (data.success) {
            showAlert(statusDiv, data.message, 'success');
            checkSystemStatus(); // Refresh system status
        } else {
            showAlert(statusDiv, `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showAlert(statusDiv, `Error: ${error.message}`, 'error');
    } finally {
        setLoadingState(btn, 'Train Models', false);
    }
}

async function generateProjections() {
    const btn = document.getElementById('generate-projections-btn');
    const statusDiv = document.getElementById('projections-status');
    
    if (!btn || !statusDiv) return;
    
    setLoadingState(btn, 'Generating...', true);
    showLoadingMessage(statusDiv, 'Generating 2025 projections...');
    
    try {
        const data = await apiCall('/api/generate-projections', { method: 'POST' });
        
        if (data.success) {
            showAlert(statusDiv, data.message, 'success');
            checkSystemStatus(); // Refresh system status
        } else {
            showAlert(statusDiv, `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showAlert(statusDiv, `Error: ${error.message}`, 'error');
    } finally {
        setLoadingState(btn, 'Generate 2025 Projections', false);
    }
}

// UI Helper Functions
function setLoadingState(button, loadingText, isLoading) {
    button.disabled = isLoading;
    button.textContent = loadingText;
}

function showLoadingMessage(element, message) {
    element.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>${message}</p>
        </div>
    `;
}

function showAlert(element, message, type = 'info') {
    element.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
}

function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type}`;
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '1000';
    notification.style.minWidth = '300px';
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Fade in
    notification.style.opacity = '0';
    notification.style.transform = 'translateY(-20px)';
    notification.style.transition = 'all 0.3s ease';
    
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateY(0)';
    }, 100);
    
    // Remove after duration
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-20px)';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// League Settings Functions
async function updateLeagueSettings(settings) {
    try {
        const data = await apiCall('/api/league-settings', {
            method: 'POST',
            body: JSON.stringify(settings)
        });
        
        if (data.success) {
            showNotification('Settings updated successfully!', 'success');
            checkSystemStatus();
            return true;
        } else {
            showNotification(`Error updating settings: ${data.error}`, 'error');
            return false;
        }
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
        return false;
    }
}

async function getLeagueSettings() {
    try {
        const data = await apiCall('/api/league-settings');
        return data;
    } catch (error) {
        console.error('Error getting league settings:', error);
        return {};
    }
}

// Utility Functions
function formatNumber(num, decimals = 1) {
    return parseFloat(num).toFixed(decimals);
}

function formatPosition(position) {
    const positionMap = {
        'QB': 'Quarterback',
        'RB': 'Running Back', 
        'WR': 'Wide Receiver',
        'TE': 'Tight End',
        'K': 'Kicker',
        'DEF': 'Defense'
    };
    return positionMap[position] || position;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Make functions globally available
window.trainModels = trainModels;
window.generateProjections = generateProjections;
window.updateLeagueSettings = updateLeagueSettings;
window.getLeagueSettings = getLeagueSettings;
window.checkSystemStatus = checkSystemStatus;
window.apiCall = apiCall;
window.showNotification = showNotification;
window.formatNumber = formatNumber;
window.formatPosition = formatPosition;