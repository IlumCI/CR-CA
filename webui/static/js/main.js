// CRCA Web UI - Main JavaScript
// State management and UI interactions

// Application State
const AppState = {
    currentAgentKey: null,
    agents: [],
    isLoading: false,
    sessionId: null
};

// DOM Elements Cache
const DOM = {
    agentSelect: null,
    agentInfo: null,
    chatMessages: null,
    chatInput: null,
    sendButton: null
};

// Initialize application
async function init() {
    try {
        // Cache DOM elements
        DOM.agentSelect = document.getElementById('agentSelect');
        DOM.agentInfo = document.getElementById('agentInfo');
        DOM.chatMessages = document.getElementById('chatMessages');
        DOM.chatInput = document.getElementById('chatInput');
        DOM.sendButton = document.getElementById('sendButton');

        // Load agents and setup
        await loadAgents();
        setupEventListeners();
        
        // Focus input
        DOM.chatInput.focus();
    } catch (error) {
        console.error('Initialization error:', error);
        addMessage('error', 'System', 'Failed to initialize application. Please refresh the page.');
    }
}

// Load available agents from API
async function loadAgents() {
    try {
        const response = await fetch('/api/agents');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        AppState.agents = data.agents || [];
        AppState.currentAgentKey = data.default || (AppState.agents.length > 0 ? AppState.agents[0].key : null);

        // Populate agent selector
        populateAgentSelector();
        updateAgentInfo();
        
    } catch (error) {
        console.error('Failed to load agents:', error);
        addMessage('error', 'System', 'Failed to load agents. Please refresh the page.');
    }
}

// Populate agent selector dropdown
function populateAgentSelector() {
    if (!DOM.agentSelect) return;
    
    DOM.agentSelect.innerHTML = '';
    
    if (AppState.agents.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No agents available';
        DOM.agentSelect.appendChild(option);
        return;
    }
    
    AppState.agents.forEach(agent => {
        const option = document.createElement('option');
        option.value = agent.key;
        option.textContent = agent.name;
        if (agent.key === AppState.currentAgentKey) {
            option.selected = true;
        }
        DOM.agentSelect.appendChild(option);
    });
}

// Update agent info display
function updateAgentInfo() {
    if (!DOM.agentInfo) return;
    
    const agent = AppState.agents.find(a => a.key === AppState.currentAgentKey);
    if (agent) {
        DOM.agentInfo.textContent = agent.description || '';
    } else {
        DOM.agentInfo.textContent = '';
    }
}

// Setup all event listeners
function setupEventListeners() {
    // Agent selection change
    if (DOM.agentSelect) {
        DOM.agentSelect.addEventListener('change', handleAgentChange);
    }

    // Send button click
    if (DOM.sendButton) {
        DOM.sendButton.addEventListener('click', sendMessage);
    }

    // Enter key to send (Shift+Enter for new line)
    if (DOM.chatInput) {
        DOM.chatInput.addEventListener('keydown', handleKeyDown);
        DOM.chatInput.addEventListener('input', handleInputResize);
    }
}

// Handle agent selection change
async function handleAgentChange(event) {
    const newAgentKey = event.target.value;
    
    if (!newAgentKey || newAgentKey === AppState.currentAgentKey) {
        return;
    }
    
    try {
        const response = await fetch('/api/switch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ agent_key: newAgentKey })
        });
        
        if (response.ok) {
            const data = await response.json();
            AppState.currentAgentKey = newAgentKey;
            updateAgentInfo();
            
            const agentName = AppState.agents.find(a => a.key === newAgentKey)?.name || newAgentKey;
            addMessage('agent', 'System', `Switched to ${agentName}`);
        } else {
            const errorData = await response.json();
            addMessage('error', 'System', `Failed to switch agent: ${errorData.error || 'Unknown error'}`);
            // Revert selection
            DOM.agentSelect.value = AppState.currentAgentKey;
        }
    } catch (error) {
        console.error('Failed to switch agent:', error);
        addMessage('error', 'System', 'Failed to switch agent. Please try again.');
        DOM.agentSelect.value = AppState.currentAgentKey;
    }
}

// Handle keyboard input
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Auto-resize textarea
function handleInputResize() {
    if (!DOM.chatInput) return;
    
    DOM.chatInput.style.height = 'auto';
    const newHeight = Math.min(DOM.chatInput.scrollHeight, 150);
    DOM.chatInput.style.height = newHeight + 'px';
}

// Add message to chat display
function addMessage(type, sender, content) {
    if (!DOM.chatMessages) return;
    
    // Remove empty state if present
    const emptyState = DOM.chatMessages.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    // Create message container
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    // Create header
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    headerDiv.textContent = sender;
    
    // Create content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    // Assemble message
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    DOM.chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom with smooth behavior
    DOM.chatMessages.scrollTo({
        top: DOM.chatMessages.scrollHeight,
        behavior: 'smooth'
    });
}

// Send message to agent
async function sendMessage() {
    if (!DOM.chatInput || !DOM.sendButton) return;
    
    const message = DOM.chatInput.value.trim();
    
    // Validation
    if (!message || AppState.isLoading || !AppState.currentAgentKey) {
        return;
    }

    // Add user message to chat
    addMessage('user', 'You', message);
    
    // Clear and reset input
    DOM.chatInput.value = '';
    DOM.chatInput.style.height = 'auto';
    
    // Update UI state
    setLoadingState(true);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                agent_key: AppState.currentAgentKey
            })
        });

        const data = await response.json();

        if (response.ok && data.response) {
            const agentName = data.agent_name || 'Agent';
            addMessage('agent', agentName, data.response);
        } else {
            addMessage('error', 'Error', data.error || 'Failed to get response from agent');
        }
    } catch (error) {
        console.error('Failed to send message:', error);
        addMessage('error', 'Error', 'Failed to send message. Please try again.');
    } finally {
        setLoadingState(false);
        DOM.chatInput.focus();
    }
}

// Set loading state
function setLoadingState(loading) {
    AppState.isLoading = loading;
    
    if (!DOM.chatInput || !DOM.sendButton) return;
    
    DOM.chatInput.disabled = loading;
    DOM.sendButton.disabled = loading;
    
    if (loading) {
        DOM.sendButton.innerHTML = '<span class="loading"></span>Sending...';
    } else {
        DOM.sendButton.textContent = 'Send';
    }
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
