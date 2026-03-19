/* ============================================================
   Eye on AI Chatbot — Client-side JavaScript
   ============================================================ */

// State
let conversationId = null;
let isLoading = false;

// DOM elements
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

// API base URL — auto-detect from current page location
const API_BASE = window.EYEONAI_API_URL || window.location.origin;

// ---- Auto-resize textarea ----

chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

// ---- Submit on Enter (Shift+Enter for newline) ----

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// ---- Suggestion chips ----

function askSuggestion(btn) {
    const text = btn.textContent.trim();
    chatInput.value = text;
    chatForm.dispatchEvent(new Event('submit'));
}

// ---- Form submit handler ----

function handleSubmit(e) {
    e.preventDefault();

    const message = chatInput.value.trim();
    if (!message || isLoading) return;

    // Hide welcome message on first interaction
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Add user message to UI
    appendMessage('user', message);

    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Send to API
    sendMessage(message);
}

// ---- Send message to API ----

async function sendMessage(message) {
    isLoading = true;
    sendBtn.disabled = true;

    // Show typing indicator
    const typingEl = appendTypingIndicator();

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                conversation_id: conversationId,
            }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${response.status})`);
        }

        const data = await response.json();

        // Update conversation ID
        conversationId = data.conversation_id;

        // Remove typing indicator
        typingEl.remove();

        // Add bot response
        appendMessage('bot', data.response, data.sources);

    } catch (err) {
        console.error('Chat error:', err);
        typingEl.remove();
        appendError(err.message || 'Something went wrong. Please try again.');
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

// ---- Render messages ----

function appendMessage(role, text, sources) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', role);

    const avatarDiv = document.createElement('div');
    avatarDiv.classList.add('message-avatar');
    avatarDiv.textContent = role === 'user' ? '👤' : '🎙️';

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');

    const bubbleDiv = document.createElement('div');
    bubbleDiv.classList.add('message-bubble');
    bubbleDiv.innerHTML = formatText(text);

    contentDiv.appendChild(bubbleDiv);

    // Add sources if present
    if (sources && sources.length > 0) {
        const sourcesDiv = createSourcesWidget(sources);
        contentDiv.appendChild(sourcesDiv);
    }

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function appendTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot');
    messageDiv.id = 'typing-indicator';

    const avatarDiv = document.createElement('div');
    avatarDiv.classList.add('message-avatar');
    avatarDiv.textContent = '🎙️';

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');

    const typingDiv = document.createElement('div');
    typingDiv.classList.add('message-bubble', 'typing-indicator');
    typingDiv.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;

    contentDiv.appendChild(typingDiv);
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);

    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    return messageDiv;
}

function appendError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.classList.add('error-message');
    errorDiv.textContent = `⚠️ ${message}`;
    chatMessages.appendChild(errorDiv);
    scrollToBottom();
}

// ---- Sources widget ----

function createSourcesWidget(sources) {
    const container = document.createElement('div');
    container.classList.add('sources');

    const toggleBtn = document.createElement('button');
    toggleBtn.classList.add('sources-toggle');
    toggleBtn.innerHTML = `<span class="arrow">▶</span> ${sources.length} source${sources.length !== 1 ? 's' : ''}`;

    const list = document.createElement('ul');
    list.classList.add('sources-list');

    sources.forEach((src) => {
        const item = document.createElement('li');
        item.classList.add('source-item');
        item.innerHTML = `
            <div class="source-episode">${escapeHtml(src.episode)}</div>
            <div class="source-snippet">${escapeHtml(src.snippet)}</div>
        `;
        list.appendChild(item);
    });

    toggleBtn.addEventListener('click', () => {
        toggleBtn.classList.toggle('open');
        list.classList.toggle('open');
    });

    container.appendChild(toggleBtn);
    container.appendChild(list);

    return container;
}

// ---- Utilities ----

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatText(text) {
    // Basic markdown-like formatting
    let html = escapeHtml(text);

    // Bold: **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Italic: *text*
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Inline code: `text`
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');

    // Line breaks → paragraphs
    html = html
        .split(/\n\n+/)
        .map((p) => `<p>${p.trim()}</p>`)
        .join('');

    // Single line breaks within paragraphs
    html = html.replace(/\n/g, '<br>');

    return html;
}
