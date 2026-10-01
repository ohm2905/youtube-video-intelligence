let ytPlayer = null;
let currentVideoId = null;
let currentMetadata = null;

// Initialize YouTube Iframe Player
function onYouTubeIframeAPIReady() {
    console.log("YouTube Iframe API loaded.");
}

function loadYouTubeVideo(videoId, startSeconds = 0) {
    currentVideoId = videoId;
    const placeholder = document.getElementById("video-placeholder");
    if (placeholder) {
        placeholder.style.display = "none";
    }
    if (ytPlayer && typeof ytPlayer.loadVideoById === "function") {
        ytPlayer.loadVideoById({
            videoId: videoId,
            startSeconds: startSeconds,
        });
    } else {
        ytPlayer = new YT.Player("yt-player", {
            height: "100%",
            width: "100%",
            videoId: videoId,
            playerVars: {
                playsinline: 1,
                autoplay: 0,
                start: Math.floor(startSeconds),
            },
        });
    }
}

// Convert "01:23:45" or "12:34" to total seconds
function timestampToSeconds(timestampStr) {
    if (!timestampStr) return 0;
    const parts = timestampStr.trim().split(":").map(Number);
    if (parts.length === 3) {
        return parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (parts.length === 2) {
        return parts[0] * 60 + parts[1];
    } else if (parts.length === 1) {
        return parts[0];
    }
    return 0;
}

// Seek video to specific timestamp
window.seekVideo = function (seconds) {
    if (typeof seconds === "string") {
        seconds = timestampToSeconds(seconds);
    }
    if (ytPlayer && typeof ytPlayer.seekTo === "function") {
        ytPlayer.seekTo(seconds, true);
        ytPlayer.playVideo();
        const playerCard = document.getElementById("video-card");
        if (playerCard) {
            playerCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    } else if (currentVideoId) {
        loadYouTubeVideo(currentVideoId, seconds);
    }
};

// Convert timestamp patterns in text into clickable buttons
function linkifyTimestamps(text) {
    if (!text) return "";
    // Matches patterns like [12:34], [01:23:45], [05:12 - 07:30]
    const tsRegex = /\[?(\b(?:\d{1,2}:)?\d{2}:\d{2}\b)(?:\s*[-–]\s*((?:\d{1,2}:)?\d{2}:\d{2}))?\]?/g;
    return text.replace(tsRegex, (match, startTs, endTs) => {
        const secs = timestampToSeconds(startTs);
        const label = endTs ? `${startTs} – ${endTs}` : `${startTs}`;
        return `<button type="button" class="timestamp-action-btn" onclick="window.seekVideo(${secs})"><span class="ts-play-icon">▶</span> ${label}</button>`;
    });
}

// Starter Prompt Helper
window.askStarter = function (questionText) {
    const input = document.getElementById("question-input");
    input.value = questionText;
    askQuestion();
};

// Reset Chat Helper
window.clearChat = function () {
    const container = document.getElementById("chat-history");
    if (!container) return;
    container.innerHTML = `
        <div class="message-item assistant">
            <div class="message-body-assistant">
                <p>Ask any question about this video. Answers are strictly synthesized from spoken transcripts with clickable timestamps.</p>
                <div class="starter-prompts">
                    <button type="button" class="starter-chip" onclick="askStarter('What are the major stages of the LLM training pipeline?')">What are the major stages of the LLM training pipeline?</button>
                    <button type="button" class="starter-chip" onclick="askStarter('What is RLHF and why is it used?')">What is RLHF and why is it used?</button>
                    <button type="button" class="starter-chip" onclick="askStarter('Does this video discuss LangGraph?')">Does this video discuss LangGraph?</button>
                </div>
            </div>
        </div>
    `;
};

// System Modal Toggle
window.toggleSystemModal = function () {
    const modal = document.getElementById("system-modal");
    if (!modal) return;
    if (modal.open) {
        modal.close();
    } else {
        modal.showModal();
    }
};

// Theme Toggle Management
function updateThemeUI(theme) {
    const icon = document.getElementById("theme-toggle-icon");
    const label = document.getElementById("theme-toggle-label");
    if (icon && label) {
        if (theme === "light") {
            icon.innerText = "☾";
            label.innerText = "Dark";
        } else {
            icon.innerText = "☼";
            label.innerText = "Light";
        }
    }
}

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
        localStorage.setItem("yt_intel_theme", theme);
    } catch (e) {}
    updateThemeUI(theme);
}

window.toggleTheme = function () {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
};

// Process Video
async function analyzeVideo() {
    const urlInput = document.getElementById("youtube-url-input");
    const url = urlInput.value.trim();
    if (!url) {
        alert("Please enter a valid YouTube URL or Video ID.");
        return;
    }

    const analyzeBtn = document.getElementById("analyze-btn");
    const progressCard = document.getElementById("progress-card");
    const stepTag = document.getElementById("step-status-tag");
    const step1 = document.getElementById("step-1");
    const step2 = document.getElementById("step-2");
    const step3 = document.getElementById("step-3");

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `<span>Analyzing...</span>`;
    
    if (progressCard) {
        progressCard.classList.remove("d-none");
        progressCard.open = true;
    }
    if (stepTag) {
        stepTag.innerText = "Processing";
        stepTag.style.color = "var(--text-secondary)";
    }
    step1.className = "step-item active";
    step2.className = "step-item";
    step3.className = "step-item";

    try {
        step1.innerHTML = `<span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--accent-red); margin-right:6px;"></span> Fetching video metadata and transcript...`;
        
        const response = await fetch("/api/videos/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ youtube_url: url }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to process video.");
        }

        const data = await response.json();
        currentMetadata = data.video;
        currentVideoId = data.video.video_id;

        step1.className = "step-item done";
        step1.innerHTML = `✓ Retrieved metadata: "${data.video.title}"`;

        step2.className = "step-item done";
        step2.innerHTML = `✓ Chunked ${data.total_snippets} transcript items into ${data.total_chunks} timestamped segments`;

        step3.className = "step-item done";
        step3.innerHTML = `✓ Vector index verified in ChromaDB (${data.already_existed ? "Reused existing index" : "Indexed newly"})`;

        if (stepTag) {
            stepTag.innerText = "Ready";
            stepTag.style.color = "var(--status-success)";
        }

        // Render Video Player & Metadata
        document.getElementById("video-card").classList.remove("d-none");
        document.getElementById("video-title").innerText = data.video.title || "YouTube Video";
        document.getElementById("video-channel").innerText = data.video.channel || "YouTube Channel";
        document.getElementById("video-duration").innerText = data.video.duration_str || "";
        
        loadYouTubeVideo(currentVideoId, 0);

        // Fetch structured topics
        fetchTopics(currentVideoId);

    } catch (error) {
        alert(error.message);
        if (progressCard) progressCard.classList.add("d-none");
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            <span>Analyze Video</span>
        `;
    }
}

// Fetch Topics
async function fetchTopics(videoId) {
    const list = document.getElementById("topics-list");
    list.innerHTML = `<div style="padding: 1rem; color: var(--text-secondary); font-size: 0.825rem; background: var(--bg-surface);">Discovering structured video chapters...</div>`;

    try {
        const res = await fetch(`/api/videos/${videoId}/topics`);
        if (!res.ok) throw new Error("Could not retrieve topics.");
        const data = await res.json();

        if (!data.topics || data.topics.length === 0) {
            list.innerHTML = `<div style="padding: 1rem; color: var(--text-muted); font-size: 0.825rem; background: var(--bg-surface);">No automated chapters generated. Use Topic Verification below to search concepts.</div>`;
            return;
        }

        list.innerHTML = "";
        data.topics.forEach((t, idx) => {
            const startSec = timestampToSeconds(t.start_timestamp);
            const subtopicsHtml = t.subtopics && t.subtopics.length > 0 
                ? `<div class="chapter-subconcepts">${t.subtopics.join(" · ")}</div>`
                : (t.summary ? `<div class="chapter-subconcepts">${t.summary}</div>` : "");
            
            const item = document.createElement("div");
            item.className = "chapter-row";
            item.onclick = () => window.seekVideo(startSec);
            item.innerHTML = `
                <div class="chapter-left">
                    <span class="chapter-idx">${String(idx + 1).padStart(2, "0")}</span>
                    <div class="chapter-content">
                        <div class="chapter-name">${t.topic}</div>
                        ${subtopicsHtml}
                    </div>
                </div>
                <button type="button" class="timestamp-action-btn" onclick="event.stopPropagation(); window.seekVideo(${startSec})">
                    <span class="ts-play-icon">▶</span> ${t.start_timestamp}
                </button>
            `;
            list.appendChild(item);
        });
    } catch (err) {
        list.innerHTML = `<div style="padding: 1rem; color: var(--text-muted); font-size: 0.825rem; background: var(--bg-surface);">Chapters available on demand via Topic Verification below.</div>`;
    }
}

// Ask Question
async function askQuestion() {
    if (!currentVideoId) {
        alert("Please analyze a video first.");
        return;
    }

    const input = document.getElementById("question-input");
    const question = input.value.trim();
    if (!question) return;

    input.value = "";
    appendChatMessage("user", question);

    const typingId = appendTypingIndicator();

    try {
        const res = await fetch(`/api/videos/${currentVideoId}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question, top_k: 5 }),
        });

        removeTypingIndicator(typingId);

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed to generate answer.");
        }

        const data = await res.json();
        appendChatMessage("assistant", data.answer, data.sources, data.is_grounded);

    } catch (error) {
        removeTypingIndicator(typingId);
        appendChatMessage("assistant", `Unable to generate answer: ${error.message}`);
    }
}

// Check Topic Presence
async function checkTopic() {
    if (!currentVideoId) {
        alert("Please analyze a video first.");
        return;
    }

    const input = document.getElementById("topic-check-input");
    const topic = input.value.trim();
    if (!topic) return;

    const resultBox = document.getElementById("topic-check-result");
    resultBox.classList.remove("d-none");
    resultBox.innerHTML = `<div style="color: var(--text-secondary); font-size: 0.8125rem;">Checking coverage of "${topic}"...</div>`;

    try {
        const res = await fetch(`/api/videos/${currentVideoId}/check-topic`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ topic: topic }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed to verify topic.");
        }

        const data = await res.json();
        
        let verdictClass = "verdict-no";
        let icon = "✕";
        if (data.confidence === "YES") {
            verdictClass = "verdict-yes";
            icon = "✓";
        } else if (data.confidence === "PARTIALLY") {
            verdictClass = "verdict-partially";
            icon = "⚠";
        }

        let timestampsHtml = "";
        if (data.relevant_timestamps && data.relevant_timestamps.length > 0) {
            timestampsHtml = `<div class="sources-row" style="margin-top: 0.5rem;">
                <span>Relevant sections:</span>
                ${data.relevant_timestamps.map(ts => `<button type="button" class="timestamp-action-btn" onclick="window.seekVideo(${ts.start_time})"><span class="ts-play-icon">▶</span> ${ts.start_timestamp} – ${ts.end_timestamp}</button>`).join("")}
            </div>`;
        }

        resultBox.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
                <span style="font-weight: 600; color: var(--text-primary); font-size: 0.84rem;">${data.topic}</span>
                <span class="verdict-badge ${verdictClass}">${icon} ${data.confidence}</span>
            </div>
            <div style="color: var(--text-secondary); font-size: 0.825rem; line-height: 1.5;">${linkifyTimestamps(data.explanation)}</div>
            ${timestampsHtml}
        `;

    } catch (err) {
        resultBox.innerHTML = `<div style="color: var(--accent-red); font-size: 0.8125rem;">Error: ${err.message}</div>`;
    }
}

// Chat UI Helpers
function appendChatMessage(role, text, sources = [], isGrounded = true) {
    const container = document.getElementById("chat-history");
    const messageItem = document.createElement("div");
    messageItem.className = `message-item ${role}`;

    let bodyHtml = "";
    if (role === "user") {
        bodyHtml = `<div class="message-body-user">${escapeHtml(text)}</div>`;
    } else {
        let formattedText = linkifyTimestamps(escapeHtml(text).replace(/\n\n/g, "</p><p>").replace(/\n/g, "<br>"));
        bodyHtml = `<div class="message-body-assistant"><p>${formattedText}</p></div>`;

        if (sources && sources.length > 0) {
            bodyHtml += `<div class="sources-row">
                <span>Citations:</span>
                ${sources.map(s => `<button type="button" class="timestamp-action-btn" onclick="window.seekVideo(${s.start_time})"><span class="ts-play-icon">▶</span> ${s.start_timestamp} – ${s.end_timestamp}</button>`).join("")}
            </div>`;
        }
    }

    messageItem.innerHTML = bodyHtml;
    container.appendChild(messageItem);
    container.scrollTop = container.scrollHeight;
}

function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function appendTypingIndicator() {
    const container = document.getElementById("chat-history");
    const id = "typing-" + Date.now();
    const item = document.createElement("div");
    item.id = id;
    item.className = "message-item assistant";
    item.innerHTML = `
        <div class="message-body-assistant" style="color: var(--text-muted); font-size: 0.8rem; display: flex; align-items: center;">
            <span class="typing-dots">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </span>
            <span>Searching transcript & synthesizing answer...</span>
        </div>
    `;
    container.appendChild(item);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// System Status & Theme Initialization
document.addEventListener("DOMContentLoaded", async () => {
    // Sync theme UI label and icon
    updateThemeUI(document.documentElement.getAttribute("data-theme") || "dark");

    try {
        const res = await fetch("/api/models");
        if (res.ok) {
            const data = await res.json();
            const badge = document.getElementById("model-status-badge");
            const modalLlm = document.getElementById("modal-llm-name");
            const modalEmb = document.getElementById("modal-emb-name");
            const modalStatus = document.getElementById("modal-ollama-status");

            if (modalLlm) modalLlm.innerText = data.default_llm || "llama3.2:3b";
            if (modalEmb) modalEmb.innerText = data.default_embedding || "BAAI/bge-small-en-v1.5";

            if (badge) {
                if (data.ollama_connected) {
                    badge.innerHTML = `<span class="status-pill-dot"></span><span>AI ready</span>`;
                    if (modalStatus) {
                        modalStatus.innerText = "Connected";
                        modalStatus.style.color = "var(--status-success)";
                    }
                } else {
                    badge.innerHTML = `<span class="status-pill-dot offline"></span><span>Ollama offline</span>`;
                    if (modalStatus) {
                        modalStatus.innerText = "Offline (run 'ollama serve')";
                        modalStatus.style.color = "var(--status-warning)";
                    }
                }
            }
        }
    } catch (e) {
        console.log("Could not query model status", e);
    }
});
