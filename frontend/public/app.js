// ──────────────────────────────────────────────
// Nav AI — Interactive Map Application
// ──────────────────────────────────────────────

// API base URL — auto-detects localhost vs production
const API_BASE = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : (window.__NAV_AI_API_BASE__ || `${window.location.origin}/api`);

let currentSessionId = null;
let userLocation = null;
let userLocationMarker = null;

// Navigation state
let navigationActive = false;
let navigationWatchId = null;
let navigationRoute = null;
let navigationStepIndex = 0;

// Initialize map
const map = L.map('map').setView([20.5937, 78.9629], 5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19
}).addTo(map);

// Layer Groups
let routeLayer = null;
let routeMarkersLayer = L.layerGroup().addTo(map);
let altRoutesLayer = L.layerGroup().addTo(map);
let poiLayer = L.layerGroup().addTo(map);
let candidateLayer = L.layerGroup().addTo(map);
let wazeAlertsLayer = L.layerGroup().addTo(map);
let wazeJamsLayer = L.layerGroup().addTo(map);

let currentPrimaryRoute = null;
let currentAlternatives = [];

// ── Waze Alert Type Icons ──
const WAZE_ALERT_ICONS = {
    ACCIDENT: { emoji: '🚨', color: '#dc2626', label: 'Accident' },
    HAZARD: { emoji: '⚠️', color: '#f97316', label: 'Hazard' },
    ROAD_CLOSED: { emoji: '🚧', color: '#991b1b', label: 'Road Closed' },
    JAM: { emoji: '🚗', color: '#eab308', label: 'Traffic Jam' },
    POLICE: { emoji: '🚔', color: '#3b82f6', label: 'Police' },
    CONSTRUCTION: { emoji: '🏗️', color: '#a16207', label: 'Construction' },
    default: { emoji: '⚪', color: '#6b7280', label: 'Alert' },
};

function getWazeAlertStyle(type) {
    return WAZE_ALERT_ICONS[type?.toUpperCase()] || WAZE_ALERT_ICONS.default;
}

// ── POI Type Icons ──
const POI_ICONS = {
    hospital: { emoji: '🏥', color: '#ef4444', label: 'Hospital' },
    fuel: { emoji: '⛽', color: '#f97316', label: 'Gas Station' },
    gas_station: { emoji: '⛽', color: '#f97316', label: 'Gas Station' },
    restaurant: { emoji: '🍽️', color: '#8b5cf6', label: 'Restaurant' },
    cafe: { emoji: '☕', color: '#a16207', label: 'Café' },
    hotel: { emoji: '🏨', color: '#0ea5e9', label: 'Hotel' },
    parking: { emoji: '🅿️', color: '#6366f1', label: 'Parking' },
    atm: { emoji: '🏧', color: '#059669', label: 'ATM' },
    charging_station: { emoji: '🔌', color: '#22c55e', label: 'EV Charging' },
    ev_charging: { emoji: '🔌', color: '#22c55e', label: 'EV Charging' },
    pharmacy: { emoji: '💊', color: '#ec4899', label: 'Pharmacy' },
    default: { emoji: '📍', color: '#6b7280', label: 'Place' }
};

function getPoiStyle(type) {
    return POI_ICONS[type?.toLowerCase()] || POI_ICONS.default;
}

// ── Haversine distance helper ──
function toRad(deg) { return (deg * Math.PI) / 180; }

function haversineMeters(a, b) {
    const R = 6371000;
    const dLat = toRad(b.lat - a.lat);
    const dLng = toRad(b.lng - a.lng);
    const h = Math.sin(dLat / 2) ** 2 +
        Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLng / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(h));
}

// ── User Location ──
function getUserLocation() {
    const locationStatus = document.getElementById('locationStatus');
    if ('geolocation' in navigator) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                userLocation = {
                    lat: position.coords.latitude,
                    lng: position.coords.longitude
                };
                map.setView([userLocation.lat, userLocation.lng], 14);

                const userIcon = L.divIcon({
                    className: 'user-marker',
                    html: '<div class="user-marker-dot"><div class="user-marker-pulse"></div></div>',
                    iconSize: [20, 20],
                    iconAnchor: [10, 10]
                });
                userLocationMarker = L.marker([userLocation.lat, userLocation.lng], { icon: userIcon, zIndexOffset: 1000 })
                    .bindPopup('<b>📍 You are here</b>')
                    .addTo(map);

                if (locationStatus) {
                    locationStatus.innerHTML = '✓ Location found';
                    locationStatus.style.background = '#10b981';
                    setTimeout(() => { locationStatus.style.display = 'none'; }, 3000);
                }
                
                // Check if start navigation button should be shown
                showStartNavButton();
            },
            (error) => {
                console.error('Geolocation error:', error);
                if (locationStatus) {
                    locationStatus.innerHTML = '⚠️ Location access denied';
                    locationStatus.style.background = '#ef4444';
                    setTimeout(() => { locationStatus.style.display = 'none'; }, 5000);
                }
            },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
        );
    }
}
getUserLocation();

// ── DOM Elements ──
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
const micButton = document.getElementById('micButton');
const ttsToggle = document.getElementById('ttsToggle');
const startNavButton = document.getElementById('startNavigationButton');
const navigationPanel = document.getElementById('navigationPanel');
const navigationInstructionEl = document.getElementById('navigationInstruction');
const navigationMetaEl = document.getElementById('navigationMeta');
const stopNavButton = document.getElementById('stopNavigationButton');

// ── Voice state ──
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let ttsEnabled = false;  // off by default, toggled by button or auto-enabled on voice input
let currentAudio = null; // currently playing TTS audio

// Sign icons for turn-by-turn
const SIGN_ICONS = {
    '-98': '↩️', '-8': '↰', '-7': '↖️', '-6': '🔄', '-3': '⬅️', '-2': '⬅️', '-1': '↙️',
    '0': '⬆️', '1': '↗️', '2': '➡️', '3': '➡️', '4': '🏁', '5': '📍', '6': '🔄', '7': '↗️', '8': '↪️'
};
function getSignIcon(sign) { return SIGN_ICONS[String(sign)] || '▶️'; }
function formatDistance(m) { return m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`; }
function formatDuration(ms) {
    const min = ms / 60000;
    if (min < 60) return `${Math.round(min)} min`;
    const h = Math.floor(min / 60);
    const m = Math.round(min % 60);
    return m > 0 ? `${h}h ${m}min` : `${h}h`;
}

// ── Simple markdown formatting ──
function formatAgentMessageContent(raw) {
    if (!raw) return '';
    const escaped = raw
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    let html = escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*\*/g, '');
    html = html
        .replace(/\r\n/g, '\n')
        .replace(/\n{2,}/g, '<br><br>')
        .replace(/\n/g, '<br>');
    return html;
}

// ── Add Message to Chat ──
function addMessage(content, isUser = false, routeData = null, intent = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'agent-message'}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    if (isUser) {
        contentDiv.textContent = content;
    } else {
        contentDiv.innerHTML = formatAgentMessageContent(content);
    }
    messageDiv.appendChild(contentDiv);

    if (routeData) {
        const routeInfo = document.createElement('div');
        routeInfo.className = 'route-info';
        routeInfo.innerHTML = `
            <strong>Distance:</strong> ${routeData.distance_km} km<br>
            <strong>Duration:</strong> ${Math.round(routeData.time_minutes)} min<br>
            <strong>From:</strong> ${routeData.from}<br>
            <strong>To:</strong> ${routeData.to}
        `;
        messageDiv.appendChild(routeInfo);

        if (routeData.detailed_instructions && routeData.detailed_instructions.length > 0) {
            const directionsPanel = document.createElement('div');
            directionsPanel.className = 'directions-panel';

            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'directions-toggle';
            toggleBtn.innerHTML = `📋 Turn-by-Turn Directions (${routeData.detailed_instructions.length} steps) <span class="toggle-arrow">▼</span>`;
            toggleBtn.onclick = () => {
                const list = directionsPanel.querySelector('.directions-list');
                const arrow = toggleBtn.querySelector('.toggle-arrow');
                list.style.display = list.style.display === 'none' ? 'block' : 'none';
                arrow.textContent = list.style.display === 'none' ? '▼' : '▲';
            };
            directionsPanel.appendChild(toggleBtn);

            const directionsList = document.createElement('div');
            directionsList.className = 'directions-list';
            directionsList.style.display = 'none';

            routeData.detailed_instructions.forEach((step, index) => {
                const stepDiv = document.createElement('div');
                if (step.sign === 4 || step.sign === 5) {
                    stepDiv.className = 'direction-step direction-step-finish';
                    stepDiv.innerHTML = `<span class="step-icon">${getSignIcon(step.sign)}</span><span class="step-text">${step.text}</span>`;
                } else {
                    stepDiv.className = 'direction-step';
                    stepDiv.innerHTML = `
                        <span class="step-number">${index + 1}</span>
                        <span class="step-icon">${getSignIcon(step.sign)}</span>
                        <div class="step-details">
                            <span class="step-text">${step.text}</span>
                            ${step.street_name ? `<span class="step-road">${step.street_name}</span>` : ''}
                            <span class="step-meta">${formatDistance(step.distance_m)} · ${formatDuration(step.time_ms)}</span>
                        </div>
                    `;
                }
                directionsList.appendChild(stepDiv);
            });

            directionsPanel.appendChild(directionsList);
            messageDiv.appendChild(directionsPanel);
        }
        
        // Add traffic analysis button ONLY for routing intent
        if (intent === 'routing') {
            const trafficButtonDiv = document.createElement('div');
            trafficButtonDiv.className = 'traffic-button-container';
            trafficButtonDiv.style.marginTop = '12px';
            
            const trafficButton = document.createElement('button');
            trafficButton.className = 'traffic-analyze-btn';
            trafficButton.innerHTML = '🚦 Analyze Traffic with Waze';
            trafficButton.onclick = () => {
                trafficButton.disabled = true;
                trafficButton.innerHTML = '🔍 Analyzing...';
                fetchTrafficData(routeData);
            };
            
            trafficButtonDiv.appendChild(trafficButton);
            messageDiv.appendChild(trafficButtonDiv);
        }
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message agent-message';
    messageDiv.id = 'loading-message';
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = 'Thinking<span class="loading"></span>';
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeLoadingMessage() {
    const loadingMsg = document.getElementById('loading-message');
    if (loadingMsg) loadingMsg.remove();
}


// ──────────────────────────────────────────────
// MAP RENDERING
// ──────────────────────────────────────────────

function drawRoute(routeData) {
    if (routeLayer) map.removeLayer(routeLayer);
    routeMarkersLayer.clearLayers();
    altRoutesLayer.clearLayers();
    wazeAlertsLayer.clearLayers();
    wazeJamsLayer.clearLayers();
    currentPrimaryRoute = routeData;

    const polyline = routeData.polyline;
    if (!polyline || polyline.length === 0) return;

    L.polyline(polyline, {
        color: '#1e3a5f', weight: 8, opacity: 0.3,
        lineCap: 'round', lineJoin: 'round'
    }).addTo(routeMarkersLayer);

    routeLayer = L.polyline(polyline, {
        color: '#3b82f6', weight: 5, opacity: 0.85,
        lineCap: 'round', lineJoin: 'round'
    }).addTo(map);

    const startIcon = L.divIcon({
        className: 'route-marker',
        html: '<div class="route-marker-pin route-start"><span class="route-marker-icon">🟢</span></div><div class="route-marker-label">Start</div>',
        iconSize: [40, 50], iconAnchor: [20, 50], popupAnchor: [0, -50]
    });
    L.marker([routeData.start_point.lat, routeData.start_point.lng], { icon: startIcon })
        .bindPopup(`<b>📍 Start</b><br>${routeData.from}`)
        .addTo(routeMarkersLayer);

    const endIcon = L.divIcon({
        className: 'route-marker',
        html: '<div class="route-marker-pin route-end"><span class="route-marker-icon">🔴</span></div><div class="route-marker-label">End</div>',
        iconSize: [40, 50], iconAnchor: [20, 50], popupAnchor: [0, -50]
    });
    L.marker([routeData.end_point.lat, routeData.end_point.lng], { icon: endIcon })
        .bindPopup(`<b>🏁 Destination</b><br>${routeData.to}`)
        .addTo(routeMarkersLayer);

    map.fitBounds(routeLayer.getBounds(), { padding: [60, 60] });

    // Show the "Start Navigation" button on the map
    showStartNavButton();
}


function addPOIMarkers(pois) {
    poiLayer.clearLayers();
    if (!pois || pois.length === 0) return;
    const bounds = [];

    pois.forEach((poi, index) => {
        const style = getPoiStyle(poi.type);
        const distText = poi.distance_km ? ` (${poi.distance_km} km)` : '';

        const icon = L.divIcon({
            className: 'poi-marker',
            html: `<div class="poi-marker-pin" style="background: ${style.color};"><span class="poi-marker-emoji">${style.emoji}</span></div><div class="poi-marker-number" style="background: ${style.color};">${index + 1}</div>`,
            iconSize: [36, 44], iconAnchor: [18, 44], popupAnchor: [0, -44]
        });

        L.marker([poi.lat, poi.lng], { icon })
            .bindPopup(`<div class="poi-popup"><div class="poi-popup-header" style="background: ${style.color};"><span>${style.emoji}</span> ${style.label}</div><div class="poi-popup-body"><b>${poi.name}</b>${distText}</div></div>`)
            .addTo(poiLayer);
        bounds.push([poi.lat, poi.lng]);
    });

    if (userLocation) bounds.push([userLocation.lat, userLocation.lng]);
    if (bounds.length > 0) map.fitBounds(L.latLngBounds(bounds), { padding: [50, 50], maxZoom: 14 });
}


function displayLocationCandidates(candidates) {
    candidateLayer.clearLayers();
    const bounds = [];

    candidates.forEach((candidate) => {
        const icon = L.divIcon({
            className: 'candidate-marker',
            html: `<div class="candidate-marker-pin"><span class="candidate-marker-number">${candidate.id}</span></div>`,
            iconSize: [36, 44], iconAnchor: [18, 44], popupAnchor: [0, -44]
        });

        const distText = candidate.distance_text ? `<br>📏 ${candidate.distance_text}` : '';
        L.marker([candidate.coordinates.lat, candidate.coordinates.lng], { icon })
            .bindPopup(`<div class="candidate-popup"><b>${candidate.id}. ${candidate.name}</b><br><small>📍 ${candidate.address}</small>${distText}<br><button class="candidate-select-btn" onclick="selectCandidate(${candidate.id}, '${candidate.name.replace(/'/g, "\\'")}')">✅ Select this location</button></div>`)
            .addTo(candidateLayer);
        bounds.push([candidate.coordinates.lat, candidate.coordinates.lng]);
    });

    if (userLocation) bounds.push([userLocation.lat, userLocation.lng]);
    if (bounds.length > 0) map.fitBounds(L.latLngBounds(bounds), { padding: [50, 50], maxZoom: 12 });
}

function selectCandidate(id, name) {
    chatInput.value = `${id}`;
    sendMessage();
}


// ──────────────────────────────────────────────
// START NAVIGATION BUTTON (on the map)
// ──────────────────────────────────────────────

function showStartNavButton() {
    if (startNavButton && !navigationActive && currentPrimaryRoute) {
        // Only show button if user is near the route start point
        if (!userLocation) {
            startNavButton.style.display = 'none';
            return;
        }
        
        const routeStart = {
            lat: currentPrimaryRoute.start_point.lat,
            lng: currentPrimaryRoute.start_point.lng
        };
        
        const distanceToStart = haversineMeters(userLocation, routeStart);
        const threshold = 100; // 100 meters
        
        if (distanceToStart <= threshold) {
            startNavButton.style.display = 'block';
        } else {
            startNavButton.style.display = 'none';
        }
    }
}

function hideStartNavButton() {
    if (startNavButton) {
        startNavButton.style.display = 'none';
    }
}

if (startNavButton) {
    startNavButton.addEventListener('click', () => {
        if (currentPrimaryRoute && currentPrimaryRoute.detailed_instructions) {
            startNavigation(currentPrimaryRoute);
        }
    });
}


// ──────────────────────────────────────────────
// LIVE NAVIGATION
// ──────────────────────────────────────────────

function updateNavigationUI() {
    if (!navigationActive || !navigationRoute) return;
    const steps = navigationRoute.detailed_instructions || [];
    if (!steps.length) return;

    const step = steps[navigationStepIndex] || steps[steps.length - 1];
    const icon = getSignIcon(step.sign);

    if (navigationInstructionEl) {
        navigationInstructionEl.textContent = `${icon}  ${step.text}${step.street_name ? ' onto ' + step.street_name : ''}`;
    }
    if (navigationMetaEl) {
        navigationMetaEl.textContent = `${formatDistance(step.distance_m || 0)} · ${formatDuration(step.time_ms || 0)}`;
    }
}

function handleNavigationPosition(position) {
    userLocation = {
        lat: position.coords.latitude,
        lng: position.coords.longitude
    };

    if (userLocationMarker) {
        userLocationMarker.setLatLng([userLocation.lat, userLocation.lng]);
    }

    if (navigationActive) {
        const zoom = map.getZoom() < 16 ? 16 : map.getZoom();
        map.setView([userLocation.lat, userLocation.lng], zoom);
    }

    if (!navigationActive || !navigationRoute) return;

    const steps = navigationRoute.detailed_instructions || [];
    const polyline = navigationRoute.polyline || [];
    if (!steps.length || !polyline.length) return;

    const currentStep = steps[navigationStepIndex];
    if (!currentStep || !currentStep.interval || currentStep.interval.length < 2) return;

    // Target the END of the current step's interval (where the maneuver happens)
    const targetIdx = currentStep.interval[1];
    const targetPt = polyline[targetIdx];
    if (!targetPt) return;

    const target = { lat: targetPt[0], lng: targetPt[1] };
    const dist = haversineMeters(userLocation, target);
    const threshold = 40;

    if (dist < threshold && navigationStepIndex < steps.length - 1) {
        navigationStepIndex += 1;
    }

    // Arrived at destination
    if (navigationStepIndex === steps.length - 1) {
        const end = { lat: navigationRoute.end_point.lat, lng: navigationRoute.end_point.lng };
        if (haversineMeters(userLocation, end) < threshold) {
            if (navigationInstructionEl) navigationInstructionEl.textContent = '🏁 You have arrived at your destination!';
            if (navigationMetaEl) navigationMetaEl.textContent = '';
            stopNavigation(false);
            return;
        }
    }

    updateNavigationUI();
}

function startNavigation(routeData) {
    if (!routeData || !routeData.detailed_instructions || !routeData.detailed_instructions.length) return;

    navigationRoute = routeData;
    navigationStepIndex = 0;
    navigationActive = true;

    hideStartNavButton();

    if (navigationPanel) navigationPanel.style.display = 'block';

    updateNavigationUI();

    if ('geolocation' in navigator) {
        if (navigationWatchId !== null) navigator.geolocation.clearWatch(navigationWatchId);
        navigationWatchId = navigator.geolocation.watchPosition(
            handleNavigationPosition,
            (err) => console.error('Navigation GPS error:', err),
            { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 }
        );
    }
}

function stopNavigation(hidePanel = true) {
    navigationActive = false;
    navigationRoute = null;
    navigationStepIndex = 0;

    if (navigationWatchId !== null && 'geolocation' in navigator) {
        navigator.geolocation.clearWatch(navigationWatchId);
        navigationWatchId = null;
    }

    if (navigationPanel && hidePanel) navigationPanel.style.display = 'none';

    // Show start button again if there's still a route
    if (currentPrimaryRoute && currentPrimaryRoute.detailed_instructions) {
        showStartNavButton();
    }
}

if (stopNavButton) {
    stopNavButton.addEventListener('click', () => stopNavigation(true));
}


// ──────────────────────────────────────────────
// SEND MESSAGE
// ──────────────────────────────────────────────

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addMessage(message, true);
    chatInput.value = '';
    addLoadingMessage();

    try {
        const token = window.getAccessToken ? window.getAccessToken() : null;
        if (!token) throw new Error('Not authenticated');

        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                message,
                session_id: currentSessionId,
                user_location: userLocation
            })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        currentSessionId = data.session_id;
        removeLoadingMessage();

        addMessage(data.message, false, data.route_data, data.intent);

        // TTS playback for agent response
        if (ttsEnabled && data.message) {
            playTTS(data.message);
        }

        if (data.route_data && data.route_data.polyline) {
            candidateLayer.clearLayers();
            drawRoute(data.route_data);

            if (data.alternative_routes && data.alternative_routes.length > 0) {
                drawAlternativeRoutes(data.alternative_routes);
            }
        }

        if (data.pois && data.pois.length > 0) {
            addPOIMarkers(data.pois);
        }

        if (data.location_candidates && data.location_candidates.length > 0) {
            displayLocationCandidates(data.location_candidates);
        }

    } catch (error) {
        removeLoadingMessage();
        console.error('Error:', error);
        if (error.message.includes('401') || error.message.includes('Not authenticated')) {
            addMessage('Authentication failed. Please login again.', false);
        } else {
            addMessage(`Sorry, I encountered an error: ${error.message}. Please try again.`, false);
        }
    }
}

if (sendButton) sendButton.addEventListener('click', sendMessage);
if (chatInput) chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });


// ──────────────────────────────────────────────
// VOICE: STT + TTS
// ──────────────────────────────────────────────

// ── TTS Toggle ──
if (ttsToggle) {
    ttsToggle.addEventListener('click', () => {
        ttsEnabled = !ttsEnabled;
        ttsToggle.textContent = ttsEnabled ? '🔈' : '🔇';
        ttsToggle.classList.toggle('active', ttsEnabled);
        ttsToggle.title = ttsEnabled ? 'Voice responses ON' : 'Voice responses OFF';
        // Stop any playing audio when disabling
        if (!ttsEnabled && currentAudio) {
            currentAudio.pause();
            currentAudio = null;
        }
    });
}

// ── TTS Playback ──
async function playTTS(text) {
    try {
        const token = window.getAccessToken ? window.getAccessToken() : null;
        if (!token) return;

        // Stop any currently playing audio
        if (currentAudio) {
            currentAudio.pause();
            currentAudio = null;
        }

        const resp = await fetch(`${API_BASE}/tts`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ text })
        });

        if (!resp.ok) {
            console.warn('TTS failed:', resp.status);
            return;
        }

        const audioBlob = await resp.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        currentAudio = new Audio(audioUrl);
        currentAudio.addEventListener('ended', () => {
            URL.revokeObjectURL(audioUrl);
            currentAudio = null;
        });
        currentAudio.play();
    } catch (err) {
        console.warn('TTS playback error:', err);
    }
}

// ── Mic Button: Voice Recording ──
if (micButton) {
    micButton.addEventListener('click', async () => {
        if (isRecording) {
            // Stop recording
            mediaRecorder.stop();
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                // Clean up recording state
                isRecording = false;
                micButton.classList.remove('recording');
                micButton.classList.add('processing');
                micButton.textContent = '⏳';
                stream.getTracks().forEach(t => t.stop());

                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

                try {
                    const token = window.getAccessToken ? window.getAccessToken() : null;
                    if (!token) throw new Error('Not authenticated');

                    const formData = new FormData();
                    formData.append('file', audioBlob, 'recording.webm');

                    const resp = await fetch(`${API_BASE}/stt`, {
                        method: 'POST',
                        headers: { 'Authorization': `Bearer ${token}` },
                        body: formData
                    });

                    if (!resp.ok) throw new Error(`STT HTTP ${resp.status}`);

                    const data = await resp.json();
                    const transcribed = (data.text || '').trim();

                    if (transcribed) {
                        // Auto-enable TTS when using voice input
                        ttsEnabled = true;
                        if (ttsToggle) {
                            ttsToggle.textContent = '🔈';
                            ttsToggle.classList.add('active');
                            ttsToggle.title = 'Voice responses ON';
                        }

                        // Put text in input and send
                        chatInput.value = transcribed;
                        sendMessage();
                    } else {
                        addMessage('I didn\'t catch that. Please try again.', false);
                    }
                } catch (err) {
                    console.error('STT error:', err);
                    addMessage('Voice transcription failed. Please try typing instead.', false);
                } finally {
                    micButton.classList.remove('processing');
                    micButton.textContent = '🎤';
                }
            };

            // Start recording
            mediaRecorder.start();
            isRecording = true;
            micButton.classList.add('recording');
            micButton.textContent = '⏹️';

        } catch (err) {
            console.error('Microphone access denied:', err);
            addMessage('Microphone access was denied. Please allow mic access in your browser settings.', false);
        }
    });
}


// ──────────────────────────────────────────────
// ALTERNATIVE ROUTES
// ──────────────────────────────────────────────

function drawAlternativeRoutes(alternatives) {
    altRoutesLayer.clearLayers();
    currentAlternatives = alternatives;

    alternatives.forEach((alt, index) => {
        if (!alt.polyline || alt.polyline.length === 0) return;

        const timeDiff = alt.time_diff_minutes || 0;
        const sign = timeDiff >= 0 ? '+' : '';
        const label = `Route ${index + 2}: ${alt.distance_km} km (${sign}${timeDiff} min)`;

        const altLine = L.polyline(alt.polyline, {
            color: '#94a3b8', weight: 5, opacity: 0.5,
            dashArray: '10, 8', lineCap: 'round', lineJoin: 'round'
        }).addTo(altRoutesLayer);

        altLine.bindTooltip(label, { sticky: true, className: 'alt-route-tooltip' });

        altLine.on('mouseover', function () { this.setStyle({ opacity: 0.85, weight: 6, color: '#64748b' }); });
        altLine.on('mouseout', function () { this.setStyle({ opacity: 0.5, weight: 5, color: '#94a3b8' }); });
        altLine.on('click', function () { switchToAlternativeRoute(index); });
    });
}

function switchToAlternativeRoute(altIndex) {
    if (!currentPrimaryRoute || !currentAlternatives[altIndex]) return;

    const oldPrimary = currentPrimaryRoute;
    const newPrimary = currentAlternatives[altIndex];

    const newAlternatives = currentAlternatives.filter((_, i) => i !== altIndex);
    const oldPrimaryAsAlt = { ...oldPrimary };
    oldPrimaryAsAlt.time_diff_minutes = Math.round((oldPrimary.time_minutes - newPrimary.time_minutes) * 10) / 10;
    oldPrimaryAsAlt.route_label = 'Previous route';
    newAlternatives.push(oldPrimaryAsAlt);

    drawRoute(newPrimary);
    drawAlternativeRoutes(newAlternatives);

    addMessage(
        `Switched to Route ${altIndex + 2}: ${newPrimary.distance_km} km, ~${Math.round(newPrimary.time_minutes)} min.`,
        false,
        newPrimary,
        'routing'  // Pass routing intent so traffic button appears
    );
}


// ──────────────────────────────────────────────
// WAZE TRAFFIC DATA
// ──────────────────────────────────────────────

async function fetchTrafficData(routeData) {
    try {
        const token = window.getAccessToken ? window.getAccessToken() : null;
        if (!token) return;

        // Show a traffic loading indicator in chat
        const trafficMsg = document.createElement('div');
        trafficMsg.className = 'message agent-message';
        trafficMsg.id = 'traffic-loading-message';
        const trafficContent = document.createElement('div');
        trafficContent.className = 'message-content';
        trafficContent.innerHTML = '🔍 Analyzing traffic conditions<span class="loading"></span>';
        trafficMsg.appendChild(trafficContent);
        chatMessages.appendChild(trafficMsg);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        const response = await fetch(`${API_BASE}/analyze-route`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ route_data: routeData })
        });

        // Remove loading message
        const loadingEl = document.getElementById('traffic-loading-message');
        if (loadingEl) loadingEl.remove();

        if (!response.ok) {
            console.error('Traffic analysis failed:', response.status);
            return;
        }

        const data = await response.json();

        // Render alerts and jams on the map
        if (data.alerts && data.alerts.length > 0) {
            renderWazeAlerts(data.alerts);
        }
        if (data.jams && data.jams.length > 0) {
            renderWazeJams(data.jams);
        }

        // Show a summary message in chat
        let summaryParts = [];
        if (data.bottleneck_analysis) {
            summaryParts.push(data.bottleneck_analysis);
        }
        if (data.alerts && data.alerts.length > 0) {
            summaryParts.push(`🚨 **${data.alerts.length} traffic alert(s)** found on your route.`);
        }
        if (data.jams && data.jams.length > 0) {
            summaryParts.push(`🚗 **${data.jams.length} traffic jam(s)** detected.`);
        }
        if (summaryParts.length === 0) {
            summaryParts.push('✅ No significant traffic issues detected on your route!');
        }
        addMessage(summaryParts.join('\n\n'), false);

    } catch (error) {
        console.error('Traffic analysis error:', error);
        const loadingEl = document.getElementById('traffic-loading-message');
        if (loadingEl) loadingEl.remove();
    }
}


function renderWazeAlerts(alerts) {
    wazeAlertsLayer.clearLayers();
    if (!alerts || alerts.length === 0) return;

    alerts.forEach((alert) => {
        if (!alert.latitude || !alert.longitude) return;

        const style = getWazeAlertStyle(alert.type);

        const icon = L.divIcon({
            className: 'waze-alert-marker',
            html: `<div class="waze-alert-pin" style="background: ${style.color};"><span class="waze-alert-emoji">${style.emoji}</span></div>`,
            iconSize: [32, 38],
            iconAnchor: [16, 38],
            popupAnchor: [0, -38]
        });

        const timeStr = alert.publish_datetime_utc
            ? new Date(alert.publish_datetime_utc).toLocaleTimeString()
            : '';

        L.marker([alert.latitude, alert.longitude], { icon })
            .bindPopup(
                `<div class="waze-popup">
                    <div class="waze-popup-header" style="background: ${style.color};">
                        ${style.emoji} ${style.label}
                    </div>
                    <div class="waze-popup-body">
                        ${alert.description ? `<b>${alert.description}</b><br>` : ''}
                        ${alert.street ? `📍 ${alert.street}` : ''}
                        ${alert.city ? `, ${alert.city}` : ''}
                        ${timeStr ? `<br>🕐 ${timeStr}` : ''}
                    </div>
                </div>`
            )
            .addTo(wazeAlertsLayer);
    });
}


function renderWazeJams(jams) {
    wazeJamsLayer.clearLayers();
    if (!jams || jams.length === 0) return;

    const JAM_COLORS = {
        1: '#fbbf24',  // yellow — light
        2: '#f59e0b',  // amber
        3: '#f97316',  // orange
        4: '#ef4444',  // red
        5: '#dc2626',  // dark red — severe
    };

    jams.forEach((jam) => {
        const color = JAM_COLORS[jam.level] || JAM_COLORS[3];

        if (jam.line && jam.line.length >= 2) {
            // Draw jam as a polyline
            const jamLine = L.polyline(jam.line, {
                color: color,
                weight: 7,
                opacity: 0.75,
                lineCap: 'round',
                lineJoin: 'round',
            }).addTo(wazeJamsLayer);

            const speedText = jam.speed_kmh ? `${jam.speed_kmh} km/h` : 'Slow';
            const lengthText = jam.length ? `${(jam.length / 1000).toFixed(1)} km` : '';

            jamLine.bindPopup(
                `<div class="waze-popup">
                    <div class="waze-popup-header" style="background: ${color};">
                        🚗 Traffic Jam (Level ${jam.level}/5)
                    </div>
                    <div class="waze-popup-body">
                        ${jam.street ? `📍 ${jam.street}<br>` : ''}
                        🏎️ Speed: ${speedText}
                        ${lengthText ? `<br>📏 Length: ${lengthText}` : ''}
                    </div>
                </div>`
            );

            jamLine.bindTooltip(`🚗 Jam Lvl ${jam.level} · ${speedText}`, { sticky: true, className: 'waze-jam-tooltip' });
        }
    });
}


// ──────────────────────────────────────────────
// CONVERSATION MANAGEMENT
// ──────────────────────────────────────────────

const conversationSidebar = document.getElementById('conversationSidebar');
const conversationList = document.getElementById('conversationList');
const newChatButton = document.getElementById('newChatButton');
const sidebarToggle = document.getElementById('sidebarToggle');

// Restore session from localStorage
(function restoreSession() {
    const saved = localStorage.getItem('navai_current_session');
    if (saved) {
        currentSessionId = saved;
    }
})();

// Save session to localStorage whenever it changes
function persistSession() {
    if (currentSessionId) {
        localStorage.setItem('navai_current_session', currentSessionId);
    } else {
        localStorage.removeItem('navai_current_session');
    }
}

// Time formatting for sidebar items
function timeAgo(dateStr) {
    if (!dateStr) return '';
    const now = new Date();
    const then = new Date(dateStr);
    const diffMs = now - then;
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return 'Just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    return then.toLocaleDateString();
}

// Load conversations list from backend
async function loadConversations() {
    try {
        const token = window.getAccessToken ? window.getAccessToken() : null;
        if (!token) return;

        const response = await fetch(`${API_BASE}/conversations`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!response.ok) return;

        const data = await response.json();
        renderConversationList(data.conversations || []);
    } catch (error) {
        console.error('Failed to load conversations:', error);
    }
}

// Render the conversation list in the sidebar
function renderConversationList(conversations) {
    if (!conversationList) return;

    if (conversations.length === 0) {
        conversationList.innerHTML = '<div class="conversation-empty">No conversations yet.<br>Start chatting!</div>';
        return;
    }

    conversationList.innerHTML = '';

    conversations.forEach(conv => {
        const item = document.createElement('div');
        item.className = 'conversation-item' + (conv.session_id === currentSessionId ? ' active' : '');

        const title = document.createElement('div');
        title.className = 'conversation-item-title';
        title.textContent = conv.title || 'New Chat';

        const meta = document.createElement('div');
        meta.className = 'conversation-item-meta';
        meta.textContent = timeAgo(conv.updated_at);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'conversation-item-delete';
        deleteBtn.innerHTML = '✕';
        deleteBtn.title = 'Delete';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            deleteConversation(conv.session_id);
        };

        item.appendChild(title);
        item.appendChild(meta);
        item.appendChild(deleteBtn);

        item.onclick = () => switchConversation(conv.session_id);

        conversationList.appendChild(item);
    });
}

// Switch to a different conversation
async function switchConversation(sessionId, forceReload = false) {
    if (sessionId === currentSessionId && !forceReload) return;

    try {
        const token = window.getAccessToken ? window.getAccessToken() : null;
        if (!token) return;

        const response = await fetch(`${API_BASE}/conversations/${sessionId}`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!response.ok) {
            console.error('Failed to load conversation:', response.status);
            return;
        }

        const data = await response.json();

        // Update current session
        currentSessionId = sessionId;
        persistSession();

        // Clear map layers
        if (routeLayer) map.removeLayer(routeLayer);
        routeMarkersLayer.clearLayers();
        altRoutesLayer.clearLayers();
        wazeAlertsLayer.clearLayers();
        wazeJamsLayer.clearLayers();
        poiLayer.clearLayers();
        candidateLayer.clearLayers();
        currentPrimaryRoute = null;
        currentAlternatives = [];

        // Populate chat messages
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }

        if (data.messages && data.messages.length > 0) {
            data.messages.forEach(msg => {
                addMessage(msg.content, msg.role === 'user', null, null);
            });
        } else {
            addMessage('Welcome back! Where would you like to go?', false);
        }

        // Restore route on map
        if (data.route_data && data.route_data.polyline) {
            drawRoute(data.route_data);
            if (data.alternative_routes && data.alternative_routes.length > 0) {
                drawAlternativeRoutes(data.alternative_routes);
            }
        }

        // Update active state in sidebar
        loadConversations();

    } catch (error) {
        console.error('Error switching conversation:', error);
    }
}

// Start a new chat
function newChat() {
    // Summarize the old conversation in the background before clearing
    if (currentSessionId) {
        const oldSessionId = currentSessionId;
        const token = window.getAccessToken ? window.getAccessToken() : null;
        if (token) {
            fetch(`${API_BASE}/conversations/${oldSessionId}/summarize`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            }).then(() => {
                console.log('🧠 Summarization started for', oldSessionId.slice(0, 8));
            }).catch(err => console.error('Summarize failed:', err));
        }
    }

    currentSessionId = null;
    persistSession();

    // Clear map layers
    if (routeLayer) map.removeLayer(routeLayer);
    routeMarkersLayer.clearLayers();
    altRoutesLayer.clearLayers();
    wazeAlertsLayer.clearLayers();
    wazeJamsLayer.clearLayers();
    poiLayer.clearLayers();
    candidateLayer.clearLayers();
    currentPrimaryRoute = null;
    currentAlternatives = [];
    hideStartNavButton();

    // Stop any active navigation
    if (navigationActive) stopNavigation(true);

    // Reset chat
    if (chatMessages) {
        chatMessages.innerHTML = `
            <div class="message agent-message">
                <div class="message-content">
                    Welcome! Where would you like to go?
                </div>
            </div>
        `;
    }

    // Update sidebar
    loadConversations();
}

// Delete a conversation
async function deleteConversation(sessionId) {
    try {
        const token = window.getAccessToken ? window.getAccessToken() : null;
        if (!token) return;

        const response = await fetch(`${API_BASE}/conversations/${sessionId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!response.ok) {
            console.error('Failed to delete conversation:', response.status);
            return;
        }

        // If we deleted the current conversation, start a new one
        if (sessionId === currentSessionId) {
            newChat();
        } else {
            loadConversations();
        }

    } catch (error) {
        console.error('Error deleting conversation:', error);
    }
}

// Toggle sidebar
function toggleSidebar() {
    if (conversationSidebar) {
        conversationSidebar.classList.toggle('collapsed');
    }
}

// Event listeners
if (newChatButton) newChatButton.addEventListener('click', newChat);
if (sidebarToggle) sidebarToggle.addEventListener('click', toggleSidebar);

// Override the original sendMessage to also persist session and refresh sidebar
const _originalSendMessage = sendMessage;
sendMessage = async function () {
    await _originalSendMessage();
    persistSession();
    // Refresh conversation list after sending a message (slight delay for DB write)
    setTimeout(loadConversations, 500);
};

// Re-bind event listeners with the new sendMessage
if (sendButton) {
    sendButton.removeEventListener('click', _originalSendMessage);
    sendButton.addEventListener('click', sendMessage);
}
if (chatInput) {
    chatInput.removeEventListener('keypress', () => {});
    chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
}

// Initialize data that depends on authentication (conversations + knowledge)
function initAuthDependentData() {
    const token = window.getAccessToken ? window.getAccessToken() : null;
    if (!token) return;

    setTimeout(loadConversations, 300);
    setTimeout(loadKnowledge, 500);

    // Auto-load the last conversation if one was saved in localStorage
    if (currentSessionId) {
        setTimeout(() => switchConversation(currentSessionId, true), 600);
    }
}

// Load conversations when auth is ready (custom event from auth-client.js),
// but also handle the case where auth was ready before this script loaded.
if (window.navaiAuthReady) {
    // Auth was already initialized before app.js registered the listener
    initAuthDependentData();
} else {
    window.addEventListener('navai-auth-ready', () => {
        initAuthDependentData();
    });
}


// ──────────────────────────────────────────────
// SIDEBAR TABS
// ──────────────────────────────────────────────

const tabChats = document.getElementById('tabChats');
const tabKnowledge = document.getElementById('tabKnowledge');
const chatsTab = document.getElementById('chatsTab');
const knowledgeTab = document.getElementById('knowledgeTab');
const knowledgeContainer = document.getElementById('knowledgeContainer');
const refreshKnowledge = document.getElementById('refreshKnowledge');

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.sidebar-tab-content').forEach(c => c.classList.remove('active'));

    if (tabName === 'chats') {
        tabChats && tabChats.classList.add('active');
        chatsTab && chatsTab.classList.add('active');
    } else {
        tabKnowledge && tabKnowledge.classList.add('active');
        knowledgeTab && knowledgeTab.classList.add('active');
        loadKnowledge();
    }
}

if (tabChats) tabChats.addEventListener('click', () => switchTab('chats'));
if (tabKnowledge) tabKnowledge.addEventListener('click', () => switchTab('knowledge'));
if (refreshKnowledge) refreshKnowledge.addEventListener('click', loadKnowledge);


const CATEGORY_ICONS = {
    personality: '🎭',
    travel: '🚗',
    places: '📍',
    preferences: '⚙️',
    patterns: '🔄',
    general: '📝',
};

const CATEGORY_LABELS = {
    personality: 'Personality & Tone',
    travel: 'Travel & Routes',
    places: 'Places & Locations',
    preferences: 'Preferences',
    patterns: 'Patterns & Habits',
    general: 'General',
};

async function loadKnowledge() {
    try {
        const token = window.getAccessToken ? window.getAccessToken() : null;
        if (!token) return;

        const response = await fetch(`${API_BASE}/knowledge`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!response.ok) return;

        const data = await response.json();
        renderKnowledge(data.knowledge || []);
    } catch (error) {
        console.error('Failed to load knowledge:', error);
    }
}

function renderKnowledge(items) {
    if (!knowledgeContainer) return;

    if (items.length === 0) {
        knowledgeContainer.innerHTML = `
            <div class="knowledge-empty">
                <div class="knowledge-empty-icon">🧠</div>
                <p>No memories yet.</p>
                <p class="knowledge-empty-hint">
                    I'll learn your preferences, favorite places, and frequent routes as you chat with me.
                </p>
            </div>
        `;
        return;
    }

    // Group by display_category
    const grouped = {};
    items.forEach(item => {
        const cat = item.display_category || 'general';
        if (!grouped[cat]) grouped[cat] = [];
        grouped[cat].push(item);
    });

    let html = '';
    const catOrder = ['personality', 'travel', 'places', 'preferences', 'patterns', 'general'];

    catOrder.forEach(cat => {
        if (!grouped[cat]) return;

        const icon = CATEGORY_ICONS[cat] || '📝';
        const label = CATEGORY_LABELS[cat] || formatKey(cat);

        html += `<div class="knowledge-group">`;
        html += `<div class="knowledge-group-title">${icon} ${label}</div>`;

        grouped[cat].forEach(item => {
            const conf = Math.round((item.confidence || 0) * 100);
            const confClass = conf >= 80 ? 'high' : conf >= 50 ? 'medium' : 'low';
            const safetyBadge = item.safety_level === 'explicit'
                ? '<span class="knowledge-safety explicit">explicit</span>'
                : '<span class="knowledge-safety inferred">inferred</span>';
            const typeLabel = formatKey(item.knowledge_type || '');
            const detail = formatKnowledgeDetail(item.value);

            html += `
                <div class="knowledge-card">
                    <div class="knowledge-card-header">
                        <span class="knowledge-card-key">${formatKey(item.key)}</span>
                        <span class="knowledge-confidence ${confClass}">${conf}%</span>
                    </div>
                    <div class="knowledge-card-type">${typeLabel} ${safetyBadge}</div>
                    <div class="knowledge-card-detail">${detail}</div>
                    <div class="knowledge-confidence-bar">
                        <div class="knowledge-confidence-fill ${confClass}" style="width: ${conf}%"></div>
                    </div>
                    <div class="knowledge-card-meta">Seen ${item.occurrence_count || 1}×</div>
                </div>
            `;
        });

        html += `</div>`;
    });

    knowledgeContainer.innerHTML = html;
}

function formatKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatKnowledgeDetail(value) {
    if (!value || typeof value !== 'object') return String(value || '');

    // Universal renderer — works with any dynamic entity type
    const parts = [];

    // Always show description first if present
    if (value.description) {
        parts.push(value.description);
    }

    // Render remaining fields
    for (const [k, v] of Object.entries(value)) {
        if (k === 'description') continue;
        if (v === null || v === undefined || v === '') continue;

        const label = formatKey(k);
        if (typeof v === 'object') {
            parts.push(`${label}: ${JSON.stringify(v)}`);
        } else {
            parts.push(`${label}: ${v}`);
        }
    }

    return parts.join('<br>');
}


// ──────────────────────────────────────────────
// MOBILE RESPONSIVE HANDLING
// ──────────────────────────────────────────────

const mobileSidebarOverlay = document.getElementById('mobileSidebarOverlay');
const mobilePanelToggle = document.getElementById('mobilePanelToggle');

let mobileViewState = 'map';

function isMobileView() {
    return window.innerWidth <= 768;
}

function toggleMobileSidebar(open) {
    if (!conversationSidebar) return;
    
    if (open) {
        conversationSidebar.classList.add('mobile-open');
        conversationSidebar.classList.remove('collapsed');
        if (mobileSidebarOverlay) mobileSidebarOverlay.classList.add('active');
    } else {
        conversationSidebar.classList.remove('mobile-open');
        if (mobileSidebarOverlay) mobileSidebarOverlay.classList.remove('active');
    }
}

function toggleMobilePanel() {
    const chatPanel = document.querySelector('.chat-panel');
    const mapPanel = document.querySelector('.map-panel');
    
    if (!chatPanel || !mapPanel || !mobilePanelToggle) return;
    
    if (mobileViewState === 'map') {
        chatPanel.style.height = '70vh';
        mapPanel.style.height = '30vh';
        chatPanel.style.order = '1';
        mapPanel.style.order = '2';
        mobilePanelToggle.textContent = '🗺️';
        mobileViewState = 'chat';
    } else {
        chatPanel.style.height = '50vh';
        mapPanel.style.height = '50vh';
        chatPanel.style.order = '2';
        mapPanel.style.order = '1';
        mobilePanelToggle.textContent = '💬';
        mobileViewState = 'map';
    }
    
    setTimeout(() => {
        if (map) map.invalidateSize();
    }, 300);
}

if (mobilePanelToggle) {
    mobilePanelToggle.addEventListener('click', toggleMobilePanel);
}

if (mobileSidebarOverlay) {
    mobileSidebarOverlay.addEventListener('click', () => toggleMobileSidebar(false));
}

if (sidebarToggle) {
    sidebarToggle.removeEventListener('click', toggleSidebar);
    sidebarToggle.addEventListener('click', () => {
        if (isMobileView()) {
            const isOpen = conversationSidebar && conversationSidebar.classList.contains('mobile-open');
            toggleMobileSidebar(!isOpen);
        } else {
            toggleSidebar();
        }
    });
}

function handleResize() {
    const chatPanel = document.querySelector('.chat-panel');
    const mapPanel = document.querySelector('.map-panel');
    
    if (!isMobileView()) {
        if (chatPanel) {
            chatPanel.style.height = '';
            chatPanel.style.order = '';
        }
        if (mapPanel) {
            mapPanel.style.height = '';
            mapPanel.style.order = '';
        }
        toggleMobileSidebar(false);
        mobileViewState = 'map';
    }
    
    setTimeout(() => {
        if (map) map.invalidateSize();
    }, 100);
}

window.addEventListener('resize', handleResize);

document.addEventListener('DOMContentLoaded', () => {
    handleResize();
});

function focusOnChat() {
    if (isMobileView() && mobileViewState === 'map') {
        toggleMobilePanel();
    }
}

const _originalAddMessage = addMessage;
addMessage = function(content, isUser = false, routeData = null, intent = null) {
    _originalAddMessage(content, isUser, routeData, intent);
    
    if (!isUser && isMobileView() && mobileViewState === 'map') {
        const chatPanel = document.querySelector('.chat-panel');
        if (chatPanel) {
            chatPanel.style.height = '60vh';
            const mapPanel = document.querySelector('.map-panel');
            if (mapPanel) mapPanel.style.height = '40vh';
        }
    }
};

const _originalDrawRoute = drawRoute;
drawRoute = function(routeData) {
    _originalDrawRoute(routeData);
    
    if (isMobileView()) {
        setTimeout(() => {
            if (map) map.invalidateSize();
        }, 100);
    }
};
