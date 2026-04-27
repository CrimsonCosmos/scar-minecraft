/**
 * SCAR-Minecraft — Renderer logic.
 *
 * Manages UI state, IPC calls, DOM updates, log viewer,
 * entity radar, hotbar display, and connection presets.
 */

// ─── State ──────────────────────────────────────────────

let uiState = 'idle'; // idle | relay_starting | relay_waiting | relay_running | agent_running
let launcherStatus = 'idle';
let pythonStatus = 'idle';
let logFilter = 'all'; // all | controller | python
let logAutoScroll = true;
let currentProtocol = 'bedrock';
let currentConnectType = 'realm';

const CATEGORY_NAMES = ['', 'Sword', 'Pick', 'Axe', 'Shovel', 'Ranged', 'Food', 'Block', 'Other'];
const CATEGORY_CLASSES = ['empty', 'sword', 'pickaxe', 'axe', 'shovel', 'ranged', 'food', 'block', 'other'];

// ─── DOM Refs ───────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const els = {
  headerStatus: $('header-status'),
  // Connection
  protocolGroup: $('protocol-group'),
  connectTypeGroup: $('connect-type-group'),
  realmFields: $('realm-fields'),
  serverFields: $('server-fields'),
  worldFields: $('world-fields'),
  realmUrl: $('realm-url'),
  serverHost: $('server-host'),
  serverPort: $('server-port'),
  worldSelect: $('world-select'),
  worldRefresh: $('world-refresh'),
  listenPort: $('listen-port'),
  bridgePort: $('bridge-port'),
  optStealth: $('opt-stealth'),
  optKeyboard: $('opt-keyboard'),
  optOnline: $('opt-online'),
  pvpStyle: $('pvp-style'),
  phaseSelect: $('phase-select'),
  mcWindowsInfo: $('mc-windows-info'),
  detectWindows: $('detect-windows'),
  presetSelect: $('preset-select'),
  presetSave: $('preset-save'),
  presetDelete: $('preset-delete'),
  // Control
  btnStartRelay: $('btn-start-relay'),
  btnStopRelay: $('btn-stop-relay'),
  btnStartAgent: $('btn-start-agent'),
  btnStopAgent: $('btn-stop-agent'),
  btnStopAll: $('btn-stop-all'),
  botToggle: $('bot-toggle'),
  btnAutoLaunch: $('btn-auto-launch'),
  launchSteps: $('launch-steps'),
  // Agent config
  agentSteps: $('agent-steps'),
  agentTransfer: $('agent-transfer'),
  agentTransferBrowse: $('agent-transfer-browse'),
  agentSave: $('agent-save'),
  agentSaveBrowse: $('agent-save-browse'),
  optObserve: $('opt-observe'),
  optScreenCapture: $('opt-screen-capture'),
  optFactored: $('opt-factored'),
  optConverge: $('opt-converge'),
  // Status
  healthBar: $('health-bar'),
  healthValue: $('health-value'),
  foodBar: $('food-bar'),
  foodValue: $('food-value'),
  positionValue: $('position-value'),
  orientationValue: $('orientation-value'),
  timeValue: $('time-value'),
  aliveValue: $('alive-value'),
  entityList: $('entity-list'),
  entityRadar: $('entity-radar'),
  hotbar: $('hotbar'),
  // Agent stats
  statStep: $('stat-step'),
  statVitality: $('stat-vitality'),
  statSurprise: $('stat-surprise'),
  statPatterns: $('stat-patterns'),
  statKd: $('stat-kd'),
  statUrgency: $('stat-urgency'),
  // Logs
  logOutput: $('log-output'),
  logClear: $('log-clear'),
};

// ─── Toggle Groups ──────────────────────────────────────

function setupToggleGroup(container, callback) {
  container.addEventListener('click', (e) => {
    const btn = e.target.closest('.toggle-btn');
    if (!btn) return;
    container.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    callback(btn.dataset.value);
  });
}

setupToggleGroup(els.protocolGroup, (val) => {
  currentProtocol = val;
  // Auto-adjust defaults
  if (val === 'java') {
    els.serverPort.placeholder = '25565';
    if (currentConnectType === 'realm') {
      // Switch to server for Java
      setConnectType('server');
    }
  } else {
    els.serverPort.placeholder = '19132';
  }
  updateConnectFields();
});

setupToggleGroup(els.connectTypeGroup, (val) => {
  setConnectType(val);
});

function setConnectType(type) {
  currentConnectType = type;
  els.connectTypeGroup.querySelectorAll('.toggle-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.value === type);
  });
  updateConnectFields();
}

function updateConnectFields() {
  document.querySelectorAll('.connect-fields').forEach(el => el.classList.add('hidden'));
  if (currentConnectType === 'realm') els.realmFields.classList.remove('hidden');
  if (currentConnectType === 'server') els.serverFields.classList.remove('hidden');
  if (currentConnectType === 'world') els.worldFields.classList.remove('hidden');

  // Realm only for Bedrock
  const realmBtn = els.connectTypeGroup.querySelector('[data-value="realm"]');
  if (currentProtocol === 'java') {
    realmBtn.style.opacity = '0.3';
    realmBtn.style.pointerEvents = 'none';
  } else {
    realmBtn.style.opacity = '';
    realmBtn.style.pointerEvents = '';
  }
}

// ─── Log Tabs ───────────────────────────────────────────

document.querySelectorAll('.log-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.log-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    logFilter = tab.dataset.filter;
    refreshLogVisibility();
  });
});

function refreshLogVisibility() {
  els.logOutput.querySelectorAll('.log-line').forEach(line => {
    if (logFilter === 'all') {
      line.style.display = '';
    } else {
      line.style.display = line.classList.contains(logFilter) ? '' : 'none';
    }
  });
}

// ─── Log Output ─────────────────────────────────────────

function addLogLine(text, source) {
  const line = document.createElement('div');
  line.className = `log-line ${source}`;
  line.textContent = text;

  // Apply filter
  if (logFilter !== 'all' && !line.classList.contains(logFilter)) {
    line.style.display = 'none';
  }

  els.logOutput.appendChild(line);

  // Trim old lines
  while (els.logOutput.children.length > 2000) {
    els.logOutput.removeChild(els.logOutput.firstChild);
  }

  // Auto-scroll
  if (logAutoScroll) {
    els.logOutput.scrollTop = els.logOutput.scrollHeight;
  }
}

// Detect manual scroll
els.logOutput.addEventListener('scroll', () => {
  const el = els.logOutput;
  logAutoScroll = (el.scrollTop + el.clientHeight >= el.scrollHeight - 20);
});

els.logClear.addEventListener('click', () => {
  els.logOutput.innerHTML = '';
});

// ─── Build Config from Form ─────────────────────────────

function buildLauncherConfig() {
  const protocol = currentConnectType === 'world' ? 'java' : currentProtocol;
  const config = {
    protocol,
    bridgePort: parseInt(els.bridgePort.value) || 3001,
    stealth: els.optStealth.checked,
    keyboardFallback: els.optKeyboard.checked,
    onlineMode: els.optOnline.checked,
    pvpStyle: els.pvpStyle.value,
    phase: parseInt(els.phaseSelect.value),
  };

  const listenPort = parseInt(els.listenPort.value);
  if (listenPort) config.listenPort = listenPort;

  if (currentConnectType === 'realm') {
    config.realmInvite = els.realmUrl.value.trim();
  } else if (currentConnectType === 'server') {
    config.serverHost = els.serverHost.value.trim();
    const port = parseInt(els.serverPort.value);
    if (port) config.serverPort = port;
  } else if (currentConnectType === 'world') {
    config.worldPath = els.worldSelect.value;
  }

  return config;
}

function buildPythonConfig() {
  return {
    host: 'localhost',
    port: parseInt(els.bridgePort.value) || 3001,
    steps: parseInt(els.agentSteps.value) || 10000,
    phase: parseInt(els.phaseSelect.value),
    transfer: els.agentTransfer.value.trim() || undefined,
    save: els.agentSave.value.trim() || undefined,
    observeOnly: els.optObserve.checked,
    screenCapture: els.optScreenCapture.checked,
    factored: els.optFactored.checked,
    untilConverge: els.optConverge.checked,
  };
}

// ─── UI State Management ────────────────────────────────

function updateUI() {
  const relayRunning = ['starting', 'waiting_client', 'running'].includes(launcherStatus);
  const relayReady = launcherStatus === 'running';
  const pyRunning = pythonStatus === 'running';

  // Determine overall state
  if (!relayRunning && !pyRunning) uiState = 'idle';
  else if (launcherStatus === 'starting') uiState = 'relay_starting';
  else if (launcherStatus === 'waiting_client') uiState = 'relay_waiting';
  else if (relayReady && !pyRunning) uiState = 'relay_running';
  else if (relayReady && pyRunning) uiState = 'agent_running';

  // Header badge
  els.headerStatus.textContent = launcherStatus;
  els.headerStatus.className = `status-badge ${launcherStatus}`;

  // Buttons
  els.btnStartRelay.disabled = relayRunning;
  els.btnStopRelay.disabled = !relayRunning;
  els.btnStartAgent.disabled = !relayReady || pyRunning;
  els.btnStopAgent.disabled = !pyRunning;
  els.btnStopAll.disabled = !relayRunning && !pyRunning;
  els.btnAutoLaunch.disabled = relayRunning || pyRunning;

  // Bot toggle
  els.botToggle.disabled = !relayReady;

  // Connection form - disable when running (but NOT agent settings)
  const formInputs = document.querySelectorAll('#connection-panel input, #connection-panel select, #connection-panel .toggle-btn');
  formInputs.forEach(el => {
    // Agent config stays editable while relay runs (user sets save path, steps, etc.)
    if (el.closest('#agent-config')) return;
    if (relayRunning) {
      el.disabled = true;
      if (el.classList?.contains('toggle-btn')) el.style.pointerEvents = 'none';
    } else {
      el.disabled = false;
      if (el.classList?.contains('toggle-btn')) el.style.pointerEvents = '';
    }
  });
}

// ─── Button Handlers ────────────────────────────────────

els.btnStartRelay.addEventListener('click', async () => {
  const config = buildLauncherConfig();
  addLogLine('Starting relay...', 'controller');
  const result = await window.scar.startLauncher(config);
  if (!result.ok) {
    addLogLine(`Error: ${result.error}`, 'controller error');
  }
});

els.btnStopRelay.addEventListener('click', async () => {
  await window.scar.stopLauncher();
  launcherStatus = 'idle';
  pythonStatus = 'idle';
  updateUI();
});

els.btnStartAgent.addEventListener('click', async () => {
  const config = buildPythonConfig();
  addLogLine('Starting Python agent...', 'python');
  const result = await window.scar.startPython(config);
  if (!result.ok) {
    addLogLine(`Error: ${result.error}`, 'python error');
  } else {
    pythonStatus = 'running';
    updateUI();
  }
});

els.btnStopAgent.addEventListener('click', async () => {
  await window.scar.stopPython();
  pythonStatus = 'idle';
  updateUI();
});

els.btnStopAll.addEventListener('click', async () => {
  await window.scar.stopPython();
  await window.scar.stopLauncher();
  launcherStatus = 'idle';
  pythonStatus = 'idle';
  updateUI();
});

els.botToggle.addEventListener('click', async () => {
  const isOn = els.botToggle.classList.contains('on');
  const result = await window.scar.setBotControl(!isOn);
  if (result.botControlActive) {
    els.botToggle.classList.remove('off');
    els.botToggle.classList.add('on');
    els.botToggle.textContent = 'ON';
  } else {
    els.botToggle.classList.remove('on');
    els.botToggle.classList.add('off');
    els.botToggle.textContent = 'OFF';
  }
});

// ─── Auto-Launch Sequence ───────────────────────────────

els.btnAutoLaunch.addEventListener('click', async () => {
  const steps = els.launchSteps.querySelectorAll('.step');
  const setStep = (name, cls) => {
    steps.forEach(s => {
      if (s.dataset.step === name) {
        s.className = `step ${cls}`;
      }
    });
  };

  // Step 1: Start relay
  setStep('relay', 'active');
  const config = buildLauncherConfig();
  const relayResult = await window.scar.startLauncher(config);
  if (!relayResult.ok) {
    addLogLine(`Launch failed: ${relayResult.error}`, 'controller error');
    setStep('relay', '');
    return;
  }
  setStep('relay', 'done');

  // Step 2: Wait for client
  setStep('client', 'active');
  await waitForStatus('running', 120000);
  setStep('client', 'done');

  // Step 3: Start agent
  setStep('agent', 'active');
  const pyConfig = buildPythonConfig();
  const pyResult = await window.scar.startPython(pyConfig);
  if (!pyResult.ok) {
    addLogLine(`Agent launch failed: ${pyResult.error}`, 'python error');
    setStep('agent', '');
    return;
  }
  pythonStatus = 'running';
  updateUI();
  setStep('agent', 'done');

  // Step 4: Bot control is enabled by the Python runner automatically
  setStep('bot', 'done');
});

function waitForStatus(target, timeout) {
  return new Promise((resolve) => {
    const start = Date.now();
    const check = () => {
      if (launcherStatus === target || Date.now() - start > timeout) {
        resolve();
        return;
      }
      setTimeout(check, 500);
    };
    check();
  });
}

// ─── Window Detection ───────────────────────────────────

els.detectWindows.addEventListener('click', async () => {
  els.mcWindowsInfo.textContent = 'Scanning...';
  const windows = await window.scar.detectMinecraftWindows();
  if (windows.length === 0) {
    els.mcWindowsInfo.textContent = 'No Minecraft windows found';
  } else {
    els.mcWindowsInfo.textContent = windows.map(w => w.title).join(', ');
  }
});

// ─── World List ─────────────────────────────────────────

async function loadWorlds() {
  const worlds = await window.scar.listWorlds();
  els.worldSelect.innerHTML = '';
  if (worlds.length === 0) {
    els.worldSelect.innerHTML = '<option value="">No worlds found</option>';
  } else {
    for (const w of worlds) {
      const opt = document.createElement('option');
      opt.value = w.path;
      opt.textContent = w.name;
      els.worldSelect.appendChild(opt);
    }
  }
}

els.worldRefresh.addEventListener('click', loadWorlds);

// ─── File Browsers ──────────────────────────────────────

els.agentTransferBrowse.addEventListener('click', async () => {
  const file = await window.scar.selectFile({ filters: [{ name: 'Pickle', extensions: ['pkl'] }] });
  if (file) els.agentTransfer.value = file;
});

els.agentSaveBrowse.addEventListener('click', async () => {
  const file = await window.scar.selectSaveFile({ filters: [{ name: 'Pickle', extensions: ['pkl'] }] });
  if (file) els.agentSave.value = file;
});

// ─── State Updates ──────────────────────────────────────

window.scar.onStateUpdate((state) => {
  if (!state) return;

  // Health
  const hp = state.health || 0;
  els.healthBar.style.width = `${(hp / 20) * 100}%`;
  els.healthValue.textContent = `${hp.toFixed(0)} / 20`;

  // Food
  const food = state.food || 0;
  els.foodBar.style.width = `${(food / 20) * 100}%`;
  els.foodValue.textContent = `${food.toFixed(0)} / 20`;

  // Position
  if (state.position) {
    const p = state.position;
    els.positionValue.textContent = `${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}`;
  }

  // Orientation
  const yaw = ((state.yaw || 0) * 180 / Math.PI).toFixed(1);
  const pitch = ((state.pitch || 0) * 180 / Math.PI).toFixed(1);
  els.orientationValue.textContent = `${yaw}\u00b0 / ${pitch}\u00b0`;

  // Time
  const t = state.time_of_day || 0;
  const period = t < 6000 ? 'Morning' : t < 12000 ? 'Day' : t < 18000 ? 'Evening' : 'Night';
  els.timeValue.textContent = `${t} (${period})`;

  // Alive
  els.aliveValue.textContent = state.alive === false ? 'DEAD' : 'Yes';
  els.aliveValue.style.color = state.alive === false ? '#e74c3c' : '';

  // Bot toggle sync
  if (state.bot_control_active) {
    els.botToggle.classList.remove('off');
    els.botToggle.classList.add('on');
    els.botToggle.textContent = 'ON';
  } else {
    els.botToggle.classList.remove('on');
    els.botToggle.classList.add('off');
    els.botToggle.textContent = 'OFF';
  }

  // Entities
  updateEntityList(state.entities);

  // Radar
  drawRadar(state);

  // Hotbar
  updateHotbar(state.inventory);
});

// ─── Entity List ────────────────────────────────────────

function updateEntityList(entities) {
  if (!entities) {
    els.entityList.innerHTML = '<span class="muted">No data</span>';
    return;
  }

  let html = '';
  const addEntities = (list, cls) => {
    if (!list) return;
    for (const e of list) {
      if (!e || !e.name) continue;
      const dist = (e.distance || 0).toFixed(1);
      html += `<div class="entity-row ${cls}"><span class="entity-name">${e.name}</span><span class="entity-dist">${dist}m</span></div>`;
    }
  };

  addEntities(entities.hostiles, 'hostile');
  addEntities(entities.players, 'player');
  addEntities(entities.passives, 'passive');

  els.entityList.innerHTML = html || '<span class="muted">None nearby</span>';
}

// ─── Entity Radar ───────────────────────────────────────

function drawRadar(state) {
  const canvas = els.entityRadar;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const radius = 60;
  const scale = radius / 32; // 32 blocks = edge of radar

  ctx.clearRect(0, 0, w, h);

  // Background circle
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fillStyle = '#0a0f1a';
  ctx.fill();
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Cross-hairs
  ctx.strokeStyle = '#1a1a3a';
  ctx.beginPath();
  ctx.moveTo(cx - radius, cy);
  ctx.lineTo(cx + radius, cy);
  ctx.moveTo(cx, cy - radius);
  ctx.lineTo(cx, cy + radius);
  ctx.stroke();

  // Player dot
  ctx.beginPath();
  ctx.arc(cx, cy, 3, 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.fill();

  if (!state || !state.entities || !state.position) return;

  const playerYaw = state.yaw || 0;
  const playerPos = state.position;

  const drawEntities = (list, color) => {
    if (!list) return;
    for (const e of list) {
      if (!e || e.distance == null) continue;
      // Use velocity direction as a rough angle indicator, or spread evenly
      const dist = Math.min(e.distance, 32) * scale;
      // Without exact angle, place at a pseudo-random consistent angle based on name hash
      let angle = 0;
      if (e.name) {
        let hash = 0;
        for (let i = 0; i < e.name.length; i++) hash = (hash * 31 + e.name.charCodeAt(i)) | 0;
        angle = (hash % 360) * Math.PI / 180;
      }
      // Rotate by player yaw so "forward" is up
      angle -= playerYaw;

      const ex = cx + Math.sin(angle) * dist;
      const ey = cy - Math.cos(angle) * dist;

      ctx.beginPath();
      ctx.arc(ex, ey, 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  };

  drawEntities(state.entities.hostiles, '#ff4444');
  drawEntities(state.entities.players, '#4488ff');
  drawEntities(state.entities.passives, '#44cc66');
}

// ─── Hotbar Display ─────────────────────────────────────

function updateHotbar(inventory) {
  const slots = els.hotbar.children;
  if (!inventory || !inventory.hotbar) {
    for (let i = 0; i < 9; i++) {
      slots[i].className = 'hotbar-slot empty';
      slots[i].textContent = '-';
    }
    return;
  }

  for (let i = 0; i < 9; i++) {
    const slot = inventory.hotbar[i];
    const el = slots[i];

    if (!slot || slot.category === 0) {
      el.className = 'hotbar-slot empty';
      el.textContent = '-';
    } else {
      const catName = CATEGORY_NAMES[slot.category] || '?';
      const catClass = CATEGORY_CLASSES[slot.category] || 'other';
      el.className = `hotbar-slot ${catClass}`;
      el.textContent = slot.count > 1 ? `${catName}\n${slot.count}` : catName;
    }

    // Selected slot highlight
    if (i === inventory.selected_slot) {
      el.classList.add('selected');
    }
  }
}

// ─── Agent Stats ────────────────────────────────────────

window.scar.onAgentStats((stats) => {
  if (!stats) return;
  els.statStep.textContent = stats.step.toLocaleString();
  els.statVitality.textContent = stats.vitality.toFixed(3);
  els.statSurprise.textContent = stats.avgSurprise.toFixed(3);
  els.statPatterns.textContent = `${stats.patterns} / ${stats.associations}`;
  const kd = stats.deaths > 0 ? (stats.kills / stats.deaths).toFixed(2) : stats.kills;
  els.statKd.textContent = `${stats.kills} / ${stats.deaths} (${kd})`;
  els.statUrgency.textContent = stats.urgency.toFixed(2);
});

// ─── Event Listeners (Push) ─────────────────────────────

window.scar.onControllerLog((line) => addLogLine(line, 'controller'));
window.scar.onPythonLog((line) => addLogLine(line, 'python'));

window.scar.onLauncherStatus((status) => {
  launcherStatus = status;
  updateUI();
});

window.scar.onPythonStatus((status) => {
  pythonStatus = status;
  updateUI();
});

// ─── Connection Presets ─────────────────────────────────

const PRESETS_KEY = 'scar-presets';

function loadPresets() {
  const presets = JSON.parse(localStorage.getItem(PRESETS_KEY) || '{}');
  els.presetSelect.innerHTML = '<option value="">-- Presets --</option>';
  for (const name of Object.keys(presets)) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    els.presetSelect.appendChild(opt);
  }
}

els.presetSave.addEventListener('click', () => {
  // window.prompt() doesn't work in Electron — use inline text input
  if (els.presetSelect.parentNode.querySelector('.preset-name-input')) return;

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'preset-name-input';
  input.placeholder = 'Type preset name, press Enter';
  els.presetSelect.style.display = 'none';
  els.presetSelect.parentNode.insertBefore(input, els.presetSelect);
  // Delay focus slightly for Electron reliability
  setTimeout(() => input.focus(), 50);

  const commit = () => {
    const name = input.value.trim();
    input.remove();
    els.presetSelect.style.display = '';
    if (!name) return;
    const presets = JSON.parse(localStorage.getItem(PRESETS_KEY) || '{}');
    presets[name] = {
      protocol: currentProtocol,
      connectType: currentConnectType,
      realmUrl: els.realmUrl.value,
      serverHost: els.serverHost.value,
      serverPort: els.serverPort.value,
      worldPath: els.worldSelect.value,
      listenPort: els.listenPort.value,
      bridgePort: els.bridgePort.value,
      stealth: els.optStealth.checked,
      keyboard: els.optKeyboard.checked,
      online: els.optOnline.checked,
      pvpStyle: els.pvpStyle.value,
      phase: els.phaseSelect.value,
      // Agent settings
      agentSteps: els.agentSteps.value,
      agentTransfer: els.agentTransfer.value,
      agentSave: els.agentSave.value,
      observeOnly: els.optObserve.checked,
      screenCapture: els.optScreenCapture.checked,
      factored: els.optFactored.checked,
      untilConverge: els.optConverge.checked,
    };
    localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
    loadPresets();
    els.presetSelect.value = name;
  };

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') commit();
    if (e.key === 'Escape') { input.value = ''; commit(); }
  });
  input.addEventListener('blur', commit);
});

els.presetDelete.addEventListener('click', () => {
  const name = els.presetSelect.value;
  if (!name) return;
  const presets = JSON.parse(localStorage.getItem(PRESETS_KEY) || '{}');
  delete presets[name];
  localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
  loadPresets();
});

els.presetSelect.addEventListener('change', () => {
  const name = els.presetSelect.value;
  if (!name) return;
  const presets = JSON.parse(localStorage.getItem(PRESETS_KEY) || '{}');
  const p = presets[name];
  if (!p) return;

  // Set protocol
  currentProtocol = p.protocol || 'bedrock';
  els.protocolGroup.querySelectorAll('.toggle-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.value === currentProtocol);
  });

  // Set connect type
  setConnectType(p.connectType || 'realm');

  // Fill fields
  els.realmUrl.value = p.realmUrl || '';
  els.serverHost.value = p.serverHost || '';
  els.serverPort.value = p.serverPort || '';
  if (p.worldPath) els.worldSelect.value = p.worldPath;
  els.listenPort.value = p.listenPort || '';
  els.bridgePort.value = p.bridgePort || '3001';
  els.optStealth.checked = p.stealth || false;
  els.optKeyboard.checked = p.keyboard || false;
  els.optOnline.checked = p.online || false;
  els.pvpStyle.value = p.pvpStyle || 'cooldown';
  els.phaseSelect.value = p.phase || '3';

  // Restore agent settings
  if (p.agentSteps) els.agentSteps.value = p.agentSteps;
  if (p.agentTransfer) els.agentTransfer.value = p.agentTransfer;
  if (p.agentSave) els.agentSave.value = p.agentSave;
  els.optObserve.checked = p.observeOnly || false;
  els.optScreenCapture.checked = p.screenCapture || false;
  els.optFactored.checked = p.factored || false;
  els.optConverge.checked = p.untilConverge || false;
});

// ─── Initialization ─────────────────────────────────────

(async function init() {
  updateConnectFields();
  updateUI();
  loadPresets();
  loadWorlds();
})();
