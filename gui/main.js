/**
 * SCAR-Minecraft — Electron main process.
 *
 * Manages the BrowserWindow, IPC handlers, ScarLauncher (in-process),
 * Python child process, state polling, window detection, and tray icon.
 */

const { app, BrowserWindow, ipcMain, dialog, Tray, Menu, Notification, nativeTheme } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const { ScarLauncher } = require('../controller/launcher');

// Resolve project root (one level up from gui/)
const PROJECT_ROOT = path.resolve(__dirname, '..');
const fs = require('fs');

/**
 * Find a usable Python 3.11+ executable.
 * Checks project venv, original dev directory, Homebrew, then system python3.
 */
function resolvePythonPath() {
  const os = require('os');
  const homeDir = os.homedir();

  const candidates = [
    // 1. Project venv (dev mode)
    path.join(PROJECT_ROOT, '.venv', 'bin', 'python'),
    path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe'),
    // 2. Original project location (when running from packaged .app)
    path.join(homeDir, 'Desktop', 'scar-minecraft', '.venv', 'bin', 'python'),
    // 3. Homebrew Python 3.11+ (typing.Self requires 3.11)
    '/opt/homebrew/bin/python3.13',
    '/opt/homebrew/bin/python3.12',
    '/opt/homebrew/bin/python3.11',
    '/usr/local/bin/python3.13',
    '/usr/local/bin/python3.12',
    '/usr/local/bin/python3.11',
  ];
  for (const p of candidates) {
    try { fs.accessSync(p, fs.constants.X_OK); return p; } catch (_) {}
  }
  return 'python3';
}

/**
 * Resolve the src/ directory containing the FPI Python modules.
 * When packaged, extraResources copies src/ into Contents/Resources/src/.
 */
function resolveSrcDir() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'src');
  }
  return path.join(PROJECT_ROOT, 'src');
}

let mainWindow = null;
let tray = null;
let launcher = null;
let pythonProcess = null;
let stateInterval = null;

// Circular log buffers
const MAX_LOG_LINES = 2000;
const controllerLogs = [];
const pythonLogs = [];

// Latest agent stats parsed from Python output
let agentStats = null;

// Track last alive state for death notifications
let lastAlive = true;

// ─── Window Creation ────────────────────────────────────────────────

function createWindow() {
  nativeTheme.themeSource = 'dark';

  mainWindow = new BrowserWindow({
    width: 1060,
    height: 860,
    minWidth: 800,
    minHeight: 600,
    backgroundColor: '#1a1a2e',
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  mainWindow.on('close', (e) => {
    // Minimize to tray instead of quitting
    if (tray && !app.isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });
}

function createTray() {
  // Use a simple template image for the tray (Electron will handle it)
  // On macOS, a 16x16 or 22x22 PNG works. We'll use a text-based fallback.
  try {
    tray = new Tray(path.join(__dirname, 'renderer', 'tray-icon.png'));
  } catch (_) {
    // No tray icon file — skip tray
    return;
  }

  const updateTrayMenu = () => {
    const botActive = launcher && launcher.adapter && launcher.adapter.botControlActive;
    const menu = Menu.buildFromTemplate([
      { label: 'Show SCAR', click: () => { mainWindow.show(); mainWindow.focus(); } },
      { type: 'separator' },
      {
        label: botActive ? 'Disable Bot Control' : 'Enable Bot Control',
        enabled: !!(launcher && launcher.adapter && launcher.adapter.ready),
        click: () => {
          if (launcher) launcher.setBotControl(!botActive);
          updateTrayMenu();
        },
      },
      { type: 'separator' },
      {
        label: 'Stop All',
        enabled: !!(launcher && launcher.status !== 'idle'),
        click: () => shutdownAll(),
      },
      {
        label: 'Quit',
        click: () => { app.isQuitting = true; app.quit(); },
      },
    ]);
    tray.setContextMenu(menu);

    const statusText = launcher ? launcher.status : 'idle';
    tray.setToolTip(`SCAR Minecraft — ${statusText}`);
  };

  updateTrayMenu();
  // Refresh tray menu periodically
  setInterval(updateTrayMenu, 2000);
}

// ─── State Polling ──────────────────────────────────────────────────

function startStatePolling() {
  if (stateInterval) return;
  stateInterval = setInterval(() => {
    if (!launcher || !mainWindow) return;
    const state = launcher.getState();
    if (state) {
      mainWindow.webContents.send('state:update', state);

      // Death notification
      if (lastAlive && !state.alive) {
        if (Notification.isSupported()) {
          new Notification({
            title: 'SCAR — Player Died',
            body: `Deaths: ${state.deaths || '?'}`,
          }).show();
        }
      }
      lastAlive = state.alive;
    }
  }, 200);
}

function stopStatePolling() {
  if (stateInterval) {
    clearInterval(stateInterval);
    stateInterval = null;
  }
}

// ─── Log Helpers ────────────────────────────────────────────────────

function pushLog(buffer, line) {
  buffer.push(line);
  if (buffer.length > MAX_LOG_LINES) buffer.shift();
}

// ─── Python Agent Stats Parsing ─────────────────────────────────────

function parseAgentLine(line) {
  // Match: [step   100] vitality=0.850  surprise=0.42  avg_surprise=0.350  patterns=47  assocs=120  valence=+12/-3  kills=2  deaths=1  urgency=0.15
  const m = line.match(
    /\[step\s+(\d+)\]\s+vitality=([\d.]+)\s+surprise=([\d.]+)\s+avg_surprise=([\d.]+)\s+patterns=(\d+)\s+assocs=(\d+)\s+valence=\+(\d+)\/-(\d+)\s+kills=(\d+)\s+deaths=(\d+)\s+urgency=([\d.]+)/
  );
  if (m) {
    agentStats = {
      step: parseInt(m[1]),
      vitality: parseFloat(m[2]),
      surprise: parseFloat(m[3]),
      avgSurprise: parseFloat(m[4]),
      patterns: parseInt(m[5]),
      associations: parseInt(m[6]),
      positiveValence: parseInt(m[7]),
      negativeValence: parseInt(m[8]),
      kills: parseInt(m[9]),
      deaths: parseInt(m[10]),
      urgency: parseFloat(m[11]),
    };
    if (mainWindow) mainWindow.webContents.send('agent:stats', agentStats);
  }
}

// ─── Shutdown ───────────────────────────────────────────────────────

async function shutdownAll() {
  stopStatePolling();

  if (pythonProcess) {
    try { pythonProcess.kill('SIGTERM'); } catch (_) {}
    pythonProcess = null;
  }

  if (launcher && launcher.status !== 'idle') {
    await launcher.shutdown();
    launcher = null;
  }

  if (mainWindow) {
    mainWindow.webContents.send('launcher:status', 'idle');
  }
}

// ─── IPC Handlers ───────────────────────────────────────────────────

function registerIPC() {

  // ── Launcher ──

  ipcMain.handle('launcher:start', async (_event, config) => {
    try {
      if (launcher && launcher.status !== 'idle') {
        return { ok: false, error: 'Launcher already running' };
      }

      launcher = new ScarLauncher({
        protocol: config.protocol || 'bedrock',
        worldPath: config.worldPath || undefined,
        realmInvite: config.realmInvite || undefined,
        serverHost: config.serverHost || undefined,
        serverPort: config.serverPort || undefined,
        listenPort: config.listenPort || undefined,
        bridgePort: config.bridgePort || 3001,
        stealth: config.stealth || false,
        keyboardFallback: config.keyboardFallback || false,
        mouseSensitivity: config.mouseSensitivity || 400,
        pvpStyle: config.pvpStyle || 'cooldown',
        onlineMode: config.onlineMode || false,
        version: config.version || undefined,
        phase: config.phase || 3,
      });

      launcher.on('log', ({ source, message }) => {
        const line = `[${source}] ${message}`;
        pushLog(controllerLogs, line);
        if (mainWindow) mainWindow.webContents.send('controller:log', line);
      });

      launcher.on('status-change', (status) => {
        if (mainWindow) mainWindow.webContents.send('launcher:status', status);
      });

      // Don't await — start() blocks until client connects.
      // We want the GUI to remain responsive.
      launcher.start().catch(err => {
        const line = `[main] Error: ${err.message}`;
        pushLog(controllerLogs, line);
        if (mainWindow) mainWindow.webContents.send('controller:log', line);
      });

      startStatePolling();
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  });

  ipcMain.handle('launcher:stop', async () => {
    await shutdownAll();
    return { ok: true };
  });

  ipcMain.handle('launcher:getStatus', () => {
    return launcher ? launcher.status : 'idle';
  });

  ipcMain.handle('launcher:getState', () => {
    if (!launcher) return null;
    return launcher.getState();
  });

  ipcMain.handle('launcher:setBotControl', (_event, enabled) => {
    if (!launcher || !launcher.adapter) return { botControlActive: false };
    launcher.setBotControl(enabled);
    return { botControlActive: launcher.adapter.botControlActive };
  });

  ipcMain.handle('launcher:getLogs', () => controllerLogs.slice());

  // ── Python Agent ──

  ipcMain.handle('python:start', (_event, config) => {
    if (pythonProcess) return { ok: false, error: 'Python already running' };

    const pythonPath = config.pythonPath || resolvePythonPath();
    const srcDir = resolveSrcDir();
    const args = ['-u', '-m', 'fpi.minecraft.runner'];

    if (config.host) args.push('--host', config.host);
    args.push('--port', String(config.port || 3001));
    args.push('--steps', String(config.steps || 10000));
    args.push('--phase', String(config.phase || 3));
    if (config.transfer) args.push('--transfer', config.transfer);
    if (config.save) args.push('--save', config.save);
    if (config.observeOnly) args.push('--observe-only');
    if (config.screenCapture) args.push('--screen-capture');
    if (config.captureRegion) args.push('--capture-region', config.captureRegion);
    if (config.visionWeights) args.push('--vision-weights', config.visionWeights);
    if (config.factored) args.push('--factored');
    if (config.untilConverge) args.push('--until-converge');

    // Build PYTHONPATH so system python3 can find the fpi package
    const pythonPathEnv = srcDir + (process.env.PYTHONPATH ? ':' + process.env.PYTHONPATH : '');

    try {
      pythonProcess = spawn(pythonPath, args, {
        cwd: app.isPackaged ? srcDir : PROJECT_ROOT,
        env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONPATH: pythonPathEnv },
      });
    } catch (err) {
      pythonProcess = null;
      return { ok: false, error: err.message };
    }

    pythonProcess.stdout.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (const line of lines) {
        if (!line.trim()) continue;
        pushLog(pythonLogs, line);
        if (mainWindow) mainWindow.webContents.send('python:log', line);
        parseAgentLine(line);
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (const line of lines) {
        if (!line.trim()) continue;
        pushLog(pythonLogs, `[stderr] ${line}`);
        if (mainWindow) mainWindow.webContents.send('python:log', `[stderr] ${line}`);
      }
    });

    pythonProcess.on('exit', (code) => {
      const msg = `Python exited with code ${code}`;
      pushLog(pythonLogs, msg);
      if (mainWindow) {
        mainWindow.webContents.send('python:log', msg);
        mainWindow.webContents.send('python:status', 'exited');
      }
      pythonProcess = null;
    });

    if (mainWindow) mainWindow.webContents.send('python:status', 'running');
    return { ok: true };
  });

  ipcMain.handle('python:stop', () => {
    if (!pythonProcess) return { ok: true };
    try { pythonProcess.kill('SIGTERM'); } catch (_) {}
    pythonProcess = null;
    if (mainWindow) mainWindow.webContents.send('python:status', 'idle');
    return { ok: true };
  });

  ipcMain.handle('python:getStatus', () => {
    return pythonProcess ? 'running' : 'idle';
  });

  ipcMain.handle('python:getLogs', () => pythonLogs.slice());

  // ── Utilities ──

  ipcMain.handle('worlds:list', () => {
    try {
      const { listWorlds } = require('../controller/auto-server');
      return listWorlds();
    } catch (_) {
      return [];
    }
  });

  ipcMain.handle('windows:detect', async () => {
    try {
      const { getWindows } = require('@nut-tree-fork/nut-js');
      if (!getWindows) return [];
      const windows = await getWindows();
      const results = [];
      for (const win of windows) {
        try {
          const title = await win.getTitle();
          if (title && title.startsWith('Minecraft') && !title.includes('SCAR')) {
            results.push({ title });
          }
        } catch (_) {}
      }
      return results;
    } catch (_) {
      return [];
    }
  });

  ipcMain.handle('app:getPythonPath', () => {
    return resolvePythonPath();
  });

  ipcMain.handle('app:selectFile', async (_event, opts) => {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile'],
      filters: opts?.filters || [{ name: 'All Files', extensions: ['*'] }],
      defaultPath: PROJECT_ROOT,
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return result.filePaths[0];
  });

  ipcMain.handle('app:selectSaveFile', async (_event, opts) => {
    const result = await dialog.showSaveDialog(mainWindow, {
      filters: opts?.filters || [{ name: 'All Files', extensions: ['*'] }],
      defaultPath: PROJECT_ROOT,
    });
    if (result.canceled || !result.filePath) return null;
    return result.filePath;
  });

  ipcMain.handle('agent:getStats', () => agentStats);
}

// ─── App Lifecycle ──────────────────────────────────────────────────

app.whenReady().then(() => {
  registerIPC();
  createWindow();
  createTray();
});

app.on('activate', () => {
  if (mainWindow) {
    mainWindow.show();
    mainWindow.focus();
  }
});

app.on('before-quit', async () => {
  app.isQuitting = true;
  await shutdownAll();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
