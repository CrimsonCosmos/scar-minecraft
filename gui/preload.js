/**
 * SCAR-Minecraft — Electron preload script.
 *
 * Exposes a safe `window.scar` API via contextBridge for the renderer.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('scar', {
  // ── Launcher ──
  startLauncher: (config) => ipcRenderer.invoke('launcher:start', config),
  stopLauncher: () => ipcRenderer.invoke('launcher:stop'),
  getLauncherStatus: () => ipcRenderer.invoke('launcher:getStatus'),
  getGameState: () => ipcRenderer.invoke('launcher:getState'),
  setBotControl: (enabled) => ipcRenderer.invoke('launcher:setBotControl', enabled),
  getLauncherLogs: () => ipcRenderer.invoke('launcher:getLogs'),

  // ── Python Agent ──
  startPython: (config) => ipcRenderer.invoke('python:start', config),
  stopPython: () => ipcRenderer.invoke('python:stop'),
  getPythonStatus: () => ipcRenderer.invoke('python:getStatus'),
  getPythonLogs: () => ipcRenderer.invoke('python:getLogs'),

  // ── Utilities ──
  listWorlds: () => ipcRenderer.invoke('worlds:list'),
  detectMinecraftWindows: () => ipcRenderer.invoke('windows:detect'),
  getPythonPath: () => ipcRenderer.invoke('app:getPythonPath'),
  selectFile: (opts) => ipcRenderer.invoke('app:selectFile', opts),
  selectSaveFile: (opts) => ipcRenderer.invoke('app:selectSaveFile', opts),
  getAgentStats: () => ipcRenderer.invoke('agent:getStats'),

  // ── Push Events (main → renderer) ──
  onStateUpdate: (cb) => ipcRenderer.on('state:update', (_e, d) => cb(d)),
  onControllerLog: (cb) => ipcRenderer.on('controller:log', (_e, d) => cb(d)),
  onPythonLog: (cb) => ipcRenderer.on('python:log', (_e, d) => cb(d)),
  onAgentStats: (cb) => ipcRenderer.on('agent:stats', (_e, d) => cb(d)),
  onLauncherStatus: (cb) => ipcRenderer.on('launcher:status', (_e, d) => cb(d)),
  onPythonStatus: (cb) => ipcRenderer.on('python:status', (_e, d) => cb(d)),
});
