/**
 * ScarLauncher — reusable startup engine for the SCAR relay + bridge.
 *
 * Extracts the core orchestration from main.js so both the CLI and the
 * Electron GUI can start/stop the controller programmatically.
 *
 * Usage:
 *   const { ScarLauncher } = require('./launcher');
 *   const launcher = new ScarLauncher({ protocol: 'bedrock', realmInvite: '...' });
 *   launcher.on('log', ({ source, message }) => console.log(`[${source}] ${message}`));
 *   await launcher.start();
 *   // ... later ...
 *   await launcher.shutdown();
 */

const EventEmitter = require('events');
const fs = require('fs');
const path = require('path');
const { createBridge } = require('./bridge');
const { getState } = require('./state');
const { StealthEngine } = require('./stealth');

class ScarLauncher extends EventEmitter {
  /**
   * @param {object} opts
   * @param {string}  opts.protocol         - 'bedrock' | 'java'
   * @param {string}  [opts.worldPath]      - World save name or path (implies java, triggers auto-server)
   * @param {string}  [opts.realmInvite]    - Bedrock Realm invite URL
   * @param {string}  [opts.serverHost]     - Direct server hostname
   * @param {number}  [opts.serverPort]     - Direct server port
   * @param {number}  [opts.listenPort]     - Relay listen port (default: protocol-dependent)
   * @param {number}  [opts.bridgePort]     - Bridge TCP port (default: 3001)
   * @param {boolean} [opts.stealth]        - Enable stealth mode
   * @param {boolean} [opts.keyboardFallback] - Enable keyboard fallback (auto for Java)
   * @param {number}  [opts.mouseSensitivity] - Mouse sensitivity (default: 400)
   * @param {string}  [opts.pvpStyle]       - 'cooldown' | 'spam'
   * @param {boolean} [opts.onlineMode]     - Microsoft auth for Java servers
   * @param {string}  [opts.version]        - Specific MC version
   * @param {number}  [opts.phase]          - Action space phase (1-4)
   */
  constructor(opts = {}) {
    super();
    this._opts = opts;
    this._status = 'idle';
    this._adapter = null;
    this._bridgeServer = null;
    this._autoServer = null;
    this._keyboardFallback = null;
    this._nativeMouse = null;
    this._cooldownInterval = null;
    this._trackingState = null;
    this._actionConfig = null;
  }

  /** @returns {'idle'|'starting'|'waiting_client'|'running'|'error'} */
  get status() { return this._status; }

  /** @returns {object|null} The relay adapter */
  get adapter() { return this._adapter; }

  /** @returns {object|null} Tracking state */
  get trackingState() { return this._trackingState; }

  /** @returns {object|null} Action config */
  get actionConfig() { return this._actionConfig; }

  _setStatus(s) {
    this._status = s;
    this.emit('status-change', s);
  }

  _log(source, message) {
    this.emit('log', { source, message });
  }

  /**
   * Start the relay + bridge.
   * Resolves when the relay is listening (client may not be connected yet).
   */
  async start() {
    if (this._status !== 'idle') {
      throw new Error(`Cannot start: status is '${this._status}'`);
    }
    this._setStatus('starting');

    const opts = this._opts;
    const isJava = opts.protocol === 'java';
    const defaultPort = isJava ? 25565 : 19132;

    // --- World save: attach mode or auto-server ---
    if (opts.worldPath) {
      const { resolveWorldPath } = require('./auto-server');
      try {
        opts.worldPath = resolveWorldPath(opts.worldPath);
      } catch (e) {
        this._log('main', `World not found: ${e.message}`);
        throw e;
      }
      const lockFile = path.join(opts.worldPath, 'session.lock');
      if (fs.existsSync(lockFile)) {
        // World is open in Minecraft — use attach mode
        this._log('main', 'World is open in Minecraft. Attaching...');
        await this._startAttachMode(opts);
        return;
      }
      // World is NOT open — start dedicated server
      const { startAutoServer } = require('./auto-server');
      this._log('main', `Starting dedicated server from world: ${opts.worldPath}`);
      this._autoServer = await startAutoServer(opts.worldPath);
      this._log('main', `Server ready on port ${this._autoServer.port} (${this._autoServer.version})`);
    }

    // --- Build internal config ---
    const config = {
      listenPort: opts.listenPort || defaultPort,
      serverHost: this._autoServer ? this._autoServer.host : (opts.serverHost || 'localhost'),
      serverPort: this._autoServer ? this._autoServer.port : (opts.serverPort || defaultPort),
      authCache: './auth_cache',
      logPackets: false,
      realmInvite: opts.realmInvite || null,
      realmId: null,
      onlineMode: opts.onlineMode || false,
      version: this._autoServer ? this._autoServer.version : (opts.version || false),
      bridgePort: opts.bridgePort || 3001,
    };

    this._log('main', `Protocol: ${(opts.protocol || 'bedrock').toUpperCase()}`);

    // --- Stealth ---
    const stealth = opts.stealth ? new StealthEngine() : null;
    if (stealth) this._log('main', 'Stealth mode enabled.');

    // --- Keyboard fallback ---
    const needKeyboard = isJava || opts.keyboardFallback;
    if (needKeyboard) {
      const { KeyboardFallback } = require('./keyboard-fallback');
      this._keyboardFallback = new KeyboardFallback({
        mouseSensitivity: opts.mouseSensitivity || 400,
        protocol: opts.protocol || 'bedrock',
      });
      this._log('main', `Keyboard input enabled (${isJava ? 'required for Java' : 'opt-in'}).`);
    }

    // --- Action config ---
    this._actionConfig = {
      tickRate: 20,
      actionDurationTicks: 4,
      stealth,
      protocol: opts.protocol || 'bedrock',
      pvpStyle: opts.pvpStyle || 'cooldown',
    };

    // --- Tracking state ---
    this._trackingState = {
      pendingRespawn: false,
      lastAttackLanded: false,
      lastPlayerHitLanded: false,
      projectileHitLanded: false,
      projectilePlayerHitLanded: false,
      killsSinceLastState: 0,
      attackedEntities: new Set(),
      attackCooldown: 0,
      knockbackCooldown: 0,
      lastMacroStatus: null,
      // Self-awareness / threat dynamics tracking
      _prevPosition: null,
      _prevPositionTime: 0,
      _prevHealth: 20,
      _prevFood: 20,
      _ticksAirborne: 0,
      _recentHits: [],       // timestamps of hits landed (last 10s)
      _recentDamage: [],     // {time, amount} of damage taken (last 5s)
      _recentKills: [],      // timestamps of kills (last 30s)
      _prevEntitySpeeds: new Map(),
    };

    // Cooldown tick
    this._cooldownInterval = setInterval(() => {
      if (this._trackingState.knockbackCooldown > 0) this._trackingState.knockbackCooldown--;
      if (this._trackingState.attackCooldown > 0) this._trackingState.attackCooldown--;
    }, 50);

    // --- Create adapter ---
    if (isJava) {
      const { JavaRelayAdapter } = require('./java-relay');
      this._adapter = new JavaRelayAdapter(config);
    } else {
      const { RelayAdapter } = require('./relay');
      this._adapter = new RelayAdapter(config);
    }

    if (this._keyboardFallback) {
      this._adapter.setKeyboardFallback(this._keyboardFallback);
    }

    // Native mouse delta for Java
    if (isJava) {
      try {
        const { NativeMouseDelta } = require('./mouse-delta');
        this._nativeMouse = new NativeMouseDelta({
          mouseSensitivity: opts.mouseSensitivity || 400,
        });
        this._nativeMouse.start();
        this._adapter.setNativeMouseDelta(this._nativeMouse);
      } catch (e) {
        this._log('main', `Native mouse delta not available: ${e.message}`);
      }
    }

    // --- Bridge ---
    this._bridgeServer = createBridge(
      config.bridgePort, this._adapter, this._trackingState, this._actionConfig,
    );
    this._log('main', `Bridge listening on port ${config.bridgePort}`);

    // --- Start relay ---
    this._log('main', 'Starting relay proxy...');
    this._setStatus('waiting_client');
    this._log('main', `Waiting for Minecraft client on port ${config.listenPort}...`);

    try {
      await this._adapter.start(this._trackingState);
      this._setStatus('running');
      this._log('main', 'Client connected. System ready.');
      this._log('main', 'Bot control is OFF — user plays normally.');
    } catch (err) {
      this._setStatus('error');
      this._log('main', `Failed to start relay: ${err.message}`);
      throw err;
    }
  }

  /**
   * Attach to an already-running Minecraft singleplayer world.
   * Automates Open-to-LAN, discovers the LAN port, connects a headless
   * observer client, and controls the real client via keyboard/mouse.
   */
  async _startAttachMode(opts) {
    const { KeyboardFallback } = require('./keyboard-fallback');
    const { NativeMouseDelta } = require('./mouse-delta');
    const { openToLanWithRetry } = require('./lan-automation');
    const { discoverLanGame } = require('./lan-discovery');
    const { AttachAdapter } = require('./attach-adapter');
    const { StealthEngine } = require('./stealth');

    const stealth = opts.stealth ? new StealthEngine() : null;
    if (stealth) this._log('main', 'Stealth mode enabled.');

    // --- Tracking state ---
    this._trackingState = {
      pendingRespawn: false,
      lastAttackLanded: false,
      lastPlayerHitLanded: false,
      projectileHitLanded: false,
      projectilePlayerHitLanded: false,
      killsSinceLastState: 0,
      attackedEntities: new Set(),
      attackCooldown: 0,
      knockbackCooldown: 0,
      lastMacroStatus: null,
      // Self-awareness / threat dynamics tracking
      _prevPosition: null,
      _prevPositionTime: 0,
      _prevHealth: 20,
      _prevFood: 20,
      _ticksAirborne: 0,
      _recentHits: [],       // timestamps of hits landed (last 10s)
      _recentDamage: [],     // {time, amount} of damage taken (last 5s)
      _recentKills: [],      // timestamps of kills (last 30s)
      _prevEntitySpeeds: new Map(),
    };

    this._cooldownInterval = setInterval(() => {
      if (this._trackingState.knockbackCooldown > 0) this._trackingState.knockbackCooldown--;
      if (this._trackingState.attackCooldown > 0) this._trackingState.attackCooldown--;
    }, 50);

    // --- Action config ---
    this._actionConfig = {
      tickRate: 20,
      actionDurationTicks: 4,
      stealth,
      protocol: 'java',
      pvpStyle: opts.pvpStyle || 'cooldown',
    };

    // --- Create adapter + bridge EARLY so Python can connect and wait ---
    const bridgePort = opts.bridgePort || 3001;

    // Read actual version from level.dat (ping reports wrong version for 26.1+)
    let worldVersion;
    if (opts.worldPath) {
      try {
        const { readWorldVersion } = require('./auto-server');
        worldVersion = await readWorldVersion(opts.worldPath);
        this._log('main', `World version from level.dat: ${worldVersion}`);
      } catch (e) {
        this._log('main', `Could not read world version: ${e.message}`);
      }
    }

    // Create adapter with placeholder port (will be updated after LAN discovery)
    this._adapter = new AttachAdapter({
      lanHost: 'localhost',
      lanPort: 0,
      targetUsername: 'Player',
      mouseSensitivity: opts.mouseSensitivity || 400,
      version: worldVersion,
    });

    this._bridgeServer = createBridge(
      bridgePort, this._adapter, this._trackingState, this._actionConfig,
    );
    this._log('main', `Bridge listening on port ${bridgePort}`);

    // --- Discover LAN (this is the blocking step) ---
    this._log('main', 'Discovering LAN game...');
    this._setStatus('waiting_client');

    try {
      const lanResult = await openToLanWithRetry(discoverLanGame, {
        log: (msg) => this._log('lan', msg),
      });
      this._log('main', `LAN server discovered on port ${lanResult.port}`);

      // --- Port conflict detection ---
      if (lanResult.port === bridgePort) {
        this._log('main', `WARNING: LAN port ${lanResult.port} conflicts with bridge port. Restarting bridge on port ${bridgePort + 1}.`);
        this._log('main', 'TIP: Don\'t set a custom LAN port in Minecraft — let it auto-pick a random port.');
        try { this._bridgeServer.close(); } catch (_) {}
        this._bridgeServer = createBridge(
          bridgePort + 1, this._adapter, this._trackingState, this._actionConfig,
        );
        this._log('main', `Bridge restarted on port ${bridgePort + 1}`);
      }

      // --- Update adapter config with real LAN port ---
      this._adapter.config.lanHost = lanResult.host || 'localhost';
      this._adapter.config.lanPort = lanResult.port;

      // --- Read username from level.dat or default ---
      try {
        const levelDat = path.join(opts.worldPath, 'level.dat');
        if (fs.existsSync(levelDat)) {
          this._log('main', 'Will track first player entity found on LAN server.');
        }
      } catch (_) {}

      // --- Connect headless observer ---
      this._log('main', 'Connecting headless observer to LAN server...');
      await this._adapter.start(this._trackingState, {
        log: (msg) => this._log('attach', msg),
      });

      this._setStatus('running');
      this._log('main', 'Attached to Minecraft. System ready.');
      this._log('main', 'Bot control is OFF — user plays normally.');
    } catch (err) {
      this._setStatus('error');
      this._log('main', `Failed to attach: ${err.message || err.toString?.() || JSON.stringify(err)}`);
      throw err;
    }
  }

  /**
   * Get current game state dict (same as bridge get_state response).
   * Returns null if adapter not ready.
   */
  getState() {
    if (!this._adapter || !this._adapter.ready) return null;
    return getState(this._adapter, this._trackingState);
  }

  /**
   * Toggle bot control.
   */
  setBotControl(enabled) {
    if (!this._adapter) return;
    if (enabled) {
      this._adapter.enableBotControl();
    } else {
      this._adapter.disableBotControl();
    }
  }

  /**
   * Graceful shutdown — stops relay, bridge, auto-server, intervals.
   */
  async shutdown() {
    this._log('main', 'Shutting down...');

    if (this._cooldownInterval) {
      clearInterval(this._cooldownInterval);
      this._cooldownInterval = null;
    }

    if (this._adapter) {
      try { this._adapter.disconnect(); } catch (_) {}
      this._adapter = null;
    }

    if (this._bridgeServer) {
      try { this._bridgeServer.close(); } catch (_) {}
      this._bridgeServer = null;
    }

    if (this._nativeMouse) {
      try { this._nativeMouse.stop(); } catch (_) {}
      this._nativeMouse = null;
    }

    if (this._autoServer) {
      try { this._autoServer.stop(); } catch (_) {}
      this._autoServer = null;
    }

    this._trackingState = null;
    this._actionConfig = null;
    this._keyboardFallback = null;
    this._setStatus('idle');
    this._log('main', 'Shutdown complete.');
  }
}

module.exports = { ScarLauncher };
