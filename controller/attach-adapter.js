/**
 * AttachAdapter — attach to an already-running Minecraft Java client via LAN.
 *
 * Architecture:
 *   Minecraft Client (user's, with singleplayer world open to LAN)
 *        | keyboard/mouse (nut-js via KeyboardFallback)
 *   AttachAdapter
 *        | headless mc.createClient reads game state
 *   LAN Server (integrated server exposed by Open to LAN)
 *
 * The user opens their singleplayer world to LAN. A headless observer client
 * connects to read game state (position, health, entities, blocks). All actions
 * are performed via OS-level keyboard/mouse simulation to the real client window.
 *
 * Implements the same adapter interface as JavaRelayAdapter so bridge.js,
 * state.js, and actions.js work unchanged.
 */

const mc = require('minecraft-protocol');
const { ENTITY_META } = require('./categories');

/** Normalize angle to [-180, 180] range (degrees). */
function normalizeAngle(a) {
  a = a % 360;
  if (a > 180) a -= 360;
  if (a < -180) a += 360;
  return a;
}

class AttachAdapter {
  constructor(config) {
    this.config = config;
    this._client = null;           // headless mc.Client
    this._ready = false;
    this._resolveStart = null;
    this._targetUsername = config.targetUsername;
    this._targetEntityId = null;   // entity ID of the original player

    // --- State tracked from clientbound packets ---
    this._position = { x: 0, y: 64, z: 0 };
    this._health = 20;
    this._food = 20;               // hardcoded (not visible to other clients)
    this._foodSaturation = 5;      // hardcoded
    this._yaw = 0;
    this._pitch = 0;
    this._onGround = true;
    this._timeOfDay = 0;
    this._xpLevel = 0;
    this._quickBarSlot = 0;
    this._entityId = null;         // OUR entity ID (the headless observer)

    // Entity registry
    this._entities = new Map();

    // Runtime entity type ID → name (from server's registry_data, overrides mcData)
    this._entityTypeRegistry = null;

    // Player info (UUID -> username)
    this._playerInfo = new Map();

    // Inventory (empty — we can't see the original player's inventory)
    this._inventory = [];

    // Version info + minecraft-data
    this._version = null;
    this._mcData = null;

    // Block cache (created after mcData loads)
    this._blockCache = null;

    // --- Bot control state ---
    this._botControlActive = false;

    // --- Auto-yield state ---
    this._inputMonitor = null;     // InputMonitor instance
    this._userActive = false;      // true when user is providing input
    this._mcFocused = true;        // true when Minecraft is the focused window

    // Keyboard fallback (primary input method for attach mode)
    this._keyboardFallback = null;

    // Native mouse delta (macOS CGEvent helper)
    this._nativeMouseDelta = null;

    // Simulated control states
    this._controlStates = {
      forward: false, back: false, left: false, right: false,
      jump: false, sprint: false, sneak: false,
    };

    // Active status effects (effect ID → {amplifier, duration, startTime})
    this._activeEffects = new Map();

    // Weather state
    this._isRaining = false;
    this._isThundering = false;

    // Biome
    this._biome = 'plains';

    // Track whether we've sent /gamemode spectator
    this._spectatorSent = false;
    this._awaitingSpectatorConfirm = false;
    this._hasPermissions = false;
    this._spectatorFailed = false;

    // Follow loop interval
    this._followInterval = null;

    // Tick synchronization interval
    this._tickEndInterval = null;

    // Auto-respawn
    this._autoRespawn = true;      // auto-click Respawn on death
    this._respawnPending = false;   // debounce flag

    // Refocus cooldown — prevent rapid refocus cycles that reopen the pause menu
    this._lastRefocusTime = 0;
  }

  /**
   * Start the attach adapter.
   * Creates keyboard fallback, connects headless observer client.
   */
  async start(trackingState, opts = {}) {
    const log = opts.log || console.log.bind(console);

    // 1. Create keyboard fallback
    const { KeyboardFallback } = require('./keyboard-fallback');
    this._keyboardFallback = new KeyboardFallback({
      mouseSensitivity: this.config.mouseSensitivity || 400,
      protocol: 'java',
    });

    // 2. Try to create NativeMouseDelta
    try {
      const { NativeMouseDelta } = require('./mouse-delta');
      this._nativeMouseDelta = new NativeMouseDelta({
        mouseSensitivity: this.config.mouseSensitivity || 400,
      });
      this._nativeMouseDelta.start();
    } catch (_) {
      // Native mouse not available -- keyboard mouse fallback is fine
    }

    // 3. Start input monitor for auto-yield + focus detection
    try {
      const { InputMonitor } = require('./input-monitor');
      this._inputMonitor = new InputMonitor();
      this._inputMonitor.on('user-input', () => {
        if (!this._userActive) {
          this._userActive = true;
          console.log('[attach] User input detected — bot yielding.');
        }
      });
      this._inputMonitor.on('user-idle', () => {
        if (this._userActive) {
          this._userActive = false;
          console.log('[attach] User idle — bot resuming.');
        }
      });
      this._inputMonitor.on('focus-lost', () => {
        this._mcFocused = false;
        // Release all held keys immediately to prevent stuck keys
        if (this._keyboardFallback) {
          this._keyboardFallback.clearControlStates().catch(() => {});
        }
        console.log('[attach] Minecraft lost focus — bot paused.');
      });
      this._inputMonitor.on('focus-gained', () => {
        this._mcFocused = true;
        console.log('[attach] Minecraft regained focus.');
      });
      this._inputMonitor.start();
    } catch (e) {
      console.warn('[attach] Input monitor not available:', e.message);
    }

    // 4. Preemptively patch ALL unsupported versions before any minecraft-protocol call.
    const { patchAllUnsupported, patchVersionSupport } = require('./version-compat');
    const patchedVersions = patchAllUnsupported();
    if (patchedVersions.length > 0) {
      log(`Patched version support: ${patchedVersions.join(', ')}`);
    }

    // 5. Detect version: always ping for the real protocol number, then combine
    // with level.dat version if available. MC 26.1+ LAN servers report "1.21"
    // (protocol 767) via ping but speak 26.1 packet format. We need level.dat
    // for the correct packet definitions and the ping for the handshake protocol.
    let serverVersion = this.config.version || false;
    let pingProtocol = null;
    try {
      log(`Pinging LAN server at ${this.config.lanHost || 'localhost'}:${this.config.lanPort}...`);
      const pingResult = await new Promise((res, rej) => {
        const timer = setTimeout(() => rej(new Error('ping timeout (5s)')), 5000);
        mc.ping({
          host: this.config.lanHost || 'localhost',
          port: this.config.lanPort,
        }, (err, result) => { clearTimeout(timer); if (err) rej(err); else res(result); });
      });
      if (pingResult && pingResult.version) {
        const detectedName = pingResult.version.name;
        pingProtocol = pingResult.version.protocol;
        log(`Server version: ${detectedName} (protocol ${pingProtocol})`);
        if (!serverVersion) {
          serverVersion = detectedName;
        }
      }
    } catch (e) {
      log(`Could not ping server: ${e.message || e.toString?.() || JSON.stringify(e)}`);
      if (e.stack) log(`Ping stack: ${e.stack.split('\n').slice(0, 3).join(' | ')}`);
    }

    // Patch version support using level.dat version name + ping protocol number.
    // This ensures the handshake uses the server's real protocol (767) while
    // the packet definitions come from the correct version (26.1).
    if (serverVersion) {
      serverVersion = patchVersionSupport(serverVersion, pingProtocol) || serverVersion;
    }

    // MC 26.1+ LAN servers report protocol 767 (1.21) in the handshake but
    // use 26.1 packet format. If the ping protocol differs from the patched
    // version data, override the data entry so the handshake sends the correct
    // protocol number while keeping the 26.1 packet definitions.
    if (serverVersion && pingProtocol) {
      const mcDataCheck = require('minecraft-data');
      try {
        const vd = mcDataCheck(serverVersion);
        if (vd && vd.version.version !== pingProtocol) {
          log(`Protocol override: data says ${vd.version.version}, server says ${pingProtocol}. Using server's protocol for handshake.`);
          const dataModule = require('minecraft-data/data');
          const entry = dataModule.pc[vd.version.majorVersion];
          if (entry) {
            const origVer = vd.version;
            Object.defineProperty(entry, 'version', {
              get() {
                return { ...origVer, version: pingProtocol };
              },
              enumerable: true,
              configurable: true,
            });
          }
        }
      } catch (_) {}
    }

    log(`Connecting with version=${serverVersion || 'auto-detect'}...`);

    // 6. Connect headless client (with 30s timeout)
    return new Promise((resolve, reject) => {
      this._resolveStart = resolve;
      let settled = false;

      const connectionTimeout = setTimeout(() => {
        if (!settled) {
          settled = true;
          const err = new Error('Connection timed out after 30s — no login packet received');
          log(err.message);
          if (this._client) {
            try { this._client.end(); } catch (_) {}
          }
          reject(err);
        }
      }, 30000);

      try {
        this._client = mc.createClient({
          host: this.config.lanHost || 'localhost',
          port: this.config.lanPort,
          username: 'SCAR_Observer',
          auth: 'offline',
          version: serverVersion,
        });
      } catch (e) {
        clearTimeout(connectionTimeout);
        settled = true;
        log(`mc.createClient() threw: ${e.message}`);
        reject(e);
        return;
      }

      log('Client created, waiting for login...');

      // Disable bundle packet buffering — we don't need atomic tick processing,
      // and bundles can trap packets if the closing delimiter is lost in the
      // config→play state transition.
      Object.defineProperty(this._client, '_hasBundlePacket', {
        get() { return false; },
        set() {},
        configurable: true,
      });

      // Wrap write() to catch serialization errors from auto-responses.
      // Some packet formats changed between 1.21.11 and 26.1+ even though
      // the IDs are now correct via patchPlayPacketIds().
      let writeCount = 0;
      const origWrite = this._client.write.bind(this._client);
      this._client.write = (name, data) => {
        writeCount++;
        if (writeCount <= 15) log(`Write #${writeCount}: ${name}`);
        try {
          origWrite(name, data);
        } catch (e) {
          if (!this._writeErrors) this._writeErrors = new Set();
          if (!this._writeErrors.has(name)) {
            this._writeErrors.add(name);
            log(`Write FAILED (${name}): ${e.message}`);
          }
        }
      };

      // Log state transitions and handle play-state entry.
      // The 'login' play-state packet is often lost during the config→play
      // deserializer swap in minecraft-protocol (stream buffer race condition).
      // We don't depend on it — resolve the connection as soon as play begins.
      let playPacketCount = 0;
      this._client.on('state', (newState) => {
        log(`Protocol state → ${newState}`);
        if (newState === 'play') {
          // Initialize immediately — don't wait for login packet
          try {
            this._mcData = require('minecraft-data')(this._client.version);
          } catch (_) { this._mcData = null; }
          if (this._mcData) {
            try {
              const { JavaBlockCache } = require('./java-block-cache');
              this._blockCache = new JavaBlockCache(this._mcData);
            } catch (_) {}
          }
          this._ready = true;

          clearTimeout(connectionTimeout);
          if (!settled) {
            settled = true;
            log('Connected (play state entered).');
            if (this._resolveStart) {
              this._resolveStart();
              this._resolveStart = null;
            }
          }

          // Hook deserializer to log raw packet IDs before parsing.
          // This reveals packets that silently fail to parse.
          try {
            const deser = this._client.deserializer;
            if (deser) {
              const origTransform = deser._transform.bind(deser);
              let rawCount = 0;
              deser._transform = function(chunk, enc, cb) {
                rawCount++;
                // Read varint packet ID from raw buffer
                let packetId = 0;
                if (chunk && chunk.length > 0) {
                  let val = 0, i = 0, byte;
                  do {
                    byte = chunk[i];
                    val |= (byte & 0x7F) << (7 * i);
                    i++;
                  } while (byte & 0x80 && i < 5);
                  packetId = val;
                }
                if (rawCount <= 40) {
                  log(`Raw #${rawCount}: id=0x${packetId.toString(16).padStart(2,'0')} (${chunk.length} bytes)`);
                }
                origTransform(chunk, enc, cb);
              };
              log('Hooked deserializer for raw packet logging.');
            } else {
              log('WARNING: deserializer is null — cannot hook raw packets.');
            }
          } catch (e) {
            log(`Could not hook deserializer: ${e.message}`);
          }

          // Send required packets immediately — server will timeout without these
          try { this._client.write('player_loaded', {}); } catch (_) {}
          try {
            this._client.write('custom_payload', {
              channel: 'minecraft:brand',
              data: Buffer.from('\x07vanilla'),
            });
          } catch (_) {}
          // Send client settings — server needs view distance, locale, etc.
          // Without this, server won't send chunks or may timeout.
          try {
            this._client.write('settings', {
              locale: 'en_us',
              viewDistance: 10,
              chatFlags: 0,        // enabled
              chatColors: true,
              skinParts: 0x7f,     // all parts visible
              mainHand: 1,         // right
              enableTextFiltering: false,
              enableServerListing: false,
              particleStatus: 0,   // all
            });
          } catch (_) {}
          // Start tick_end interval — MC 26.1+ tick synchronization.
          // Server kicks clients that don't send periodic tick_end acknowledgments.
          this._tickEndInterval = setInterval(() => {
            try { this._client.write('tick_end', {}); } catch (_) {}
          }, 50); // 20 TPS = every 50ms
        }
      });

      this._client.on('packet', (data, meta) => {
        // Capture entity type registry from config state
        if (meta.state === 'configuration' && meta.name === 'registry_data') {
          this._captureEntityTypeRegistry(data);
        }
        if (meta.state !== 'play') {
          if (meta.name === 'disconnect') {
            const reason = data.reason || data.message || JSON.stringify(data);
            log(`Server disconnect during ${meta.state}: ${reason}`);
          }
          return;
        }

        // Diagnostic: log first play-state packets to help debug connection issues
        if (playPacketCount < 30) {
          playPacketCount++;
          log(`Play packet #${playPacketCount}: ${meta.name}`);
        }

        // Capture entityId from login packet if it does arrive
        if (meta.name === 'login' && data.entityId !== undefined) {
          this._entityId = data.entityId;
          log(`Login packet received — entity ID: ${this._entityId}`);
        }

        if (meta.name === 'kick_disconnect' || meta.name === 'disconnect') {
          let reason = data.reason || data.message || data;
          if (typeof reason === 'object') reason = reason.text || reason.translate || JSON.stringify(reason);
          log(`Server kicked us: ${reason}`);
        }

        try {
          this._handlePacket(meta.name, data, trackingState);
        } catch (e) {
          if (!this._warnedPackets) this._warnedPackets = new Set();
          if (!this._warnedPackets.has(meta.name)) {
            this._warnedPackets.add(meta.name);
            log(`Packet parse warning (${meta.name}): ${e.message}`);
          }
        }
      });

      // On login packet (may or may not fire — login packet often lost in
      // config→play deserializer swap). If it does fire, capture entityId.
      this._client.on('login', (data) => {
        this._entityId = data.entityId;
        log(`Login event fired — entity ID: ${this._entityId}`);
        // Detect initial gamemode — if LAN was opened with "Game Mode: Spectator",
        // the observer joins directly in spectator mode (no command needed).
        const gm = data.worldState?.gamemode;
        if (gm === 'spectator' || gm === 3) {
          this._hasPermissions = true;
          this._spectatorSent = true;
          this._awaitingSpectatorConfirm = false;
          log('Observer joined in spectator mode (LAN default gamemode).');
          this._startFollowLoop();
        }
      });

      // Error/end handlers
      this._client.on('error', (err) => {
        const msg = err.message || err.toString?.() || JSON.stringify(err);
        // Catch all parse/deserialization error variants from minecraft-protocol
        if (msg.includes('PartialReadError') || msg.includes('Read error') ||
            msg.includes('deserialization') || msg.includes('Parse error') ||
            msg.includes('DeserializationError') || msg.includes('not a valid value')) {
          if (!this._parseErrorCount) this._parseErrorCount = 0;
          this._parseErrorCount++;
          if (this._parseErrorCount <= 10) {
            log(`Parse error #${this._parseErrorCount}: ${msg.substring(0, 150)}`);
          }
          return;
        }
        log(`Client error: ${msg}`);
        if (err.stack) log(`Stack: ${err.stack.split('\n').slice(0, 3).join(' | ')}`);
        if (!settled) {
          settled = true;
          clearTimeout(connectionTimeout);
          reject(err);
        }
      });

      this._client.on('end', (reason) => {
        log(`Client disconnected. ${reason || ''}`);
        this._ready = false;
        this._stopFollowLoop();
        if (this._tickEndInterval) {
          clearInterval(this._tickEndInterval);
          this._tickEndInterval = null;
        }
        if (!settled) {
          settled = true;
          clearTimeout(connectionTimeout);
          reject(new Error('Connection ended before login: ' + (reason || 'unknown reason')));
        }
      });
    });
  }

  /**
   * Process packets from the LAN server.
   * Tracks the original player's state via entity packets.
   */
  _handlePacket(name, data, trackingState) {
    switch (name) {
      case 'named_entity_spawn':
        this._handleNamedEntitySpawn(data);
        break;

      case 'spawn_entity':
      case 'spawn_entity_living':
        this._handleSpawnEntity(data);
        break;

      case 'sync_entity_position':
      case 'entity_teleport': {
        // sync_entity_position (26.1+): authoritative entity position with velocity.
        // entity_teleport: legacy large-distance entity position correction.
        // Both carry entityId + x/y/z + yaw.
        const entity = this._entities.get(data.entityId);
        if (entity) {
          const newPos = { x: data.x, y: data.y, z: data.z };
          const now = Date.now();
          const dt = (now - (entity._lastMoveTime || now)) / 1000;
          if (entity._prevPos && dt >= 0.01 && dt < 5.0) {
            entity.velocity = {
              x: (newPos.x - entity._prevPos.x) / dt,
              y: (newPos.y - entity._prevPos.y) / dt,
              z: (newPos.z - entity._prevPos.z) / dt,
            };
          }
          entity._prevPos = { ...newPos };
          entity._lastMoveTime = now;
          entity.position = newPos;
          if (data.yaw !== undefined) entity.yaw = data.yaw;
        }
        // If this is the target player, update our tracked position
        if (data.entityId === this._targetEntityId) {
          this._position = { x: data.x, y: data.y, z: data.z };
          if (data.yaw !== undefined) this._yaw = normalizeAngle(data.yaw);
        }
        break;
      }

      case 'rel_entity_move':
      case 'entity_move_look': {
        const entity = this._entities.get(data.entityId);
        if (entity && entity.position) {
          const newPos = {
            x: entity.position.x + (data.dX || 0) / 4096,
            y: entity.position.y + (data.dY || 0) / 4096,
            z: entity.position.z + (data.dZ || 0) / 4096,
          };
          const now = Date.now();
          const dt = (now - (entity._lastMoveTime || now)) / 1000;
          if (entity._prevPos && dt >= 0.01 && dt < 5.0) {
            entity.velocity = {
              x: (newPos.x - entity._prevPos.x) / dt,
              y: (newPos.y - entity._prevPos.y) / dt,
              z: (newPos.z - entity._prevPos.z) / dt,
            };
          }
          entity._prevPos = { ...newPos };
          entity._lastMoveTime = now;
          entity.position = newPos;
          if (data.yaw !== undefined) entity.yaw = data.yaw;
        }
        // If this is the target player, update our tracked position + yaw
        if (data.entityId === this._targetEntityId && this._entities.has(data.entityId)) {
          const ent = this._entities.get(data.entityId);
          if (ent && ent.position) {
            this._position = { x: ent.position.x, y: ent.position.y, z: ent.position.z };
          }
          if (data.yaw !== undefined) this._yaw = normalizeAngle(data.yaw);
        }
        break;
      }

      case 'player_rotation': {
        // MC 26.1+: separate yaw/pitch packet (0x47) with relative flags
        const relYaw = data.relativeYaw;
        const relPitch = data.relativePitch;
        // This is the OBSERVER's rotation — update if it matters for spatial awareness
        // (We mainly track _yaw/_pitch of the TARGET player, not the observer)
        break;
      }

      case 'entity_destroy': {
        const ids = data.entityIds || (data.entityId !== undefined ? [data.entityId] : []);
        for (const id of ids) {
          if (trackingState.attackedEntities.has(id)) {
            trackingState.killsSinceLastState++;
            trackingState.attackedEntities.delete(id);
          }
          this._entities.delete(id);
        }
        break;
      }

      case 'entity_metadata': {
        const entity = this._entities.get(data.entityId);
        if (entity) {
          for (const entry of (data.metadata || [])) {
            switch (entry.key) {
              case ENTITY_META.FLAGS:
                entity._flags = entry.value;
                break;
              case ENTITY_META.HAND_STATE:
                entity._handState = entry.value;
                break;
              case ENTITY_META.HEALTH:
                if (typeof entry.value === 'number') entity._health = entry.value;
                break;
              case ENTITY_META.ZOMBIE_BABY:
                entity._isBaby = !!entry.value;
                break;
              case ENTITY_META.CREEPER_STATE:
                entity._creeperState = entry.value;
                break;
              case ENTITY_META.CREEPER_CHARGED:
                entity._creeperCharged = !!entry.value;
                break;
            }
          }
        }
        // Target player death detection via health
        if (data.entityId === this._targetEntityId) {
          for (const entry of (data.metadata || [])) {
            if (entry.key === ENTITY_META.HEALTH && typeof entry.value === 'number') {
              const oldHealth = this._health;
              this._health = entry.value;
              if (this._health <= 0 && oldHealth > 0) {
                console.log('[attach] Player died (entity_metadata).');
                trackingState.pendingRespawn = true;
                this._triggerAutoRespawn(trackingState);
              }
            }
          }
        }
        break;
      }

      case 'map_chunk':
        if (this._blockCache) {
          this._blockCache.handleMapChunk(data);
          this._blockCache.prune(this._position);
        }
        break;

      case 'block_change':
        if (this._blockCache) this._blockCache.handleBlockChange(data);
        break;

      case 'multi_block_change':
        if (this._blockCache) this._blockCache.handleMultiBlockChange(data);
        break;

      case 'update_time':
        // Opaque packet (format changed in 26.1). Extract dayTime from raw bytes.
        if (data.data && data.data.length >= 16) {
          this._timeOfDay = Math.abs(Number(data.data.readBigInt64BE(8))) % 24000;
        } else if (data.time !== undefined) {
          this._timeOfDay = Math.abs(Number(data.time)) % 24000;
        }
        break;

      case 'player_info':
        this._handlePlayerInfo(data);
        break;

      case 'entity_status':
        if (data.entityStatus === 2) {
          if (data.entityId === this._targetEntityId) {
            // Target player took damage — identify likely attacker
            trackingState.knockbackCooldown = 2;
            const attacker = this._findNearestHostile();
            if (attacker) {
              trackingState.lastAttackerEntityId = attacker.id;
              trackingState.lastAttackerTime = Date.now();
            }
          } else if (this._lastItemReleaseTime &&
                     Date.now() - this._lastItemReleaseTime < 3000 &&
                     !trackingState.attackedEntities.has(data.entityId)) {
            // Non-self entity hurt within 3s of our projectile release
            trackingState.projectileHitLanded = true;
            trackingState.attackedEntities.add(data.entityId);
            const entity = this._entities.get(data.entityId);
            if (entity && entity.type === 'player') {
              trackingState.projectilePlayerHitLanded = true;
            }
          }
        }
        break;

      case 'entity_head_rotation': {
        const entity = this._entities.get(data.entityId);
        if (entity) {
          // headYaw is i8 (256ths of a turn) — convert to degrees
          entity.headYaw = normalizeAngle(data.headYaw * 360 / 256);
        }
        break;
      }

      case 'entity_look': {
        const entity = this._entities.get(data.entityId);
        if (entity) {
          if (data.yaw !== undefined) entity.yaw = normalizeAngle(data.yaw * 360 / 256);
          if (data.pitch !== undefined) entity.pitch = data.pitch * 360 / 256;
        }
        break;
      }

      case 'entity_equipment': {
        const entity = this._entities.get(data.entityId);
        if (entity) {
          if (!entity._equipment) entity._equipment = {};
          for (const equip of (data.equipments || [])) {
            entity._equipment[equip.slot] = equip.item;
          }
        }
        break;
      }

      case 'respawn':
        this._health = 20;
        trackingState.pendingRespawn = false;
        console.log('[attach] Player respawned.');
        break;

      case 'game_state_change': {
        // reason 3 = change_game_mode, gameMode 3.0 = spectator
        if (data.reason === 3 || data.reason === 'change_game_mode') {
          if (data.gameMode === 3 || data.gameMode === 3.0) {
            this._hasPermissions = true;
            this._awaitingSpectatorConfirm = false;
            console.log('[attach] Spectator mode confirmed (game_state_change packet).');
            if (!this._followInterval) this._startFollowLoop();
          }
        }
        // Weather tracking
        if (data.reason === 1) this._isRaining = true;
        else if (data.reason === 2) this._isRaining = false;
        else if (data.reason === 7) this._isThundering = data.gameMode > 0;
        break;
      }

      case 'entity_effect':
        if (data.entityId === this._targetEntityId) {
          this._activeEffects.set(data.effectId, {
            amplifier: data.amplifier || 0,
            duration: data.duration || 0,
            startTime: Date.now(),
          });
        }
        break;

      case 'remove_entity_effect':
        if (data.entityId === this._targetEntityId) {
          this._activeEffects.delete(data.effectId);
        }
        break;

      case 'damage_event':
        if (data.entityId === this._targetEntityId && trackingState) {
          trackingState.lastDamageSourceType = data.sourceTypeId || 0;
          trackingState.lastDamageTime = Date.now();
          if (data.sourceCauseId > 0) {
            trackingState.lastAttackerEntityId = data.sourceCauseId;
            trackingState.lastAttackerTime = Date.now();
          }
        }
        break;

      case 'system_chat': {
        // Parse chat component into plain text
        const content = data.content;
        let msg = '';
        if (typeof content === 'string') {
          msg = content;
        } else if (content) {
          msg = content.text || '';
          if (content.translate) msg = content.translate;
          if (content.with) {
            for (const part of content.with) {
              if (typeof part === 'string') msg += ' ' + part;
              else if (part && part.text) msg += ' ' + part.text;
            }
          }
          if (content.extra) {
            for (const part of content.extra) {
              if (typeof part === 'string') msg += part;
              else if (part && part.text) msg += part.text;
            }
          }
        }

        // Log all system chat while awaiting spectator (helps debug command failures)
        if (this._awaitingSpectatorConfirm || !this._spectatorSent) {
          console.log(`[attach] system_chat: ${msg || JSON.stringify(content)}`);
        }

        // Detect death of tracked player from broadcast death message
        if (this._targetUsername && msg.includes(this._targetUsername)) {
          // Common death message patterns (all start with player name)
          const deathPatterns = [
            'death.', 'was slain', 'was shot', 'was killed', 'was blown',
            'drowned', 'fell', 'burned', 'suffocated', 'starved', 'withered',
            'was pricked', 'hit the ground', 'went up in flames', 'walked into',
            'was squished', 'was impaled', 'was fireballed', 'was stung',
            'was squashed', 'tried to swim', 'was frozen', 'was struck',
            'didn\'t want to live', 'experienced kinetic energy',
          ];
          const msgLower = msg.toLowerCase();
          const isDeath = deathPatterns.some(p => msgLower.includes(p));
          if (isDeath) {
            console.log(`[attach] Death detected from chat: ${msg}`);
            this._health = 0;
            trackingState.pendingRespawn = true;
            this._triggerAutoRespawn(trackingState);
          }
        }
        break;
      }

      case 'position': {
        // This is the OBSERVER's position correction.
        // Must send BOTH teleport_confirm AND position_look back.
        // Without the position response, server considers client "not in world" and times out.
        try {
          this._client.write('teleport_confirm', {
            teleportId: data.teleportId || 0,
          });
        } catch (_) {}
        try {
          this._client.write('position_look', {
            x: data.x || 0,
            y: data.y || 64,
            z: data.z || 0,
            yaw: data.yaw || 0,
            pitch: data.pitch || 0,
            flags: { onGround: true, hasHorizontalCollision: false },
          });
        } catch (e) {
          // Fallback: try with numeric flags (bitfield: onGround=0x01)
          try {
            this._client.write('position_look', {
              x: data.x || 0,
              y: data.y || 64,
              z: data.z || 0,
              yaw: data.yaw || 0,
              pitch: data.pitch || 0,
              flags: 0x01,
            });
          } catch (_) {}
        }

        // After first position, switch to spectator mode
        if (!this._spectatorSent) {
          this._spectatorSent = true;
          this._awaitingSpectatorConfirm = true;
          setTimeout(() => {
            // Attempt 1: change_gamemode packet (protocol-level, no commands needed)
            try {
              this._client.write('change_gamemode', { mode: 3 });
              console.log('[attach] Sent change_gamemode(spectator) packet');
            } catch (e) {
              console.log('[attach] change_gamemode failed:', e.message);
            }
            // Attempt 2: chat_command after 2s (requires LAN cheats enabled)
            setTimeout(() => {
              if (!this._hasPermissions) {
                try {
                  this._client.write('chat_command', { command: 'gamemode spectator' });
                  console.log('[attach] Sent /gamemode spectator (chat_command)');
                } catch (e) {
                  console.log('[attach] chat_command failed:', e.message);
                }
              }
            }, 2000);
            // Wait for confirmation before starting follow loop
            setTimeout(() => {
              if (this._hasPermissions) {
                this._startFollowLoop();
              } else {
                this._spectatorFailed = true;
                this._awaitingSpectatorConfirm = false;
                console.warn('[attach] No op permissions detected. Observer will stay at spawn position.');
                console.warn('[attach] For full tracking: re-open LAN with "Game Mode: Spectator"');
                console.warn('[attach]   or enable "Allow Commands" when opening to LAN.');
              }
            }, 6000);
          }, 2000);
        }
        break;
      }

      // MC 1.20.2+ chunk batch acknowledgment — server sends chunks in batches
      // and expects the client to ACK each batch. Without this, server times out.
      case 'chunk_batch_finished':
        try {
          this._client.write('chunk_batch_received', { chunksPerTick: 20.0 });
        } catch (_) {}
        break;
    }
  }

  _handleSpawnEntity(data) {
    const typeId = data.type;
    const uuid = data.entityUUID || data.objectUUID;

    // UUID-based player detection (version-resilient — entity type IDs may differ between versions)
    const knownPlayerName = uuid ? this._playerInfo.get(uuid) : null;
    const entityName = knownPlayerName ? 'player' : this._getEntityName(typeId);

    // Log first few spawns to verify entity type resolution
    if (!this._spawnLogCount) this._spawnLogCount = 0;
    if (this._spawnLogCount < 20) {
      this._spawnLogCount++;
      console.log(`[attach] spawn_entity: typeId=${typeId} -> "${entityName}" pos=(${data.x?.toFixed(0)},${data.y?.toFixed(0)},${data.z?.toFixed(0)})`);
    }
    const isPlayer = !!knownPlayerName || entityName === 'player';
    const username = knownPlayerName || (isPlayer ? 'player' : null);

    // Get entity dimensions from minecraft-data
    const entityData = this._mcData ? this._mcData.entities[typeId] : null;
    const height = entityData ? entityData.height : (isPlayer ? 1.8 : 1.0);
    const width = entityData ? entityData.width : 0.6;

    this._entities.set(data.entityId, {
      id: data.entityId,
      runtimeId: data.entityId,
      type: isPlayer ? 'player' : entityName,
      name: isPlayer ? (username || 'player') : entityName,
      displayName: isPlayer ? username : entityName,
      username: username,
      _uuid: uuid || null,  // stored for deferred player reclassification
      position: { x: data.x, y: data.y, z: data.z },
      height,
      width,
      yaw: data.yaw || 0,
      velocity: { x: 0, y: 0, z: 0 },
      _prevPos: { x: data.x, y: data.y, z: data.z },
      _lastMoveTime: Date.now(),
    });

    // Track target player (26.1+ merged named_entity_spawn into spawn_entity)
    if (isPlayer && username && username === this._targetUsername) {
      this._targetEntityId = data.entityId;
      this._position = { x: data.x, y: data.y, z: data.z };
      if (data.yaw !== undefined) this._yaw = normalizeAngle(data.yaw);
      console.log(`[attach] Found target player "${this._targetUsername}" — entity ID: ${data.entityId}`);
    }
    // If no target username set, track the first non-observer player
    if (isPlayer && username && !this._targetEntityId && username !== 'SCAR_Observer') {
      this._targetUsername = username;
      this._targetEntityId = data.entityId;
      this._position = { x: data.x, y: data.y, z: data.z };
      if (data.yaw !== undefined) this._yaw = normalizeAngle(data.yaw);
      console.log(`[attach] Auto-tracking player "${username}" — entity ID: ${data.entityId}`);
    }
  }

  _handleNamedEntitySpawn(data) {
    const uuid = data.playerUUID;
    const username = this._playerInfo.get(uuid) || 'player';

    this._entities.set(data.entityId, {
      id: data.entityId,
      runtimeId: data.entityId,
      type: 'player',
      name: username,
      displayName: username,
      username: username,
      position: { x: data.x, y: data.y, z: data.z },
      height: 1.8,
      yaw: data.yaw || 0,
      velocity: { x: 0, y: 0, z: 0 },
      _prevPos: { x: data.x, y: data.y, z: data.z },
      _lastMoveTime: Date.now(),
    });

    // KEY: if this is the target player, track their entity ID
    if (username === this._targetUsername) {
      this._targetEntityId = data.entityId;
      this._position = { x: data.x, y: data.y, z: data.z };
      if (data.yaw !== undefined) this._yaw = normalizeAngle(data.yaw);
      console.log(`[attach] Found target player "${this._targetUsername}" — entity ID: ${data.entityId}`);
    }
  }

  _handlePlayerInfo(data) {
    const entries = data.data || [];
    for (const entry of entries) {
      const uuid = entry.uuid || entry.UUID;
      const name = entry.player?.name || entry.name;
      if (uuid && name) {
        const isNew = !this._playerInfo.has(uuid);
        this._playerInfo.set(uuid, name);

        // Deferred reclassification: if spawn_entity arrived BEFORE player_info,
        // the entity was registered as a non-player. Fix it now.
        if (isNew) {
          for (const [entityId, entity] of this._entities) {
            if (entity._uuid === uuid && entity.type !== 'player') {
              entity.type = 'player';
              entity.name = name;
              entity.displayName = name;
              entity.username = name;
              entity.height = 1.8;
              console.log(`[attach] Reclassified entity ${entityId} as player "${name}" (deferred UUID match)`);

              // Check if this is the target player
              if (name === this._targetUsername) {
                this._targetEntityId = entityId;
                this._position = { ...entity.position };
                if (entity.yaw !== undefined) this._yaw = entity.yaw;
                console.log(`[attach] Found target player "${name}" — entity ID: ${entityId} (deferred)`);
              }
              // Auto-track first non-observer player
              if (!this._targetEntityId && name !== 'SCAR_Observer') {
                this._targetUsername = name;
                this._targetEntityId = entityId;
                this._position = { ...entity.position };
                if (entity.yaw !== undefined) this._yaw = entity.yaw;
                console.log(`[attach] Auto-tracking player "${name}" — entity ID: ${entityId} (deferred)`);
              }
            }
          }
        }
      }
    }
  }

  /**
   * Find the nearest hostile entity to the tracked player.
   * Used for attacker identification when player takes damage.
   */
  _findNearestHostile() {
    if (!this._position) return null;
    const { HOSTILE_MOBS } = require('./categories');
    let best = null;
    let bestDist = Infinity;
    for (const entity of this._entities.values()) {
      if (!entity.position) continue;
      const name = (entity.name || '').toLowerCase();
      if (!HOSTILE_MOBS.has(name) && entity.type !== 'player') continue;
      const dx = this._position.x - entity.position.x;
      const dy = this._position.y - entity.position.y;
      const dz = this._position.z - entity.position.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < bestDist) {
        bestDist = dist;
        best = entity;
      }
    }
    return best;
  }

  _getEntityName(typeId) {
    // Prefer runtime registry from server (correct for any MC version)
    if (this._entityTypeRegistry && this._entityTypeRegistry.has(typeId)) {
      return this._entityTypeRegistry.get(typeId);
    }
    // Fall back to minecraft-data (may have wrong IDs for newer versions)
    if (!this._mcData) return `entity_${typeId}`;
    const entity = this._mcData.entities[typeId];
    return entity ? entity.name : `entity_${typeId}`;
  }

  /**
   * Capture entity type registry from server's registry_data packet (config state).
   */
  _captureEntityTypeRegistry(data) {
    const registryId = data.id || '';
    // Log all registry IDs to diagnose entity type capture
    if (!this._registryIdsLogged) this._registryIdsLogged = [];
    this._registryIdsLogged.push(`${registryId}(${(data.entries || []).length})`);
    // Dump all IDs after last registry arrives (when finish_configuration is sent)
    if (!this._registryDumpScheduled) {
      this._registryDumpScheduled = true;
      setTimeout(() => {
        console.log(`[attach] All registries (${this._registryIdsLogged.length}): ${this._registryIdsLogged.join(', ')}`);
      }, 2000);
    }
    if (!registryId.includes('entity_type')) return;

    const entries = data.entries || [];
    if (entries.length === 0) return;

    this._entityTypeRegistry = new Map();
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      // entry.key is like "minecraft:cow" — array index is the type ID
      const name = (entry.key || '').replace('minecraft:', '');
      if (name) {
        this._entityTypeRegistry.set(i, name);
      }
    }
    console.log(`[attach] Captured entity type registry: ${this._entityTypeRegistry.size} types`);
  }

  /**
   * Periodically move the headless observer near the tracked player
   * by sending position packets (works in spectator mode without /tp).
   */
  _startFollowLoop() {
    console.log('[attach] Follow loop started — observer will track player position.');
    this._followInterval = setInterval(() => {
      if (!this._client || !this._targetEntityId) return;
      const p = this._position;
      // Fly observer 20 blocks above the tracked player (spectator mode allows free flight)
      try {
        this._client.write('position', {
          x: p.x,
          y: p.y + 20,
          z: p.z,
          flags: { onGround: false, hasHorizontalCollision: false },
        });
      } catch (e) {
        // Fallback: try with numeric flags (bitfield)
        try {
          this._client.write('position', {
            x: p.x, y: p.y + 20, z: p.z, flags: 0,
          });
        } catch (_) {}
      }
    }, 2000); // Every 2 seconds
  }

  _stopFollowLoop() {
    if (this._followInterval) {
      clearInterval(this._followInterval);
      this._followInterval = null;
    }
  }

  // ---- Auto-yield helpers ----

  /**
   * Check if Minecraft is focused. Does NOT attempt to refocus —
   * AppleScript activation causes focus cycling and opens the game menu.
   * The bot simply skips actions when MC isn't focused and resumes
   * when the user switches back to MC.
   */
  _ensureFocus() {
    return this._mcFocused;
  }

  /** Whether the bot can send keyboard/mouse input right now. */
  _canAct() {
    return this._botControlActive && !this._userActive && this._mcFocused;
  }

  /** Whether the user is currently active (for state dict). */
  get userActive() { return this._userActive; }

  // ---- Adapter interface (properties) ----

  get ready() { return this._ready; }

  get position() {
    const p = this._position;
    return {
      x: p.x, y: p.y, z: p.z,
      floored: () => ({ x: Math.floor(p.x), y: Math.floor(p.y), z: Math.floor(p.z) }),
      distanceTo: (other) => Math.sqrt(
        (p.x - other.x) ** 2 + (p.y - other.y) ** 2 + (p.z - other.z) ** 2
      ),
    };
  }

  get flooredPosition() {
    return {
      x: Math.floor(this._position.x),
      y: Math.floor(this._position.y),
      z: Math.floor(this._position.z),
    };
  }

  get health() { return this._health; }
  get food() { return this._food; }                   // hardcoded 20
  get foodSaturation() { return this._foodSaturation; } // hardcoded 5
  get yaw() { return this._yaw; }
  get pitch() { return this._pitch; }
  get onGround() { return this._onGround; }
  get isInWater() { return false; }
  get isRaining() { return this._isRaining; }
  get isThundering() { return this._isThundering; }
  get activeEffects() { return this._activeEffects; }
  get timeOfDay() { return this._timeOfDay; }
  get xpLevel() { return this._xpLevel; }
  get xpPoints() { return 0; }
  get quickBarSlot() { return this._quickBarSlot; }

  get allEntities() {
    // Return all entities EXCEPT our observer and the target player (don't count self)
    return Array.from(this._entities.values())
      .filter(e => e.id !== this._entityId && e.id !== this._targetEntityId);
  }

  get inventoryItems() { return this._inventory; } // empty for now

  get bedrockClient() { return null; }

  // ---- Adapter interface (methods) ----

  isSelf(entity) {
    return entity.id === this._targetEntityId;
  }

  lightAt(_pos) {
    if (this._timeOfDay >= 12500 && this._timeOfDay <= 23500) return 4;
    return 15;
  }

  blockAt(pos) {
    if (!this._blockCache) return null;
    return this._blockCache.blockAt(pos);
  }

  nearestEntity(filter) {
    let nearest = null;
    let nearestDist = Infinity;
    for (const entity of this._entities.values()) {
      if (!filter(entity)) continue;
      if (!entity.position) continue;
      const dx = this._position.x - entity.position.x;
      const dy = this._position.y - entity.position.y;
      const dz = this._position.z - entity.position.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearest = entity;
      }
    }
    return nearest;
  }

  async setControlState(key, val) {
    this._controlStates[key] = val;
    if (!this._keyboardFallback) return;
    if (!this._canAct()) return;
    if (this._inputMonitor) this._inputMonitor.markBotAction();
    await this._keyboardFallback.setControlState(key, val);
  }

  async clearControlStates() {
    for (const key of Object.keys(this._controlStates)) {
      this._controlStates[key] = false;
    }
    if (this._keyboardFallback) {
      if (this._inputMonitor) this._inputMonitor.markBotAction();
      await this._keyboardFallback.clearControlStates();
    }
  }

  async attack(_entity) {
    // In attach mode, we can't target specific entities via packets.
    // We look at the target (handled by actions.js lookAt call before attack)
    // and left-click.
    if (!this._canAct()) return;
    if (this._inputMonitor) this._inputMonitor.markBotAction();
    if (this._keyboardFallback) {
      await this._keyboardFallback.attack();
    }
  }

  async lookAt(pos) {
    const dx = pos.x - this._position.x;
    const dy = pos.y - this._position.y;
    const dz = pos.z - this._position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    const newYaw = -Math.atan2(dx, dz) * (180 / Math.PI);
    const newPitch = -Math.atan2(dy, dist) * (180 / Math.PI);
    await this.look(newYaw, newPitch);
  }

  async look(yaw, pitch) {
    if (this._canAct()) {
      if (this._inputMonitor) this._inputMonitor.markBotAction();
      const deltaYaw = (yaw - this._yaw) * (Math.PI / 180);
      const deltaPitch = (pitch - this._pitch) * (Math.PI / 180);
      if (this._nativeMouseDelta) {
        this._nativeMouseDelta.move(deltaYaw, deltaPitch);
      } else if (this._keyboardFallback) {
        await this._keyboardFallback.look(deltaYaw, deltaPitch);
      }
    }
    this._yaw = yaw;
    this._pitch = pitch;
  }

  async swingArm() {
    if (!this._canAct()) return;
    if (this._inputMonitor) this._inputMonitor.markBotAction();
    if (this._keyboardFallback) await this._keyboardFallback.swingArm();
  }

  async activateItem() {
    if (!this._canAct()) return;
    if (this._inputMonitor) this._inputMonitor.markBotAction();
    if (this._keyboardFallback) await this._keyboardFallback.activateItem();
  }

  async pressUseItem() {
    this._isUsingItem = true;
    if (!this._canAct()) return;
    if (this._inputMonitor) this._inputMonitor.markBotAction();
    if (this._keyboardFallback) await this._keyboardFallback.pressUseItem();
  }

  async releaseUseItem() {
    this._isUsingItem = false;
    this._lastItemReleaseTime = Date.now();
    if (!this._canAct()) return;
    if (this._inputMonitor) this._inputMonitor.markBotAction();
    if (this._keyboardFallback) await this._keyboardFallback.releaseUseItem();
  }

  async setQuickBarSlot(slot) {
    this._quickBarSlot = slot;
    if (!this._canAct()) return;
    if (this._inputMonitor) this._inputMonitor.markBotAction();
    if (this._keyboardFallback) await this._keyboardFallback.setQuickBarSlot(slot);
  }

  chat(msg) {
    // Can't chat as the original player -- this is the observer
    console.log(`[attach] Chat suppressed (attach mode): ${msg}`);
  }

  /**
   * Respawn by clicking the "Respawn" button on the death screen.
   * In Minecraft Java, the Respawn button is centered horizontally,
   * slightly above center vertically.
   */
  async respawn() {
    // If health already recovered (MC auto-respawn), skip the click entirely
    if (this._health > 0) {
      console.log('[attach] Respawn skipped — player already alive (MC auto-respawn).');
      return;
    }
    if (!this._keyboardFallback) {
      console.warn('[attach] No keyboard fallback — cannot click Respawn button.');
      return;
    }

    console.log('[attach] Clicking Respawn button...');

    // Ensure Minecraft is focused
    const { focusMinecraft, getMcWindowBounds } = require('./lan-automation');
    await focusMinecraft();
    const { sleep } = require('./utils');
    await sleep(300);

    // Get window bounds to calculate Respawn button position
    const bounds = getMcWindowBounds();
    let cx, cy, guiScale;

    if (bounds) {
      cx = bounds.x + Math.round(bounds.width / 2);
      cy = bounds.y + Math.round(bounds.height / 2);
      guiScale = Math.max(1, Math.min(4, Math.floor(Math.min(bounds.width, bounds.height) / 240)));
    } else {
      // Fallback: use approximate screen center
      try {
        const nut = require('@nut-tree-fork/nut-js');
        const sw = await nut.screen.width();
        const sh = await nut.screen.height();
        cx = Math.round(sw / 2);
        cy = Math.round(sh / 2);
      } catch (_) {
        cx = 960;
        cy = 540;
      }
      guiScale = 2;
    }

    // Death screen layout: "Respawn" button is at approximately y = center - 10*guiScale
    // (slightly above center, first of two buttons)
    const respawnY = cy - Math.round(10 * guiScale);

    if (this._inputMonitor) this._inputMonitor.markBotAction();
    try {
      const nut = require('@nut-tree-fork/nut-js');
      await nut.mouse.setPosition({ x: cx, y: respawnY });
      await sleep(100);
      await nut.mouse.click(nut.Button.LEFT);
      console.log(`[attach] Clicked Respawn at (${cx}, ${respawnY})`);
    } catch (e) {
      console.warn('[attach] Failed to click Respawn button:', e.message);
    }

    await sleep(1000);
  }

  /**
   * Auto-respawn: click Respawn after a short delay when death is detected.
   * Debounced — multiple death signals (entity_metadata + system_chat) don't
   * trigger multiple clicks.
   */
  _triggerAutoRespawn(trackingState) {
    if (!this._autoRespawn || this._respawnPending) return;
    this._respawnPending = true;
    console.log('[attach] Death detected. Waiting for auto-respawn...');
    setTimeout(async () => {
      // Check if MC's built-in auto-respawn already handled it
      if (this._health > 0) {
        console.log('[attach] Player already respawned (MC auto-respawn).');
        trackingState.pendingRespawn = false;
        this._respawnPending = false;
        return;
      }
      // Still dead — try clicking the Respawn button
      console.log('[attach] Player still dead — clicking Respawn button.');
      try {
        await this.respawn();
        this._health = 20;
        trackingState.pendingRespawn = false;
      } catch (e) {
        console.warn('[attach] Auto-respawn failed:', e.message);
      }
      this._respawnPending = false;
    }, 3000);
  }

  async craftPlanks() {
    console.log('[attach] Crafting not available in attach mode.');
  }

  async craftToolOrSticks() {
    console.log('[attach] Crafting not available in attach mode.');
  }

  // ---- Bot control ----

  enableBotControl() {
    this._botControlActive = true;
    console.log('[attach] Bot control ENABLED — make sure Minecraft is focused.');
  }

  disableBotControl() {
    this._botControlActive = false;
    for (const key of Object.keys(this._controlStates)) {
      this._controlStates[key] = false;
    }
    if (this._keyboardFallback) {
      this._keyboardFallback.clearControlStates().catch(() => {});
    }
    console.log('[attach] Bot control DISABLED.');
  }

  get botControlActive() { return this._botControlActive; }

  // ---- Disconnect ----

  disconnect() {
    this._ready = false;
    this._stopFollowLoop();
    if (this._tickEndInterval) {
      clearInterval(this._tickEndInterval);
      this._tickEndInterval = null;
    }
    if (this._inputMonitor) {
      try { this._inputMonitor.stop(); } catch (_) {}
      this._inputMonitor = null;
    }
    if (this._nativeMouseDelta) {
      try { this._nativeMouseDelta.stop(); } catch (_) {}
    }
    if (this._client) {
      try { this._client.end(); } catch (_) {}
      this._client = null;
    }
  }
}

module.exports = { AttachAdapter };
