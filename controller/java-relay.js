/**
 * JavaRelayAdapter — transparent MITM proxy for Java Edition using minecraft-protocol.
 *
 * Same architecture as RelayAdapter (Bedrock):
 *   Real Java Client → localhost:25565 → Proxy Server → Upstream Java Server
 *                                              ↕
 *                                    Reads clientbound packets (state)
 *                                    Injects serverbound packets (actions)
 *                                    Suppresses client packets when bot drives
 *
 * Key differences from Bedrock:
 * - Movement: client-side physics — we inject position_look packets (server validates)
 * - Attack cooldown: tracked internally for 1.9+ PvP
 * - Entity IDs: integer (not BigInt runtime IDs)
 * - Chunk/block parsing: JavaBlockCache parses map_chunk / block_change / multi_block_change
 * - Auth: local server runs offline-mode, upstream uses Microsoft auth if needed
 *
 * Implements the same adapter interface as RelayAdapter so bridge.js, state.js,
 * and actions.js work unchanged.
 */

/** Normalize angle to [-180, 180] range. */
function normalizeAngle(a) {
  a = a % 360;
  if (a > 180) a -= 360;
  if (a < -180) a += 360;
  return a;
}

const mc = require('minecraft-protocol');
const { sleep } = require('./utils');
const { JavaBlockCache } = require('./java-block-cache');
const { ENTITY_META } = require('./categories');

class JavaRelayAdapter {
  constructor(config) {
    this.config = config;
    this._server = null;     // mc.Server (listens for real client)
    this._upstream = null;   // mc.Client (connects to target server)
    this._player = null;     // The connected client (real player)
    this._ready = false;
    this._resolveStart = null;

    // --- State tracked from clientbound packets ---
    this._position = { x: 0, y: 64, z: 0 };
    this._health = 20;
    this._food = 20;
    this._foodSaturation = 5;
    this._yaw = 0;
    this._pitch = 0;
    this._onGround = true;
    this._isInWater = false;
    this._timeOfDay = 0;
    this._xpLevel = 0;
    this._xpPoints = 0;
    this._quickBarSlot = 0;
    this._entityId = null;

    // Entity registry
    this._entities = new Map();

    // Runtime entity type ID → name (from server's registry_data, overrides mcData)
    this._entityTypeRegistry = null;

    // Player info (UUID → username)
    this._playerInfo = new Map();

    // Inventory
    this._inventory = [];

    // Version info + minecraft-data for entity/item lookups
    this._version = null;
    this._mcData = null;

    // Block cache (created after mcData loads)
    this._blockCache = null;

    // --- Bot control state ---
    this._botControlActive = false;
    this._transitionStart = 0;
    this._transitionDurationMs = 0;

    // Keyboard fallback
    this._keyboardFallback = null;
    this._kbFallbackActive = false;

    // Native mouse delta (macOS CGEvent helper for pointer-lock-safe aiming)
    this._nativeMouseDelta = null;

    // Simulated control states
    this._controlStates = {
      forward: false, back: false, left: false, right: false,
      jump: false, sprint: false, sneak: false,
    };

    // (No movement interval — keyboard-driven via real client)

    // Active status effects (effect ID → {amplifier, duration, startTime})
    this._activeEffects = new Map();

    // Weather state
    this._isRaining = false;
    this._isThundering = false;

    // Crafting
    this._javaRecipes = null;  // Map: resultName → recipeInfo
    this._windowActionId = 1;  // Incrementing action ID for window_click

    // Biome
    this._biome = 'plains';
  }

  /**
   * Start the proxy server.
   * Listens for the real Java client, then connects upstream.
   */
  async start(trackingState) {
    // Patch ALL unsupported versions before any minecraft-protocol call
    const { patchAllUnsupported, patchVersionSupport } = require('./version-compat');
    patchAllUnsupported();

    // Auto-detect upstream server version so the proxy sends matching
    // registry data during configuration. Without this, the proxy defaults
    // to the latest version, causing "Not a map: null null" decoder errors
    // when the client is on a different version.
    let serverVersion = this.config.version || false;
    let pingProtocol = null;
    try {
      const host = this.config.serverHost || 'localhost';
      const port = this.config.serverPort || 25565;
      const result = await new Promise((res, rej) => {
        mc.ping({ host, port }, (err, data) => err ? rej(err) : res(data));
      });
      pingProtocol = result.version.protocol;
      if (!serverVersion) serverVersion = result.version.name;
      console.log(`[java-relay] Detected upstream: ${result.version.name} (protocol ${pingProtocol})`);
    } catch (e) {
      console.warn('[java-relay] Could not ping upstream, using default version.');
    }
    if (serverVersion) {
      serverVersion = patchVersionSupport(serverVersion, pingProtocol) || serverVersion;
    }

    // Override protocol number if ping reports different from patched data
    // (MC 26.1+ reports protocol 767 but uses 26.1 packet format)
    if (serverVersion && pingProtocol) {
      const mcDataCheck = require('minecraft-data');
      try {
        const vd = mcDataCheck(serverVersion);
        if (vd && vd.version.version !== pingProtocol) {
          console.log(`[java-relay] Protocol override: data=${vd.version.version}, server=${pingProtocol}`);
          const dataModule = require('minecraft-data/data');
          const entry = dataModule.pc[vd.version.majorVersion];
          if (entry) {
            const origVer = vd.version;
            Object.defineProperty(entry, 'version', {
              get() { return { ...origVer, version: pingProtocol }; },
              enumerable: true, configurable: true,
            });
          }
        }
      } catch (_) {}
    }

    return new Promise((resolve, reject) => {
      const listenPort = this.config.listenPort || 25565;

      this._server = mc.createServer({
        host: '0.0.0.0',
        port: listenPort,
        'online-mode': false,
        keepAlive: false,  // Relay keep_alive from upstream
        version: serverVersion,
      });

      this._resolveStart = resolve;

      this._server.on('login', (player) => {
        if (this._player) {
          player.end('Another client is already connected.');
          return;
        }

        console.log(`[java-relay] Client connected: ${player.username}`);
        this._player = player;

        // Store version for minecraft-data lookups
        this._version = player.version;
        try {
          this._mcData = require('minecraft-data')(player.version);
        } catch (e) {
          console.warn('[java-relay] Could not load minecraft-data for version', player.version);
          this._mcData = null;
        }

        // Initialize block cache
        if (this._mcData) {
          this._blockCache = new JavaBlockCache(this._mcData);
          console.log('[java-relay] Block cache initialized.');
        }

        // Connect upstream
        const upstreamOpts = {
          host: this.config.serverHost || 'localhost',
          port: this.config.serverPort || 25565,
          username: player.username,
          keepAlive: false,
          version: player.version,
        };

        if (this.config.onlineMode) {
          upstreamOpts.auth = 'microsoft';
          upstreamOpts.profilesFolder = this.config.authCache || './auth_cache';
        }

        console.log(`[java-relay] Connecting upstream to ${upstreamOpts.host}:${upstreamOpts.port}...`);
        this._upstream = mc.createClient(upstreamOpts);

        this._upstream.on('error', (err) => {
          console.error('[java-relay] Upstream error:', err.message);
          if (!this._ready) reject(err);
        });

        this._setupProxy(player, this._upstream, trackingState);
      });

      this._server.on('error', (err) => {
        console.error('[java-relay] Server error:', err.message);
        if (!this._ready) reject(err);
      });

      console.log(`[java-relay] Listening on port ${listenPort}`);
      console.log(`[java-relay] Connect your Minecraft Java client to localhost:${listenPort}`);
    });
  }

  _setupProxy(player, upstream, trackingState) {
    // Clientbound: upstream → player (read state + forward raw)
    // Use writeRaw(buffer) to avoid re-serialization issues with NBT packets.
    // The packet event provides (data, meta, buffer, fullBuffer) — we parse
    // data for state tracking but forward the raw buffer to the client.
    upstream.on('packet', (data, meta, buffer) => {
      // Capture entity type registry from config state (before play state)
      if (meta.state === 'configuration' && meta.name === 'registry_data') {
        this._captureEntityTypeRegistry(data);
      }
      if (meta.state !== 'play') return;
      this._handleClientbound(meta.name, data, trackingState);
      if (player.state === 'play') {
        try {
          if (buffer) {
            player.writeRaw(buffer);
          } else {
            player.write(meta.name, data);
          }
        }
        catch (_) {}
      }
    });

    // Serverbound: player → upstream (suppress when bot drives, rewrite look)
    // Use writeRaw for unmodified packets, write() only when rewriting.
    player.on('packet', (data, meta, buffer) => {
      if (meta.state !== 'play') return;
      if (this._shouldSuppressServerbound(meta.name)) return;
      const rewritten = this._rewriteServerbound(meta.name, data);
      if (upstream.state === 'play') {
        try {
          if (rewritten !== data) {
            upstream.write(meta.name, rewritten);
          } else if (buffer) {
            upstream.writeRaw(buffer);
          } else {
            upstream.write(meta.name, data);
          }
        }
        catch (_) {}
      }
    });

    player.on('end', () => {
      console.log('[java-relay] Client disconnected.');
      this._ready = false;
      this._player = null;
      this._stopMovementLoop();
      if (this._upstream) {
        this._upstream.end();
        this._upstream = null;
      }
    });

    upstream.on('end', () => {
      console.log('[java-relay] Upstream disconnected.');
      this._ready = false;
      this._stopMovementLoop();
      if (player) {
        try { player.end('Server disconnected.'); }
        catch (_) {}
      }
    });
  }

  /**
   * Process clientbound packets (server → client).
   * Extract game state without modifying packets.
   */
  _handleClientbound(name, data, trackingState) {
    switch (name) {
      case 'login':
        this._entityId = data.entityId;
        console.log(`[java-relay] Game joined. Entity ID: ${this._entityId}`);
        if (!this._ready) {
          this._ready = true;
          this._startMovementLoop();
          if (this._resolveStart) {
            this._resolveStart();
            this._resolveStart = null;
          }
        }
        break;

      case 'update_health': {
        const oldHealth = this._health;
        this._health = data.health;
        this._food = data.food;
        this._foodSaturation = data.foodSaturation;
        if (this._health <= 0 && oldHealth > 0) {
          console.log('[java-relay] Player died.');
          trackingState.pendingRespawn = true;
        }
        break;
      }

      case 'position': {
        // Server position correction — may be relative or absolute.
        // In 26.1+, flags is PositionUpdateRelatives (object with boolean props).
        // In older versions, flags is an integer bitmask.
        const f = data.flags || 0;
        const isObj = typeof f === 'object';
        this._position = {
          x: (isObj ? f.x : (f & 0x01)) ? this._position.x + data.x : data.x,
          y: (isObj ? f.y : (f & 0x02)) ? this._position.y + data.y : data.y,
          z: (isObj ? f.z : (f & 0x04)) ? this._position.z + data.z : data.z,
        };
        this._yaw = (isObj ? f.yaw : (f & 0x08)) ? this._yaw + data.yaw : (data.yaw || 0);
        this._pitch = (isObj ? f.pitch : (f & 0x10)) ? this._pitch + data.pitch : (data.pitch || 0);
        this._yaw = normalizeAngle(this._yaw);
        this._pitch = Math.max(-90, Math.min(90, this._pitch));
        this._onGround = true;
        this._yVelocity = 0;
        // Respond with teleport_confirm if teleportId is present (26.1+)
        if (data.teleportId !== undefined) {
          try {
            this._upstream.write('teleport_confirm', { teleportId: data.teleportId });
          } catch (_) {}
        }
        break;
      }

      case 'player_rotation': {
        // 26.1+: separate yaw/pitch packet (no position change)
        const relYaw = data.relativeYaw;
        const relPitch = data.relativePitch;
        this._yaw = relYaw ? this._yaw + data.yaw : (data.yaw || 0);
        this._pitch = relPitch ? this._pitch + data.pitch : (data.pitch || 0);
        this._yaw = normalizeAngle(this._yaw);
        this._pitch = Math.max(-90, Math.min(90, this._pitch));
        break;
      }

      case 'sync_entity_position': {
        // 26.1+: authoritative entity position correction (replaces some entity_teleport uses)
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
        break;
      }

      case 'spawn_entity':
        this._handleSpawnEntity(data);
        break;

      case 'spawn_entity_living':
        this._handleSpawnEntity(data);
        break;

      case 'named_entity_spawn':
        this._handleNamedEntitySpawn(data);
        break;

      case 'entity_teleport': {
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

      case 'update_time':
        // Opaque packet (format changed in 26.1). Extract dayTime from raw bytes.
        if (data.data && data.data.length >= 16) {
          this._timeOfDay = Math.abs(Number(data.data.readBigInt64BE(8))) % 24000;
        } else if (data.time !== undefined) {
          this._timeOfDay = Math.abs(Number(data.time)) % 24000;
        }
        break;

      case 'window_items':
        if (data.windowId === 0) {
          this._parseInventory(data);
        }
        break;

      case 'set_slot':
        if (data.windowId === 0) {
          this._handleSetSlot(data);
        }
        break;

      case 'respawn':
        this._health = 20;
        this._food = 20;
        trackingState.pendingRespawn = false;
        console.log('[java-relay] Player respawned.');
        break;

      case 'entity_status':
        if (data.entityStatus === 2) {
          if (data.entityId === this._entityId) {
            // Player took damage — identify likely attacker
            trackingState.knockbackCooldown = 2;
            const attacker = this._findNearestHostile();
            if (attacker) {
              trackingState.lastAttackerEntityId = attacker.id;
              trackingState.lastAttackerTime = Date.now();
            }
            if (this._botControlActive) {
              this.clearControlStates();
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
        break;
      }

      case 'held_item_slot':
        if (data.slot !== undefined) {
          this._quickBarSlot = data.slot;
        }
        break;

      case 'player_info':
        this._handlePlayerInfo(data);
        break;

      case 'experience':
        this._xpLevel = data.level || 0;
        this._xpPoints = data.totalExperience || 0;
        break;

      case 'entity_effect':
        if (data.entityId === this._entityId) {
          this._activeEffects.set(data.effectId, {
            amplifier: data.amplifier || 0,
            duration: data.duration || 0,
            startTime: Date.now(),
          });
        }
        break;

      case 'remove_entity_effect':
        if (data.entityId === this._entityId) {
          this._activeEffects.delete(data.effectId);
        }
        break;

      case 'game_state_change':
        if (data.reason === 1) this._isRaining = true;
        else if (data.reason === 2) this._isRaining = false;
        else if (data.reason === 7) this._isThundering = data.gameMode > 0;
        break;

      case 'damage_event':
        if (data.entityId === this._entityId && trackingState) {
          trackingState.lastDamageSourceType = data.sourceTypeId || 0;
          trackingState.lastDamageTime = Date.now();
          if (data.sourceCauseId > 0) {
            trackingState.lastAttackerEntityId = data.sourceCauseId;
            trackingState.lastAttackerTime = Date.now();
          }
        }
        break;

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

      case 'declare_recipes':
        this._parseJavaRecipes(data);
        break;

      case 'unlock_recipes':
        // unlock_recipes is sent after declare_recipes to tell client
        // which recipes are available. We already have all recipes from
        // declare_recipes, so just log it.
        break;
    }
  }

  _handleSpawnEntity(data) {
    const typeId = data.type;
    const entityName = this._getEntityName(typeId);
    const isPlayer = entityName === 'player';
    const uuid = data.entityUUID || data.objectUUID;
    const username = isPlayer ? (this._playerInfo.get(uuid) || 'player') : null;

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
      position: { x: data.x, y: data.y, z: data.z },
      height,
      width,
      yaw: data.yaw || 0,
      velocity: { x: 0, y: 0, z: 0 },
      _prevPos: { x: data.x, y: data.y, z: data.z },
      _lastMoveTime: Date.now(),
    });
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
  }

  _handlePlayerInfo(data) {
    const entries = data.data || [];
    for (const entry of entries) {
      const uuid = entry.uuid || entry.UUID;
      const name = entry.player?.name || entry.name;
      if (uuid && name) {
        this._playerInfo.set(uuid, name);
      }
    }
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
   * Builds a runtime typeId → name map that overrides minecraft-data's static IDs.
   */
  _captureEntityTypeRegistry(data) {
    const registryId = data.id || '';
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
    console.log(`[java-relay] Captured entity type registry: ${this._entityTypeRegistry.size} types`);
  }

  _parseInventory(data) {
    this._inventory = [];
    const items = data.items || [];
    for (let i = 0; i < items.length; i++) {
      // Only hotbar slots (window 36-44) — avoids collision with armor/crafting 0-8
      if (i < 36 || i > 44) continue;
      const item = items[i];
      if (!item) continue;
      // 1.13+: has 'present' field. Pre-1.13: blockId !== -1
      if (item.present === false) continue;
      if (item.blockId !== undefined && item.blockId === -1) continue;
      if (!item.itemId && !item.blockId) continue;

      const slot = i - 36;
      const itemName = this._getItemName(item);
      this._inventory.push({
        name: itemName,
        count: item.itemCount || 1,
        slot,
        network_id: item.itemId || item.blockId || 0,
        metadata: item.nbtData || {},
      });
    }
  }

  _handleSetSlot(data) {
    const item = data.item;
    const windowSlot = data.slot;
    // Only hotbar slots (window 36-44) — avoids collision with armor/crafting 0-8
    if (windowSlot < 36 || windowSlot > 44) return;

    const slot = windowSlot - 36;

    // Remove old item in this slot
    this._inventory = this._inventory.filter(i => i.slot !== slot);

    if (!item) return;
    if (item.present === false) return;
    if (item.blockId !== undefined && item.blockId === -1) return;
    if (!item.itemId && !item.blockId) return;

    const itemName = this._getItemName(item);
    this._inventory.push({
      name: itemName,
      count: item.itemCount || 1,
      slot,
      network_id: item.itemId || item.blockId || 0,
      metadata: item.nbtData || {},
    });
  }

  _getItemName(item) {
    const itemId = item.itemId || item.blockId;
    if (!itemId || !this._mcData) return `item_${itemId || 0}`;
    const info = this._mcData.items[itemId];
    return info ? info.name : `item_${itemId}`;
  }

  _parseJavaRecipes(data) {
    this._javaRecipes = new Map();
    const recipes = data.recipes || data;
    if (!Array.isArray(recipes)) return;

    for (const recipe of recipes) {
      if (!recipe.result) continue;
      const resultId = recipe.result.itemId;
      if (!resultId) continue;
      const resultInfo = this._mcData ? this._mcData.items[resultId] : null;
      const resultName = resultInfo ? resultInfo.name : `item_${resultId}`;

      let ingredientIds = [];
      if (recipe.inShape) {
        // Shaped recipe
        for (const row of recipe.inShape) {
          for (const cell of row) {
            if (cell && cell.itemId) ingredientIds.push(cell.itemId);
          }
        }
      } else if (recipe.ingredients) {
        // Shapeless recipe
        for (const ing of recipe.ingredients) {
          if (ing && ing.itemId) ingredientIds.push(ing.itemId);
        }
      }

      const ingredients = ingredientIds.map(id => {
        const info = this._mcData ? this._mcData.items[id] : null;
        return info ? info.name : `item_${id}`;
      });

      // Store first matching recipe per result name (prefer simpler recipes)
      if (!this._javaRecipes.has(resultName) || ingredients.length < this._javaRecipes.get(resultName).ingredients.length) {
        this._javaRecipes.set(resultName, {
          recipeId: recipe.recipeId || '',
          type: recipe.type || '',
          ingredients,
          width: recipe.width || 0,
          height: recipe.height || 0,
          resultCount: recipe.result.itemCount || 1,
        });
      }
    }

    console.log(`[java-relay] Parsed ${this._javaRecipes.size} recipes from declare_recipes.`);
  }

  _hasIngredient(name) {
    return this._inventory.some(item => item.name === name && item.count > 0);
  }

  _countIngredient(name) {
    return this._inventory
      .filter(item => item.name === name)
      .reduce((sum, item) => sum + item.count, 0);
  }

  /**
   * Determine if a serverbound packet should be suppressed when bot is driving.
   *
   * The real client handles all movement physics via keyboard simulation.
   * Position/look packets pass through — they're the real physics output.
   * We only suppress action packets so user clicks don't conflict with
   * bot-injected attacks.
   */
  _shouldSuppressServerbound(name) {
    if (!this._botControlActive || this._kbFallbackActive) return false;

    const suppressed = new Set([
      'use_entity',
      'arm_animation',
      'block_dig',
      'block_place',
      'use_item',
    ]);

    if (!suppressed.has(name)) return false;

    // During bot control transition, gradually increase suppression
    if (this._transitionStart > 0) {
      const elapsed = Date.now() - this._transitionStart;
      if (elapsed < this._transitionDurationMs) {
        const progress = elapsed / this._transitionDurationMs;
        const t = progress * progress * (3 - 2 * progress); // smoothstep
        if (Math.random() > t) return false; // Let packet through
      } else {
        this._transitionStart = 0;
      }
    }

    return true;
  }

  /**
   * Rewrite serverbound packets when bot is driving.
   * - Tracks client's real position from its own movement packets.
   * - Replaces yaw/pitch in look packets with bot's desired aim so the
   *   server sees correct combat aim regardless of where the spectator's
   *   camera is pointed.
   */
  _rewriteServerbound(name, data) {
    // Track client's real position from its own packets
    if (name === 'position' || name === 'position_look') {
      this._position.x = data.x;
      this._position.y = data.y;
      this._position.z = data.z;
    }

    // Rewrite look when bot is driving
    if (!this._botControlActive) return data;

    if (name === 'look' || name === 'position_look') {
      return { ...data, yaw: this._yaw, pitch: this._pitch };
    }
    return data;
  }

  // ---- Bot control ----

  enableBotControl() {
    this._botControlActive = true;
    // Gradual transition: ramp suppression over 1.5-2.5s
    this._transitionStart = Date.now();
    this._transitionDurationMs = 1500 + Math.random() * 1000;
    console.log('[java-relay] Bot control ENABLED — ramping up.');
  }

  disableBotControl() {
    this._botControlActive = false;
    this._transitionStart = 0;
    // Release all keys before clearing the flag so clearControlStates
    // still sees botControlActive=false and skips keyboard (user takes over)
    for (const key of Object.keys(this._controlStates)) {
      this._controlStates[key] = false;
    }
    if (this._keyboardFallback) {
      this._keyboardFallback.clearControlStates().catch(() => {});
    }
    console.log('[java-relay] Bot control DISABLED — user in control.');
  }

  get botControlActive() { return this._botControlActive; }

  setKeyboardFallback(fb) { this._keyboardFallback = fb; }
  setNativeMouseDelta(nmd) { this._nativeMouseDelta = nmd; }

  // ---- Movement ----
  //
  // Java Edition requires the CLIENT to send position packets (unlike Bedrock
  // where the server handles physics from input flags). We let the real client
  // do this — setControlState() presses keys via keyboard simulation, the client
  // runs its own physics (collision, gravity, water, ladders — everything), and
  // sends valid position packets to the server. No physics simulation needed.
  //
  // This is why --keyboard-fallback is required for Java. Without it, the bot
  // can still observe state but cannot control movement.

  _startMovementLoop() {
    // No loop needed — setControlState() drives keyboard directly,
    // and the real client handles physics + position packets.
  }

  _stopMovementLoop() {
    // Nothing to stop — movement is keyboard-driven.
  }

  // ---- Adapter interface (identical to RelayAdapter) ----

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
  get food() { return this._food; }
  get foodSaturation() { return this._foodSaturation; }
  get yaw() { return this._yaw; }
  get pitch() { return this._pitch; }
  get onGround() { return this._onGround; }
  get isInWater() { return this._isInWater; }
  get isRaining() { return this._isRaining; }
  get isThundering() { return this._isThundering; }
  get activeEffects() { return this._activeEffects; }
  get timeOfDay() { return this._timeOfDay; }
  get xpLevel() { return this._xpLevel; }
  get xpPoints() { return this._xpPoints; }
  get quickBarSlot() { return this._quickBarSlot; }
  get allEntities() { return Array.from(this._entities.values()); }
  get inventoryItems() { return this._inventory; }

  get bedrockClient() { return null; } // Not Bedrock

  _findNearestHostile() {
    const pos = this._position;
    if (!pos) return null;
    const { HOSTILE_MOBS } = require('./categories');
    let best = null;
    let bestDist = Infinity;
    for (const entity of this._entities.values()) {
      if (!entity.position) continue;
      const name = (entity.name || '').toLowerCase();
      if (!HOSTILE_MOBS.has(name) && entity.type !== 'player') continue;
      const dx = pos.x - entity.position.x;
      const dy = pos.y - entity.position.y;
      const dz = pos.z - entity.position.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < bestDist) {
        bestDist = dist;
        best = entity;
      }
    }
    return best;
  }

  isSelf(entity) {
    return entity.id === this._entityId;
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

  setControlState(key, val) {
    this._controlStates[key] = val;
    // Keyboard-driven: press/release real keys so the client moves
    if (this._keyboardFallback && this._botControlActive) {
      this._keyboardFallback.setControlState(key, val).catch(() => {});
    }
  }

  clearControlStates() {
    for (const key of Object.keys(this._controlStates)) {
      this._controlStates[key] = false;
    }
    if (this._keyboardFallback && this._botControlActive) {
      this._keyboardFallback.clearControlStates().catch(() => {});
    }
  }

  async attack(entity) {
    if (!this._upstream) {
      if (this._keyboardFallback) {
        console.warn('[java-relay] No upstream, keyboard fallback for attack.');
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.attack(); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
    try {
      this._upstream.write('use_entity', {
        target: entity.id,
        mouse: 1,  // 1 = attack
        sneaking: this._controlStates.sneak,
      });
      // Also swing arm
      this._upstream.write('arm_animation', { hand: 0 });
    } catch (e) {
      console.error('[java-relay] Attack failed:', e.message);
      if (this._keyboardFallback) {
        console.warn('[java-relay] Packet injection failed, keyboard fallback for attack.');
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.attack(); }
        finally { this._kbFallbackActive = false; }
      }
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
    // Native mouse delta for visual immersion (best-effort, non-blocking)
    if (this._nativeMouseDelta && this._botControlActive) {
      const deltaYawRad = (yaw - this._yaw) * (Math.PI / 180);
      const deltaPitchRad = (pitch - this._pitch) * (Math.PI / 180);
      this._nativeMouseDelta.move(deltaYawRad, deltaPitchRad);
    }
    // Packet rewriting (_rewriteServerbound) handles server-side aim.
    this._yaw = yaw;
    this._pitch = pitch;
  }

  async swingArm() {
    if (!this._upstream) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.swingArm(); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
    try {
      this._upstream.write('arm_animation', { hand: 0 });
    } catch (e) {
      if (this._keyboardFallback) {
        console.warn('[java-relay] Packet injection failed, keyboard fallback for swingArm.');
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.swingArm(); }
        finally { this._kbFallbackActive = false; }
      }
    }
  }

  async activateItem() {
    if (!this._upstream) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.activateItem(); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
    try {
      this._upstream.write('use_item', {
        hand: 0,
        sequence: 0,
      });
    } catch (e) {
      if (this._keyboardFallback) {
        console.warn('[java-relay] Packet injection failed, keyboard fallback for activateItem.');
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.activateItem(); }
        finally { this._kbFallbackActive = false; }
      }
    }
  }

  async pressUseItem() {
    this._isUsingItem = true;
    if (!this._upstream) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.pressUseItem(); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
    try {
      this._upstream.write('use_item', {
        hand: 0,
        sequence: 0,
      });
    } catch (e) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.pressUseItem(); }
        finally { this._kbFallbackActive = false; }
      }
    }
  }

  async releaseUseItem() {
    this._isUsingItem = false;
    this._lastItemReleaseTime = Date.now();
    if (!this._upstream) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.releaseUseItem(); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
    try {
      this._upstream.write('block_dig', {
        status: 5,  // release_use_item
        location: { x: 0, y: 0, z: 0 },
        face: 0,
        sequence: 0,
      });
    } catch (e) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.releaseUseItem(); }
        finally { this._kbFallbackActive = false; }
      }
    }
  }

  async setQuickBarSlot(slot) {
    this._quickBarSlot = slot;
    if (!this._upstream) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.setQuickBarSlot(slot); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
    try {
      this._upstream.write('held_item_slot', { slotId: slot });
    } catch (e) {
      if (this._keyboardFallback) {
        console.warn('[java-relay] Packet injection failed, keyboard fallback for setQuickBarSlot.');
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.setQuickBarSlot(slot); }
        finally { this._kbFallbackActive = false; }
      }
    }
  }

  chat(msg) {
    if (!this._upstream) return;
    try {
      this._upstream.write('chat', { message: msg });
    } catch (_) {}
  }

  respawn() {
    if (!this._upstream) return;
    try {
      this._upstream.write('client_command', { actionId: 0 });
    } catch (_) {}
  }

  async craftPlanks() {
    if (!this._javaRecipes || !this._upstream) {
      console.log('[java-relay] Cannot craft: recipes not loaded or no upstream.');
      return;
    }

    // Find a planks recipe where we have the ingredients (any log type)
    const plankNames = ['oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks',
                        'acacia_planks', 'dark_oak_planks', 'mangrove_planks',
                        'cherry_planks', 'crimson_planks', 'warped_planks'];

    for (const name of plankNames) {
      const recipe = this._javaRecipes.get(name);
      if (!recipe) continue;

      // Check if we have all ingredients
      const have = recipe.ingredients.every(ing => this._hasIngredient(ing));
      if (!have) continue;

      try {
        this._upstream.write('craft_recipe_request', {
          windowId: 0,
          recipe: recipe.recipeId,
          makeAll: false,
        });
        console.log(`[java-relay] Crafting ${name} (recipe: ${recipe.recipeId})`);

        // Pick up result from crafting output slot (slot 0 of inventory window)
        await sleep(100);
        this._upstream.write('window_click', {
          windowId: 0,
          slot: 0,
          mouseButton: 0,
          action: this._windowActionId++,
          mode: 1,  // shift-click to put result in inventory
          item: { present: false },
        });
        return;
      } catch (e) {
        console.error(`[java-relay] craftPlanks error: ${e.message}`);
      }
    }

    console.log('[java-relay] No planks recipe available (missing log/wood).');
  }

  async craftToolOrSticks() {
    if (!this._javaRecipes || !this._upstream) {
      console.log('[java-relay] Cannot craft: recipes not loaded or no upstream.');
      return;
    }

    // Try sticks first (2x2 shapeless/shaped, only needs planks)
    const sticksRecipe = this._javaRecipes.get('stick');
    if (sticksRecipe) {
      const have = sticksRecipe.ingredients.every(ing => this._hasIngredient(ing));
      if (have) {
        try {
          this._upstream.write('craft_recipe_request', {
            windowId: 0,
            recipe: sticksRecipe.recipeId,
            makeAll: false,
          });
          console.log(`[java-relay] Crafting sticks (recipe: ${sticksRecipe.recipeId})`);
          await sleep(100);
          this._upstream.write('window_click', {
            windowId: 0,
            slot: 0,
            mouseButton: 0,
            action: this._windowActionId++,
            mode: 1,
            item: { present: false },
          });
          return;
        } catch (e) {
          console.error(`[java-relay] craftToolOrSticks error: ${e.message}`);
        }
      }
    }

    // Try wooden tools (pickaxe preferred, then sword, then axe)
    // Note: these need 3x3 crafting table. Player inventory is only 2x2.
    const toolNames = ['wooden_pickaxe', 'wooden_sword', 'wooden_axe'];
    for (const name of toolNames) {
      const recipe = this._javaRecipes.get(name);
      if (!recipe) continue;
      if (recipe.width > 2 || recipe.height > 2) {
        console.log(`[java-relay] ${name} requires crafting table (${recipe.width}x${recipe.height}), skipping.`);
        continue;
      }
      const have = recipe.ingredients.every(ing => this._hasIngredient(ing));
      if (!have) continue;

      try {
        this._upstream.write('craft_recipe_request', {
          windowId: 0,
          recipe: recipe.recipeId,
          makeAll: false,
        });
        console.log(`[java-relay] Crafting ${name} (recipe: ${recipe.recipeId})`);
        await sleep(100);
        this._upstream.write('window_click', {
          windowId: 0,
          slot: 0,
          mouseButton: 0,
          action: this._windowActionId++,
          mode: 1,
          item: { present: false },
        });
        return;
      } catch (e) {
        console.error(`[java-relay] craftToolOrSticks error: ${e.message}`);
      }
    }

    console.log('[java-relay] No tool/stick recipes available with current inventory.');
  }

  disconnect() {
    this._stopMovementLoop();
    this._ready = false;
    if (this._nativeMouseDelta) {
      this._nativeMouseDelta.stop();
    }
    if (this._upstream) {
      this._upstream.end();
      this._upstream = null;
    }
    if (this._server) {
      this._server.close();
    }
  }
}

module.exports = { JavaRelayAdapter };
