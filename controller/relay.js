/**
 * RelayAdapter — transparent MITM proxy using bedrock-protocol Relay.
 *
 * Sits between a real Minecraft Bedrock client and the server/Realm.
 * Reads ALL game state from clientbound packets (same data quality as direct bot).
 * Injects serverbound packets for actions when FPI agent is in control.
 *
 * The real client handles rendering, anti-cheat, Xbox auth, chunk loading.
 * We just observe and occasionally steer.
 *
 * Architecture:
 *   Real MC Client → localhost:19132 → Relay → Bedrock Realm
 *                                         ↕
 *                                   Reads all packets
 *                                   Injects actions
 */

const bedrock = require('bedrock-protocol');
const { sleep, waitTicks } = require('./utils');
const { BlockCache } = require('./block-cache');

class RelayAdapter {
  constructor(config) {
    this.config = config;
    this._relay = null;
    this._player = null;  // The connected player (real client)
    this._ready = false;

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
    this._runtimeEntityId = null;

    // Entity registry (from add_entity/add_player/move_entity packets)
    this._entities = new Map();

    // Inventory (from inventory_content packets)
    this._inventory = [];

    // Block cache (parses level_chunk / update_block for blockAt())
    this._blockCache = new BlockCache();

    // --- Bot control state ---
    // When true, FPI agent controls the character.
    // When false, real client's inputs pass through unmodified.
    this._botControlActive = false;

    // Simulated control states (used when bot is active)
    this._controlStates = {
      forward: false, back: false, left: false, right: false,
      jump: false, sprint: false, sneak: false,
    };

    // Movement injection interval
    this._movementInterval = null;

    // Biome (estimated)
    this._biome = 'plains';
  }

  /**
   * Start the Relay proxy.
   * - Listens on config.listenPort (default 19132) for the real client
   * - Forwards to config.destination (Realm or server)
   */
  async start(trackingState) {
    return new Promise((resolve, reject) => {
      const relayOpts = {
        host: '0.0.0.0',
        port: this.config.listenPort || 19132,
        destination: this._buildDestination(),
        // Let bedrock-protocol handle Xbox Live auth
        profilesFolder: this.config.authCache || './auth_cache',
      };

      // Logging
      if (this.config.logPackets) {
        relayOpts.logging = true;
      }

      this._relay = new bedrock.Relay(relayOpts);

      this._relay.on('connect', (player) => {
        console.log('[relay] Real client connected, waiting for upstream...');
      });

      this._relay.on('join', (player) => {
        console.log('[relay] Upstream connected — relay fully active.');
        this._player = player;
        this._setupPacketListeners(player, trackingState);
        this._ready = true;
        this._startMovementLoop();
        resolve();
      });

      this._relay.on('error', (err) => {
        console.error('[relay] Error:', err.message);
        if (!this._ready) reject(err);
      });

      this._relay.listen();
      console.log(`[relay] Listening on port ${relayOpts.port}`);
      console.log(`[relay] Connect your Minecraft client to localhost:${relayOpts.port}`);
    });
  }

  _buildDestination() {
    const dest = {};

    if (this.config.realmInvite) {
      // Connect to a Realm via invite link
      dest.realms = { realmInvite: this.config.realmInvite };
    } else if (this.config.realmId) {
      dest.realms = { realmId: this.config.realmId };
    } else {
      // Direct server
      dest.host = this.config.serverHost || 'localhost';
      dest.port = this.config.serverPort || 19132;
    }

    return dest;
  }

  /**
   * Set up listeners on ALL packets flowing through the relay.
   */
  _setupPacketListeners(player, trackingState) {
    // --- CLIENTBOUND: server → client (we read game state) ---
    player.on('clientbound', ({ name, params }) => {
      this._handleClientbound(name, params, trackingState);
    });

    // --- SERVERBOUND: client → server (we can suppress/modify) ---
    player.on('serverbound', ({ name, params }, descriptor) => {
      this._handleServerbound(name, params, descriptor, trackingState);
    });

    player.on('close', () => {
      console.log('[relay] Real client disconnected.');
      this._ready = false;
      this._player = null;
      this._stopMovementLoop();
    });
  }

  /**
   * Process clientbound packets (server → client).
   * Extract game state without modifying anything.
   */
  _handleClientbound(name, params, trackingState) {
    switch (name) {
      case 'start_game':
        this._runtimeEntityId = params.runtime_entity_id;
        if (params.player_position) {
          this._updatePosition(params.player_position);
        }
        this._blockCache.handleStartGame(params);
        console.log(`[relay] Game started. Runtime ID: ${this._runtimeEntityId}`);
        break;

      case 'update_attributes':
        if (params.runtime_entity_id !== this._runtimeEntityId) break;
        for (const attr of params.attributes || []) {
          switch (attr.name) {
            case 'minecraft:health': {
              const oldHealth = this._health;
              this._health = attr.current;
              if (this._health <= 0 && oldHealth > 0) {
                console.log('[relay] Player died.');
                trackingState.pendingRespawn = true;
              }
              break;
            }
            case 'minecraft:player.hunger':
              this._food = attr.current;
              break;
            case 'minecraft:player.saturation':
              this._foodSaturation = attr.current;
              break;
            case 'minecraft:player.level':
              this._xpLevel = Math.floor(attr.current);
              break;
            case 'minecraft:player.experience':
              this._xpPoints = Math.floor(attr.current);
              break;
          }
        }
        break;

      case 'move_player':
        if (params.runtime_id === this._runtimeEntityId) {
          this._updatePosition(params.position);
          this._yaw = params.yaw || 0;
          this._pitch = params.pitch || 0;
          this._onGround = params.on_ground !== false;
        }
        break;

      case 'correct_player_move_prediction':
        if (params.position) {
          this._updatePosition(params.position);
        }
        break;

      case 'add_entity':
        this._entities.set(params.runtime_id, {
          id: params.runtime_id,
          runtimeId: params.runtime_id,
          type: params.entity_type || 'unknown',
          name: (params.entity_type || '').replace('minecraft:', ''),
          displayName: params.entity_type,
          position: params.position ? { ...params.position } : null,
          height: 1.8,
          username: null,
          yaw: params.yaw || 0,
        });
        break;

      case 'add_player':
        this._entities.set(params.runtime_id, {
          id: params.runtime_id,
          runtimeId: params.runtime_id,
          type: 'player',
          name: params.username || 'player',
          displayName: params.username,
          username: params.username,
          position: params.position ? { ...params.position } : null,
          height: 1.8,
          yaw: params.yaw || 0,
        });
        break;

      case 'move_entity_delta':
      case 'move_entity': {
        const entity = this._entities.get(params.runtime_entity_id);
        if (entity && params.position) {
          entity.position = { ...params.position };
          if (params.yaw !== undefined) entity.yaw = params.yaw;
        }
        break;
      }

      case 'remove_entity': {
        const rid = params.entity_id_self;
        if (trackingState.attackedEntities.has(rid)) {
          trackingState.killsSinceLastState++;
          trackingState.attackedEntities.delete(rid);
        }
        this._entities.delete(rid);
        break;
      }

      case 'set_time':
        this._timeOfDay = params.time || 0;
        break;

      case 'inventory_content':
        if (params.window_id === 0) {
          this._inventory = (params.input || []).filter(item =>
            item && item.network_id !== 0
          ).map(item => ({
            name: (item.metadata?.name || `item_${item.network_id}`).replace('minecraft:', ''),
            count: item.count || 1,
          }));
        }
        break;

      case 'respawn':
        this._health = 20;
        this._food = 20;
        trackingState.pendingRespawn = false;
        if (params.player_position) {
          this._updatePosition(params.player_position);
        }
        console.log('[relay] Player respawned.');
        break;

      case 'entity_event':
        if (params.runtime_entity_id === this._runtimeEntityId && params.event_id === 2) {
          // Player took damage — knockback cooldown
          trackingState.knockbackCooldown = 2;
          if (this._botControlActive) {
            this.clearControlStates();
          }
        }
        break;

      case 'level_chunk':
        this._blockCache.handleLevelChunk(params);
        this._blockCache.prune(this._position);
        break;

      case 'sub_chunk':
        this._blockCache.handleSubChunk(params);
        break;

      case 'update_block':
        this._blockCache.handleUpdateBlock(params);
        break;

      case 'player_hotbar':
        if (params.selected_hotbar_slot !== undefined) {
          this._quickBarSlot = params.selected_hotbar_slot;
        }
        break;
    }
  }

  /**
   * Process serverbound packets (client → server).
   * When bot is in control, suppress the real client's movement/action packets
   * so they don't conflict with injected actions.
   */
  _handleServerbound(name, params, descriptor, trackingState) {
    if (!this._botControlActive) return; // Pass-through when user is in control

    // Suppress real client movement when bot is driving
    const suppressedPackets = new Set([
      'player_auth_input',
      'move_player',
      'player_action',
      'animate',
      'inventory_transaction',
    ]);

    if (suppressedPackets.has(name)) {
      descriptor.canceled = true;
    }
  }

  _updatePosition(pos) {
    if (!pos) return;
    this._position = {
      x: pos.x || 0,
      y: pos.y || 0,
      z: pos.z || 0,
    };
  }

  // ---- Bot control: enable/disable FPI agent control ----

  /**
   * Enable FPI agent control. Real client's movement is suppressed.
   * The agent can now inject actions via the adapter interface.
   */
  enableBotControl() {
    this._botControlActive = true;
    console.log('[relay] Bot control ENABLED — FPI agent driving.');
  }

  /**
   * Disable FPI agent control. Real client resumes normal play.
   * FPI agent still observes state but doesn't inject actions.
   */
  disableBotControl() {
    this._botControlActive = false;
    this.clearControlStates();
    console.log('[relay] Bot control DISABLED — user in control.');
  }

  get botControlActive() {
    return this._botControlActive;
  }

  // ---- Movement injection (when bot is in control) ----

  _startMovementLoop() {
    this._movementInterval = setInterval(() => {
      if (!this._player || !this._ready || !this._botControlActive) return;

      let inputFlags = 0;
      if (this._controlStates.forward) inputFlags |= (1 << 0);
      if (this._controlStates.back) inputFlags |= (1 << 1);
      if (this._controlStates.left) inputFlags |= (1 << 2);
      if (this._controlStates.right) inputFlags |= (1 << 3);
      if (this._controlStates.jump) inputFlags |= (1 << 4);
      if (this._controlStates.sneak) inputFlags |= (1 << 5);
      if (this._controlStates.sprint) inputFlags |= (1 << 6);

      if (inputFlags === 0) return;

      try {
        // Inject movement to the SERVER (upstream) as if the real client sent it
        this._player.upstream.queue('player_auth_input', {
          pitch: this._pitch,
          yaw: this._yaw,
          position: { x: this._position.x, y: this._position.y, z: this._position.z },
          move_vector: {
            x: (this._controlStates.right ? 1 : 0) - (this._controlStates.left ? 1 : 0),
            z: (this._controlStates.forward ? 1 : 0) - (this._controlStates.back ? 1 : 0),
          },
          head_yaw: this._yaw,
          input_data: inputFlags,
          input_mode: 1,
          play_mode: 0,
          tick: BigInt(Date.now()),
        });
      } catch (_) {}
    }, 50); // 20 TPS
  }

  _stopMovementLoop() {
    if (this._movementInterval) {
      clearInterval(this._movementInterval);
      this._movementInterval = null;
    }
  }

  // ---- Adapter interface (same as BedrockBot / JavaBot) ----
  // This lets all existing code (state.js, actions.js, bridge.js) work unchanged.

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
  get isRaining() { return false; }
  get timeOfDay() { return this._timeOfDay; }
  get xpLevel() { return this._xpLevel; }
  get xpPoints() { return this._xpPoints; }
  get quickBarSlot() { return this._quickBarSlot; }
  get allEntities() { return Array.from(this._entities.values()); }
  get inventoryItems() { return this._inventory; }

  // Mark as Bedrock for protocol-aware code
  get bedrockClient() { return this._player?.upstream || null; }

  isSelf(entity) {
    return entity.runtimeId === this._runtimeEntityId;
  }

  lightAt(_pos) {
    if (this._timeOfDay >= 12500 && this._timeOfDay <= 23500) return 4;
    return 15;
  }

  blockAt(pos) {
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
  }

  clearControlStates() {
    for (const key of Object.keys(this._controlStates)) {
      this._controlStates[key] = false;
    }
  }

  async attack(entity) {
    if (!this._player) return;
    try {
      // Inject attack packet upstream to server
      this._player.upstream.queue('inventory_transaction', {
        transaction: {
          legacy: { type: 'none' },
          transaction_type: 'item_use_on_entity',
          actions: [],
          transaction_data: {
            entity_runtime_id: entity.runtimeId,
            action_type: 1,
            hotbar_slot: this._quickBarSlot,
            held_item: { network_id: 0 },
            player_pos: { x: this._position.x, y: this._position.y, z: this._position.z },
            click_pos: entity.position || { x: 0, y: 0, z: 0 },
          },
        },
      });
    } catch (e) {
      console.error('[relay] Attack failed:', e.message);
    }
  }

  async lookAt(pos) {
    const dx = pos.x - this._position.x;
    const dy = pos.y - this._position.y;
    const dz = pos.z - this._position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    this._yaw = -Math.atan2(dx, dz) * (180 / Math.PI);
    this._pitch = -Math.atan2(dy, dist) * (180 / Math.PI);
  }

  async look(yaw, pitch) {
    this._yaw = yaw;
    this._pitch = pitch;
  }

  swingArm() {
    if (!this._player) return;
    try {
      this._player.upstream.queue('animate', {
        action_id: 1,
        runtime_entity_id: this._runtimeEntityId,
      });
    } catch (_) {}
  }

  async activateItem() {
    if (!this._player) return;
    try {
      this._player.upstream.queue('inventory_transaction', {
        transaction: {
          legacy: { type: 'none' },
          transaction_type: 'item_use',
          actions: [],
          transaction_data: {
            action_type: 1,
            block_position: { x: 0, y: 0, z: 0 },
            face: -1,
            hotbar_slot: this._quickBarSlot,
            held_item: { network_id: 0 },
            player_pos: { x: this._position.x, y: this._position.y, z: this._position.z },
            click_pos: { x: 0, y: 0, z: 0 },
            block_runtime_id: 0,
          },
        },
      });
    } catch (_) {}
  }

  setQuickBarSlot(slot) {
    this._quickBarSlot = slot;
    if (!this._player) return;
    try {
      this._player.upstream.queue('player_hotbar', {
        selected_hotbar_slot: slot,
        window_id: 0,
        select_hotbar_slot: true,
      });
    } catch (_) {}
  }

  chat(msg) {
    if (!this._player) return;
    try {
      this._player.upstream.queue('text', {
        type: 'chat',
        needs_translation: false,
        source_name: '',
        message: msg,
        xuid: '',
        platform_chat_id: '',
      });
    } catch (_) {}
  }

  async craftPlanks() {
    console.log('[relay] Crafting not yet implemented.');
  }

  async craftToolOrSticks() {
    console.log('[relay] Crafting not yet implemented.');
  }

  disconnect() {
    this._stopMovementLoop();
    if (this._relay) {
      // Close gracefully
      this._ready = false;
    }
  }
}

module.exports = { RelayAdapter };
