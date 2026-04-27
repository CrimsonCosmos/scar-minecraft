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
const { HumanTiming } = require('./stealth');
const { BEDROCK_ENTITY_META } = require('./categories');

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

    // Keyboard fallback (null unless --keyboard-fallback enabled)
    this._keyboardFallback = null;
    // True during keyboard fallback execution — suppression is lifted
    this._kbFallbackActive = false;

    // Simulated control states (used when bot is active)
    this._controlStates = {
      forward: false, back: false, left: false, right: false,
      jump: false, sprint: false, sneak: false,
    };

    // Movement injection interval
    this._movementInterval = null;

    // Active status effects (effect ID → {amplifier, duration, startTime})
    this._activeEffects = new Map();

    // Weather state
    this._isRaining = false;
    this._isThundering = false;

    // Biome (estimated)
    this._biome = 'plains';

    // Human timing generator for movement tick jitter
    this._humanTiming = new HumanTiming();

    // Bot control transition state
    this._transitionStart = 0;
    this._transitionDurationMs = 0;

    // Crafting recipes (from crafting_data packet)
    // Map of recipe output name → { network_id, inputs: [{name, count}], outputs: [{name, count, network_id}] }
    this._recipes = new Map();

    // Auto-incrementing request ID for item_stack_request
    this._nextRequestId = 1;
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
          velocity: { x: 0, y: 0, z: 0 },
          _prevPos: params.position ? { ...params.position } : null,
          _lastMoveTime: Date.now(),
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
          velocity: { x: 0, y: 0, z: 0 },
          _prevPos: params.position ? { ...params.position } : null,
          _lastMoveTime: Date.now(),
        });
        break;

      case 'move_entity_delta':
      case 'move_entity': {
        const entity = this._entities.get(params.runtime_entity_id);
        if (entity && params.position) {
          const now = Date.now();
          const dt = (now - (entity._lastMoveTime || now)) / 1000;
          if (entity._prevPos && dt >= 0.01 && dt < 5.0) {
            entity.velocity = {
              x: (params.position.x - entity._prevPos.x) / dt,
              y: (params.position.y - entity._prevPos.y) / dt,
              z: (params.position.z - entity._prevPos.z) / dt,
            };
          }
          entity._prevPos = { ...params.position };
          entity._lastMoveTime = now;
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
          this._inventory = [];
          const items = params.input || [];
          for (let slot = 0; slot < items.length; slot++) {
            const item = items[slot];
            if (!item || item.network_id === 0) continue;
            this._inventory.push({
              name: (item.metadata?.name || `item_${item.network_id}`).replace('minecraft:', ''),
              count: item.count || 1,
              slot,
              network_id: item.network_id,
              stack_id: item.stack_id || 0,
              metadata: item.metadata || {},
            });
          }
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
        if (params.event_id === 2) {
          const rid = params.runtime_entity_id;
          if (rid === this._runtimeEntityId) {
            // Player took damage — knockback cooldown
            trackingState.knockbackCooldown = 2;
            if (this._botControlActive) {
              this.clearControlStates();
            }
          } else if (this._lastItemReleaseTime &&
                     Date.now() - this._lastItemReleaseTime < 3000 &&
                     !trackingState.attackedEntities.has(rid)) {
            // Non-self entity hurt within 3s of our projectile release — attribute as projectile hit
            trackingState.projectileHitLanded = true;
            trackingState.attackedEntities.add(rid);
            const entity = this._entities.get(rid);
            if (entity && entity.type === 'player') {
              trackingState.projectilePlayerHitLanded = true;
            }
          }
        }
        break;

      case 'set_entity_data': {
        const rid = params.runtime_entity_id;
        const entity = this._entities.get(rid);
        if (entity) {
          for (const entry of (params.metadata || [])) {
            switch (entry.key) {
              case BEDROCK_ENTITY_META.FLAGS: {
                const flags = BigInt(entry.value || 0);
                entity._flags = Number(flags & 0xFFn);
                entity._isBaby = !!(flags & (1n << 22n));
                break;
              }
              case BEDROCK_ENTITY_META.HEALTH:
                if (typeof entry.value === 'number') entity._health = entry.value;
                break;
              case BEDROCK_ENTITY_META.FUSE_LENGTH:
                if (typeof entry.value === 'number') entity._creeperState = entry.value;
                break;
            }
          }
        }
        break;
      }

      case 'crafting_data':
        this._parseCraftingData(params);
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

      case 'mob_effect': {
        if (params.runtime_entity_id !== this._runtimeEntityId) break;
        const eventId = params.event_id; // 1=add, 2=modify, 3=remove
        if (eventId === 1 || eventId === 2) {
          this._activeEffects.set(params.effect_id, {
            amplifier: params.amplifier || 0,
            duration: params.duration || 0,
            startTime: Date.now(),
          });
        } else if (eventId === 3) {
          this._activeEffects.delete(params.effect_id);
        }
        break;
      }

      case 'level_event':
        // Bedrock weather events: 3001 = start rain, 3003 = stop rain, 3002 = start thunder, 3004 = stop thunder
        if (params.event === 3001) this._isRaining = true;
        else if (params.event === 3003) this._isRaining = false;
        else if (params.event === 3002) this._isThundering = true;
        else if (params.event === 3004) this._isThundering = false;
        break;
    }
  }

  /**
   * Process serverbound packets (client → server).
   * When bot is in control, suppress the real client's movement/action packets
   * so they don't conflict with injected actions.
   */
  _handleServerbound(name, params, descriptor, trackingState) {
    if (!this._botControlActive || this._kbFallbackActive) return; // Pass-through when user is in control or keyboard fallback active

    // Suppress real client movement when bot is driving
    const suppressedPackets = new Set([
      'player_auth_input',
      'move_player',
      'player_action',
      'animate',
      'inventory_transaction',
    ]);

    if (suppressedPackets.has(name)) {
      // During bot control transition, gradually increase suppression
      // so the packet pattern change isn't instant
      if (this._transitionStart > 0) {
        const elapsed = Date.now() - this._transitionStart;
        if (elapsed < this._transitionDurationMs) {
          const progress = elapsed / this._transitionDurationMs;
          // Smoothstep curve for natural ramp
          const t = progress * progress * (3 - 2 * progress);
          if (Math.random() > t) return; // Let this packet through
        } else {
          this._transitionStart = 0; // Transition complete
        }
      }
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
    // Gradual transition: ramp suppression over 1.5-2.5s
    this._transitionStart = Date.now();
    this._transitionDurationMs = 1500 + Math.random() * 1000;
    console.log('[relay] Bot control ENABLED — ramping up.');
  }

  /**
   * Disable FPI agent control. Real client resumes normal play.
   * FPI agent still observes state but doesn't inject actions.
   */
  disableBotControl() {
    this._botControlActive = false;
    this._transitionStart = 0;
    this.clearControlStates();
    console.log('[relay] Bot control DISABLED — user in control.');
  }

  get botControlActive() {
    return this._botControlActive;
  }

  /**
   * Set a KeyboardFallback instance for OS-level input when packet injection fails.
   */
  setKeyboardFallback(fb) {
    this._keyboardFallback = fb;
  }

  // ---- Movement injection (when bot is in control) ----

  _startMovementLoop() {
    this._movementLoopActive = true;

    const tick = () => {
      if (!this._movementLoopActive) return;

      if (this._player && this._ready && this._botControlActive) {
        let inputFlags = 0;
        if (this._controlStates.forward) inputFlags |= (1 << 0);
        if (this._controlStates.back) inputFlags |= (1 << 1);
        if (this._controlStates.left) inputFlags |= (1 << 2);
        if (this._controlStates.right) inputFlags |= (1 << 3);
        if (this._controlStates.jump) inputFlags |= (1 << 4);
        if (this._controlStates.sneak) inputFlags |= (1 << 5);
        if (this._controlStates.sprint) inputFlags |= (1 << 6);

        if (inputFlags !== 0) {
          try {
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
          } catch (e) {
            if (this._keyboardFallback) {
              console.warn('[relay] Movement packet failed, keyboard fallback for tick.');
              this._kbFallbackActive = true;
              const fb = this._keyboardFallback;
              const states = { ...this._controlStates };
              Promise.resolve().then(async () => {
                try {
                  for (const [key, val] of Object.entries(states)) {
                    await fb.setControlState(key, val);
                  }
                } finally {
                  this._kbFallbackActive = false;
                }
              });
            }
          }
        }
      }

      // Variable tick interval: 38-65ms instead of fixed 50ms
      const nextMs = this._humanTiming.nextMovementTick();
      this._movementTimeout = setTimeout(tick, nextMs);
    };

    tick();
  }

  _stopMovementLoop() {
    this._movementLoopActive = false;
    if (this._movementTimeout) {
      clearTimeout(this._movementTimeout);
      this._movementTimeout = null;
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
  get isRaining() { return this._isRaining; }
  get isThundering() { return this._isThundering; }
  get activeEffects() { return this._activeEffects; }
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
    if (!this._player) {
      if (this._keyboardFallback) {
        console.warn('[relay] No player connection, keyboard fallback for attack.');
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.attack(); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
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
      if (this._keyboardFallback) {
        console.warn('[relay] Packet injection failed, keyboard fallback for attack.');
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
    this._yaw = -Math.atan2(dx, dz) * (180 / Math.PI);
    this._pitch = -Math.atan2(dy, dist) * (180 / Math.PI);
  }

  async look(yaw, pitch) {
    this._yaw = yaw;
    this._pitch = pitch;
  }

  async swingArm() {
    if (!this._player) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.swingArm(); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
    try {
      this._player.upstream.queue('animate', {
        action_id: 1,
        runtime_entity_id: this._runtimeEntityId,
      });
    } catch (e) {
      if (this._keyboardFallback) {
        console.warn('[relay] Packet injection failed, keyboard fallback for swingArm.');
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.swingArm(); }
        finally { this._kbFallbackActive = false; }
      }
    }
  }

  async activateItem() {
    if (!this._player) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.activateItem(); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
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
    } catch (e) {
      if (this._keyboardFallback) {
        console.warn('[relay] Packet injection failed, keyboard fallback for activateItem.');
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.activateItem(); }
        finally { this._kbFallbackActive = false; }
      }
    }
  }

  async pressUseItem() {
    this._isUsingItem = true;
    if (!this._player) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.pressUseItem(); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
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
    if (!this._player) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.releaseUseItem(); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
    try {
      this._player.upstream.queue('player_action', {
        runtime_entity_id: this._runtimeEntityId,
        action: 'stop_item_use',
        position: { x: 0, y: 0, z: 0 },
        result_position: { x: 0, y: 0, z: 0 },
        face: 0,
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
    if (!this._player) {
      if (this._keyboardFallback) {
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.setQuickBarSlot(slot); }
        finally { this._kbFallbackActive = false; }
      }
      return;
    }
    try {
      // Find the item currently in the target slot
      const slotItem = this._inventory.find(i => i.slot === slot);
      this._player.upstream.queue('mob_equipment', {
        runtime_entity_id: this._runtimeEntityId,
        item: slotItem
          ? { network_id: slotItem.network_id, count: slotItem.count, metadata: 0,
              has_stack_id: 1, stack_id: slotItem.stack_id, block_runtime_id: 0, extra: { has_nbt: 0, can_place_on: [], can_destroy: [] } }
          : { network_id: 0 },
        slot: slot,
        selected_slot: slot,
        window_id: 0,
      });
    } catch (e) {
      if (this._keyboardFallback) {
        console.warn('[relay] Packet injection failed, keyboard fallback for setQuickBarSlot.');
        this._kbFallbackActive = true;
        try { await this._keyboardFallback.setQuickBarSlot(slot); }
        finally { this._kbFallbackActive = false; }
      }
    }
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

  respawn() {
    if (!this._player) return;
    try {
      this._player.upstream.queue('respawn', {
        state: 2,
        runtime_entity_id: this._runtimeEntityId || 0n,
      });
    } catch (_) {}
  }

  // ---- Crafting ----

  _parseCraftingData(params) {
    this._recipes.clear();
    const recipes = params.recipes || [];
    for (const entry of recipes) {
      const recipe = entry.recipe;
      if (!recipe) continue;
      // Only handle shapeless and shaped recipes
      const type = entry.type;
      if (type !== 'shapeless' && type !== 'shaped' && type !== 0 && type !== 1) continue;
      const outputs = recipe.output || [];
      const networkId = recipe.network_id;
      if (!networkId || outputs.length === 0) continue;

      const inputs = (recipe.input || []).map(i => ({
        name: (i.name || i.network_id_or_tag || '').toString().replace('minecraft:', ''),
        count: i.count || 1,
        network_id: i.network_id || i.network_id_or_tag || 0,
      }));

      for (const out of outputs) {
        const outName = (out.name || `item_${out.network_id}`).replace('minecraft:', '');
        if (!this._recipes.has(outName)) {
          this._recipes.set(outName, []);
        }
        this._recipes.get(outName).push({
          network_id: networkId,
          uuid: recipe.uuid,
          inputs,
          output_name: outName,
          output_count: out.count || 1,
          output_network_id: out.network_id || 0,
          block: recipe.block || '',
        });
      }
    }
    console.log(`[relay] Parsed ${this._recipes.size} unique craftable items from crafting_data.`);
  }

  /**
   * Find a recipe that we can craft with current inventory.
   * Returns { recipe, matchedInputs } or null.
   */
  _findCraftableRecipe(outputPattern, requireBlock) {
    // Search recipes whose output name contains the pattern
    for (const [outName, recipes] of this._recipes) {
      if (!outName.includes(outputPattern)) continue;
      for (const recipe of recipes) {
        // If we need a crafting table, skip 2x2 recipes
        if (requireBlock && recipe.block !== 'crafting_table') continue;
        // Check if we have all inputs in inventory
        const invCopy = this._inventory.map(i => ({ ...i }));
        let canCraft = true;
        const matchedInputs = [];
        for (const input of recipe.inputs) {
          const idx = invCopy.findIndex(i =>
            i.name.includes(input.name) && i.count >= input.count
          );
          if (idx < 0) { canCraft = false; break; }
          matchedInputs.push({ ...invCopy[idx], needed: input.count });
          invCopy[idx].count -= input.count;
        }
        if (canCraft) return { recipe, matchedInputs };
      }
    }
    return null;
  }

  /**
   * Send an item_stack_request to craft a recipe.
   */
  async _sendCraftRequest(recipe, matchedInputs) {
    if (!this._player) return false;
    const requestId = this._nextRequestId++;

    // Build the ingredient list for craft_recipe_auto
    const ingredients = matchedInputs.map(i => ({
      network_id: i.network_id,
      count: i.needed,
      metadata: 0,
    }));

    try {
      this._player.upstream.queue('item_stack_request', {
        requests: [{
          request_id: requestId,
          actions: [
            {
              type_id: 'craft_recipe_auto',
              recipe_network_id: recipe.network_id,
              times_crafted: 1,
              times_crafted_2: 1,
              ingredients,
            },
            {
              type_id: 'results_deprecated',
              result_items: [{
                network_id: recipe.output_network_id,
                count: recipe.output_count,
                metadata: 0,
                has_stack_id: 0,
                stack_id: 0,
                block_runtime_id: 0,
                extra: { has_nbt: 0, can_place_on: [], can_destroy: [] },
              }],
              times_crafted: 1,
            },
          ],
        }],
      });
      console.log(`[relay] Sent craft request for ${recipe.output_name} (recipe ${recipe.network_id}).`);
      // Give server time to process and send updated inventory
      await sleep(500);
      return true;
    } catch (e) {
      console.error('[relay] Craft request failed:', e.message);
      return false;
    }
  }

  async craftPlanks() {
    const result = this._findCraftableRecipe('planks', false);
    if (!result) {
      console.log('[relay] Cannot craft planks — no logs in inventory or recipe not found.');
      return;
    }
    await this._sendCraftRequest(result.recipe, result.matchedInputs);
  }

  async craftToolOrSticks() {
    // Try wooden pickaxe first, then sticks
    let result = this._findCraftableRecipe('pickaxe', false);
    if (!result) result = this._findCraftableRecipe('stick', false);
    if (!result) {
      console.log('[relay] Cannot craft tool or sticks — missing materials or recipe not found.');
      return;
    }
    await this._sendCraftRequest(result.recipe, result.matchedInputs);
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
