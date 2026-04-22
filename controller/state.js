/**
 * Game state collection for the Relay adapter.
 *
 * Same output format as fpi-minecraft/src/state.js — the Python encoder
 * expects identical state dict keys.
 */

const { categorizeBlock, HOSTILE_MOBS, PASSIVE_MOBS } = require('./categories');

function scanNearbyBlocks(adapter, radius) {
  const counts = { air: 0, stone: 0, dirt: 0, wood: 0, water: 0, ore: 0, danger: 0, other: 0 };
  let total = 0;

  const pos = adapter.flooredPosition;
  for (let dx = -radius; dx <= radius; dx++) {
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dz = -radius; dz <= radius; dz++) {
        const block = adapter.blockAt({ x: pos.x + dx, y: pos.y + dy, z: pos.z + dz });
        if (block) {
          const cat = categorizeBlock(block.name);
          counts[cat]++;
        }
        total++;
      }
    }
  }

  const ratios = {};
  for (const [cat, count] of Object.entries(counts)) {
    ratios[cat] = total > 0 ? count / total : 0;
  }
  return ratios;
}

function computeFacing(entity, targetPos) {
  if (!entity || !entity.position || entity.yaw === undefined) {
    return null;
  }
  const dx = targetPos.x - entity.position.x;
  const dz = targetPos.z - entity.position.z;
  const angleToTarget = Math.atan2(-dx, dz);
  const entityYaw = entity.yaw || 0;
  let angleDiff = Math.abs(((angleToTarget - entityYaw) + Math.PI) % (2 * Math.PI) - Math.PI);
  const facingUs = 1.0 - (angleDiff / Math.PI);
  return { facing_us: Math.max(0, Math.min(1, facingUs)), angle_diff: angleDiff };
}

function getNearbyEntities(adapter, maxDist) {
  const entities = adapter.allEntities;
  const pos = adapter.position;

  let nearestHostile = null;
  let nearestHostileDist = Infinity;
  let nearestHostileEntity = null;
  let nearestPassive = null;
  let nearestPassiveDist = Infinity;
  let nearestPlayer = null;
  let nearestPlayerDist = Infinity;
  let nearestPlayerEntity = null;

  for (const entity of entities) {
    if (adapter.isSelf(entity)) continue;
    if (!entity.position) continue;

    const dx = pos.x - entity.position.x;
    const dy = pos.y - entity.position.y;
    const dz = pos.z - entity.position.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > maxDist) continue;

    const name = entity.name || entity.displayName || '';
    const nameLower = name.toLowerCase();

    if (entity.type === 'player') {
      if (dist < nearestPlayerDist) {
        nearestPlayerDist = dist;
        nearestPlayer = { name: entity.username || nameLower, distance: dist };
        nearestPlayerEntity = entity;
      }
    } else if (HOSTILE_MOBS.has(nameLower)) {
      if (dist < nearestHostileDist) {
        nearestHostileDist = dist;
        nearestHostile = { name: nameLower, distance: dist };
        nearestHostileEntity = entity;
      }
    } else if (PASSIVE_MOBS.has(nameLower)) {
      if (dist < nearestPassiveDist) {
        nearestPassiveDist = dist;
        nearestPassive = { name: nameLower, distance: dist };
      }
    }
  }

  const hostileFacing = computeFacing(nearestHostileEntity, pos);
  const playerFacing = computeFacing(nearestPlayerEntity, pos);

  return {
    hostile: nearestHostile,
    passive: nearestPassive,
    player: nearestPlayer,
    hostile_facing: hostileFacing,
    player_facing: playerFacing,
  };
}

function summarizeInventory(adapter) {
  const items = adapter.inventoryItems;
  let slotsUsed = 0;
  let hasWeapon = false;
  let hasFood = false;
  let hasWood = false;
  let hasTool = false;

  const weaponNames = new Set([
    'wooden_sword', 'stone_sword', 'iron_sword', 'golden_sword',
    'diamond_sword', 'netherite_sword', 'bow', 'crossbow', 'trident',
  ]);
  const toolNames = new Set([
    'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'golden_pickaxe',
    'diamond_pickaxe', 'netherite_pickaxe',
    'wooden_axe', 'stone_axe', 'iron_axe', 'golden_axe',
    'diamond_axe', 'netherite_axe',
    'wooden_shovel', 'stone_shovel', 'iron_shovel', 'golden_shovel',
    'diamond_shovel', 'netherite_shovel',
    'wooden_hoe', 'stone_hoe', 'iron_hoe', 'golden_hoe',
    'diamond_hoe', 'netherite_hoe',
  ]);
  const woodNames = new Set([
    'oak_log', 'spruce_log', 'birch_log', 'jungle_log', 'acacia_log',
    'dark_oak_log', 'mangrove_log', 'cherry_log',
    'oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks',
    'acacia_planks', 'dark_oak_planks', 'mangrove_planks', 'cherry_planks',
    'stick',
  ]);

  for (const item of items) {
    slotsUsed++;
    const name = item.name;
    if (weaponNames.has(name)) hasWeapon = true;
    if (toolNames.has(name)) hasTool = true;
    if (woodNames.has(name)) hasWood = true;
    if (name.includes('apple') || name.includes('bread') ||
        name.includes('cooked') || name.includes('steak') ||
        name.includes('porkchop') || name.includes('mutton') ||
        name.includes('chicken') || name.includes('rabbit') ||
        name.includes('cod') || name.includes('salmon') ||
        name.includes('potato') || name.includes('carrot') ||
        name.includes('beetroot') || name.includes('melon') ||
        name.includes('berries') || name.includes('cookie') ||
        name.includes('pie') || name.includes('cake') ||
        name.includes('mushroom_stew') || name.includes('golden_carrot')) {
      hasFood = true;
    }
  }

  return {
    slots_used: slotsUsed,
    has_weapon: hasWeapon,
    has_food: hasFood,
    has_wood: hasWood,
    has_tool: hasTool,
  };
}

function getState(adapter, trackingState) {
  if (trackingState.pendingRespawn || adapter.health <= 0) {
    const kills = trackingState.killsSinceLastState;
    trackingState.killsSinceLastState = 0;
    const pos = adapter.position;
    return {
      health: 0,
      food: 0,
      food_saturation: 0,
      xp_level: 0,
      xp_points: 0,
      position: { x: pos.x, y: pos.y, z: pos.z },
      yaw: 0,
      pitch: 0,
      on_ground: true,
      is_in_water: false,
      is_raining: false,
      time_of_day: 0,
      light_level: 0,
      altitude: 64,
      block_composition: { air: 0.5, stone: 0.1, dirt: 0.2, wood: 0.05, water: 0, ore: 0, danger: 0, other: 0.15 },
      entities: { hostile: null, passive: null },
      inventory: { slots_used: 0, has_weapon: false, has_food: false, has_wood: false, has_tool: false },
      alive: false,
      hit_landed: false,
      player_hit_landed: false,
      kills: kills,
      attack_cooldown: 0,
      hostile_facing: null,
      player_facing: null,
      bot_control_active: adapter.botControlActive || false,
    };
  }

  const pos = adapter.position;
  const lightLevel = adapter.lightAt(pos);

  const kills = trackingState.killsSinceLastState;
  trackingState.killsSinceLastState = 0;

  const entityInfo = getNearbyEntities(adapter, 64);

  return {
    health: adapter.health,
    food: adapter.food,
    food_saturation: adapter.foodSaturation,
    xp_level: adapter.xpLevel,
    xp_points: adapter.xpPoints,
    position: { x: pos.x, y: pos.y, z: pos.z },
    yaw: adapter.yaw,
    pitch: adapter.pitch,
    on_ground: adapter.onGround,
    is_in_water: adapter.isInWater,
    is_raining: adapter.isRaining,
    time_of_day: adapter.timeOfDay,
    light_level: lightLevel,
    altitude: pos.y,
    block_composition: scanNearbyBlocks(adapter, 8),
    entities: {
      hostile: entityInfo.hostile,
      passive: entityInfo.passive,
      player: entityInfo.player,
    },
    inventory: summarizeInventory(adapter),
    alive: true,
    hit_landed: trackingState.lastAttackLanded,
    player_hit_landed: trackingState.lastPlayerHitLanded,
    kills: kills,
    attack_cooldown: trackingState.attackCooldown || 0,
    hostile_facing: entityInfo.hostile_facing,
    player_facing: entityInfo.player_facing,
    bot_control_active: adapter.botControlActive || false,
  };
}

module.exports = { getState, getNearbyEntities, scanNearbyBlocks, summarizeInventory };
