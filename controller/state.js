/**
 * Game state collection for the Relay adapter.
 *
 * Same output format as fpi-minecraft/src/state.js — the Python encoder
 * expects identical state dict keys.
 */

const { classifyVoxel, HOSTILE_MOBS, PASSIVE_MOBS, categorizeItem } = require('./categories');

/**
 * Scan a 7x7x5 voxel grid around the player and compute 24 player-relative
 * spatial features for navigation awareness.
 *
 * Grid: dx,dz in [-3,+3], dy in [-2,+2] relative to player feet.
 * Voxels classified as air(0), solid(1), danger(2).
 * All directional features are relative to the player's yaw.
 *
 * Returns: { body_clear, drop_depth, overhead, danger, composition, immediate }
 */
function scanSpatialGrid(adapter) {
  const pos = adapter.flooredPosition;
  const yaw = adapter.yaw || 0;
  const yawRad = yaw * Math.PI / 180;

  // Player-relative direction vectors (Minecraft: yaw=0 -> south/+Z)
  const fwdX = -Math.sin(yawRad);
  const fwdZ = Math.cos(yawRad);
  const rightX = -Math.cos(yawRad);
  const rightZ = -Math.sin(yawRad);

  // Classify all 245 voxels in the 7x7x5 grid
  const W = 7, D = 7, H = 5;
  const grid = new Uint8Array(W * D * H); // 0=air, 1=solid, 2=danger
  let totalAir = 0, totalDanger = 0;

  for (let dx = -3; dx <= 3; dx++) {
    for (let dz = -3; dz <= 3; dz++) {
      for (let dy = -2; dy <= 2; dy++) {
        const block = adapter.blockAt({ x: pos.x + dx, y: pos.y + dy, z: pos.z + dz });
        const cls = classifyVoxel(block ? block.name : null);
        const idx = (dx + 3) * D * H + (dz + 3) * H + (dy + 2);
        if (cls === 'danger') { grid[idx] = 2; totalDanger++; }
        else if (cls === 'solid') { grid[idx] = 1; }
        else { totalAir++; } // air stays 0
      }
    }
  }

  const total = W * D * H; // 245

  function voxelAt(dx, dz, dy) {
    if (dx < -3 || dx > 3 || dz < -3 || dz > 3 || dy < -2 || dy > 2) return 1; // OOB = solid
    return grid[(dx + 3) * D * H + (dz + 3) * H + (dy + 2)];
  }

  // 4 direction vectors: Forward, Back, Left, Right
  const dirs = [
    [fwdX, fwdZ],
    [-fwdX, -fwdZ],
    [-rightX, -rightZ],
    [rightX, rightZ],
  ];

  // Body clearance (4 dims): distance to first solid at foot+head level
  const bodyClear = [0, 0, 0, 0];
  for (let d = 0; d < 4; d++) {
    const [dirX, dirZ] = dirs[d];
    let clearDist = 3;
    for (let step = 1; step <= 3; step++) {
      const bx = Math.round(dirX * step);
      const bz = Math.round(dirZ * step);
      const foot = voxelAt(bx, bz, 0);
      const head = voxelAt(bx, bz, 1);
      if (foot === 1 || head === 1) {
        clearDist = step - 1;
        break;
      }
    }
    bodyClear[d] = clearDist / 3.0;
  }

  // Drop depth (4 dims): air blocks below feet 1 step in each direction
  const dropDepth = [0, 0, 0, 0];
  for (let d = 0; d < 4; d++) {
    const [dirX, dirZ] = dirs[d];
    const bx = Math.round(dirX);
    const bz = Math.round(dirZ);
    let drop = 0;
    for (let dy = -1; dy >= -2; dy--) {
      if (voxelAt(bx, bz, dy) === 0) drop++;
      else break;
    }
    dropDepth[d] = drop / 2.0;
  }

  // Overhead clearance (4 dims): air above head 1 step in each direction
  const overhead = [0, 0, 0, 0];
  for (let d = 0; d < 4; d++) {
    const [dirX, dirZ] = dirs[d];
    const bx = Math.round(dirX);
    const bz = Math.round(dirZ);
    overhead[d] = voxelAt(bx, bz, 2) === 0 ? 1.0 : 0.0;
  }

  // Danger map (4 dims): density in FL/FR/BL/BR quadrants
  const dangerQuad = [0, 0, 0, 0];
  const quadCounts = [0, 0, 0, 0];
  for (let dx = -3; dx <= 3; dx++) {
    for (let dz = -3; dz <= 3; dz++) {
      if (dx === 0 && dz === 0) continue;
      const fwd = dx * fwdX + dz * fwdZ;
      const rgt = dx * rightX + dz * rightZ;
      const qIdx = (fwd >= 0 ? 0 : 2) + (rgt >= 0 ? 1 : 0); // FL=0, FR=1, BL=2, BR=3
      for (let dy = -2; dy <= 2; dy++) {
        quadCounts[qIdx]++;
        if (voxelAt(dx, dz, dy) === 2) dangerQuad[qIdx]++;
      }
    }
  }
  for (let q = 0; q < 4; q++) {
    dangerQuad[q] = quadCounts[q] > 0 ? dangerQuad[q] / quadCounts[q] : 0;
  }

  // Composition (4 dims): air_ratio, wall_density, ground_coverage, danger_ratio
  const airRatio = totalAir / total;
  const dangerRatio = totalDanger / total;

  let bodyTotal = 0, bodySolid = 0;
  for (let dx = -3; dx <= 3; dx++) {
    for (let dz = -3; dz <= 3; dz++) {
      if (dx === 0 && dz === 0) continue;
      for (let dy = 0; dy <= 1; dy++) {
        bodyTotal++;
        if (voxelAt(dx, dz, dy) === 1) bodySolid++;
      }
    }
  }
  const wallDensity = bodyTotal > 0 ? bodySolid / bodyTotal : 0;

  let groundSolid = 0;
  for (let dx = -3; dx <= 3; dx++) {
    for (let dz = -3; dz <= 3; dz++) {
      if (voxelAt(dx, dz, -1) === 1) groundSolid++;
    }
  }
  const groundCoverage = groundSolid / 49; // 7x7

  // Immediate (4 dims): solid flags for blocks adjacent to player
  const belowFeet = voxelAt(0, 0, -1) === 1 ? 1.0 : 0.0;
  const ffx = Math.round(fwdX);
  const ffz = Math.round(fwdZ);
  const frontFoot = voxelAt(ffx, ffz, 0) === 1 ? 1.0 : 0.0;
  const frontHead = voxelAt(ffx, ffz, 1) === 1 ? 1.0 : 0.0;
  const aboveHead = voxelAt(0, 0, 2) === 1 ? 1.0 : 0.0;

  return {
    body_clear: bodyClear,
    drop_depth: dropDepth,
    overhead,
    danger: dangerQuad,
    composition: [airRatio, wallDensity, groundCoverage, dangerRatio],
    immediate: [belowFeet, frontFoot, frontHead, aboveHead],
  };
}

// Default spatial data for death/respawn state
const DEAD_SPATIAL = {
  body_clear: [1, 1, 1, 1],
  drop_depth: [0, 0, 0, 0],
  overhead: [1, 1, 1, 1],
  danger: [0, 0, 0, 0],
  composition: [0.6, 0, 1.0, 0],
  immediate: [1, 0, 0, 0],
};

/**
 * Compute bearing from player to entity, relative to player's facing direction.
 * Returns {sin, cos} of the relative angle — continuous, wrapping-safe encoding.
 * Result: sin>0 = entity to the right, cos>0 = entity in front.
 */
function computeBearing(playerPos, entityPos, playerYaw) {
  const dx = entityPos.x - playerPos.x;
  const dz = entityPos.z - playerPos.z;
  // Minecraft yaw: 0 = south (+Z), 90 = west (-X)
  const angleToEntity = Math.atan2(-dx, dz) * (180 / Math.PI);
  const relAngle = ((angleToEntity - playerYaw) + 540) % 360 - 180;
  const rad = relAngle * Math.PI / 180;
  return { sin: Math.sin(rad), cos: Math.cos(rad) };
}

/**
 * Compute per-entity facing score: how much is the entity looking toward the player.
 * Uses headYaw if available (from entity_head_rotation), falls back to body yaw.
 * Returns float [0, 1] — 1.0 = looking directly at player.
 */
function computePerEntityFacing(entity, playerPos) {
  if (!entity || !entity.position) return 0;
  const entityYaw = entity.headYaw !== undefined ? entity.headYaw : (entity.yaw || 0);
  const dx = playerPos.x - entity.position.x;
  const dz = playerPos.z - entity.position.z;
  const angleToPlayer = Math.atan2(-dx, dz) * (180 / Math.PI);
  let angleDiff = ((angleToPlayer - entityYaw) + 540) % 360 - 180;
  angleDiff = Math.abs(angleDiff);
  return Math.max(0, Math.min(1, 1.0 - (angleDiff / 180)));
}

/**
 * Compute approach speed: dot product of entity velocity with entity→player direction.
 * Returns float [-1, 1] — 1.0 = charging toward player, -1.0 = fleeing.
 */
function computeApproach(playerPos, entityPos, velocity) {
  const dx = playerPos.x - entityPos.x;
  const dz = playerPos.z - entityPos.z;
  const dist = Math.sqrt(dx * dx + dz * dz);
  if (dist < 0.01) return 0;
  const vx = velocity.x || 0;
  const vz = velocity.z || 0;
  const speed = Math.sqrt(vx * vx + vz * vz);
  if (speed < 0.01) return 0;
  // Dot product of normalized velocity with normalized direction to player
  return (vx * dx + vz * dz) / (speed * dist);
}

/**
 * Compute horizontal speed in blocks/second.
 */
function computeSpeed(velocity) {
  const vx = velocity.x || 0;
  const vz = velocity.z || 0;
  return Math.sqrt(vx * vx + vz * vz);
}

/**
 * Compute 4-quadrant hostile density weighted by proximity.
 * Quadrants: FL(0), FR(1), BL(2), BR(3), relative to player yaw.
 */
function computeQuadrantDensity(hostiles, playerPos, playerYaw) {
  const yawRad = playerYaw * Math.PI / 180;
  const fwdX = -Math.sin(yawRad);
  const fwdZ = Math.cos(yawRad);
  const rightX = -Math.cos(yawRad);
  const rightZ = -Math.sin(yawRad);
  const quad = [0, 0, 0, 0];
  for (const h of hostiles) {
    const dist = h.entry.distance;
    if (dist > 32) continue;
    const w = Math.max(0, 1.0 - dist / 32);
    const raw = h.raw;
    const dx = raw.position.x - playerPos.x;
    const dz = raw.position.z - playerPos.z;
    const fwd = dx * fwdX + dz * fwdZ;
    const rgt = dx * rightX + dz * rightZ;
    const qIdx = (fwd >= 0 ? 0 : 2) + (rgt >= 0 ? 1 : 0);
    quad[qIdx] += w;
  }
  // Normalize: cap at 5.0 (5 close enemies in one quadrant = saturated)
  for (let i = 0; i < 4; i++) quad[i] = Math.min(1.0, quad[i] / 5.0);
  return quad;
}

/**
 * Compute strongest threat direction as {sin, cos, magnitude} vector.
 * Weighted average of hostile directions by 1/distance^2.
 */
function computeThreatDirection(hostiles, playerPos, playerYaw) {
  if (hostiles.length === 0) return { sin: 0, cos: 0, magnitude: 0 };
  const yawRad = playerYaw * Math.PI / 180;
  let wx = 0, wz = 0, totalWeight = 0;
  for (const h of hostiles) {
    const raw = h.raw;
    const dx = raw.position.x - playerPos.x;
    const dz = raw.position.z - playerPos.z;
    const dist = h.entry.distance;
    if (dist < 0.5) continue;
    const weight = 1.0 / (dist * dist);
    wx += dx * weight;
    wz += dz * weight;
    totalWeight += weight;
  }
  if (totalWeight < 0.001) return { sin: 0, cos: 0, magnitude: 0 };
  wx /= totalWeight;
  wz /= totalWeight;
  const mag = Math.sqrt(wx * wx + wz * wz);
  if (mag < 0.001) return { sin: 0, cos: 0, magnitude: 0 };
  // Convert to player-relative angle
  const absAngle = Math.atan2(-wx, wz);
  const relAngle = absAngle - yawRad;
  return {
    sin: Math.sin(relAngle),
    cos: Math.cos(relAngle),
    magnitude: Math.min(1.0, totalWeight),
  };
}

/**
 * Get average armor tier from entity equipment (slots 2-5: boots, legs, chest, helm).
 * Returns 0.0 (none) to 1.0 (full netherite).
 */
function getArmorTier(entity) {
  if (!entity._equipment) return 0;
  let total = 0;
  for (const slot of [2, 3, 4, 5]) {
    const item = entity._equipment[slot];
    if (item && item.name) {
      const name = item.name.toLowerCase();
      if (name.includes('netherite')) total += 1.0;
      else if (name.includes('diamond')) total += 0.8;
      else if (name.includes('iron')) total += 0.6;
      else if (name.includes('chain')) total += 0.5;
      else if (name.includes('gold')) total += 0.3;
      else if (name.includes('leather')) total += 0.2;
    }
  }
  return total / 4; // average over 4 armor slots
}

// Projectile entity types to track as incoming threats
const PROJECTILE_TYPES = new Set([
  'arrow', 'spectral_arrow', 'fireball', 'small_fireball',
  'trident', 'shulker_bullet', 'wind_charge', 'wither_skull',
]);

/**
 * Get weapon tier from entity equipment (mainhand slot).
 * Returns 0.0 (none) to 1.0 (netherite).
 */
function getEquipmentTier(entity) {
  if (!entity._equipment) return 0;
  const mainhand = entity._equipment[0]; // slot 0 = mainhand
  if (!mainhand || !mainhand.name) return 0;
  const name = mainhand.name.toLowerCase();
  if (name.includes('netherite')) return 1.0;
  if (name.includes('diamond')) return 0.8;
  if (name.includes('iron')) return 0.6;
  if (name.includes('stone')) return 0.4;
  if (name.includes('wood') || name.includes('gold')) return 0.2;
  return 0.1; // has something but not a tiered weapon
}

function getNearbyEntities(adapter, maxDist, trackingState) {
  const entities = adapter.allEntities;
  const pos = adapter.position;
  const playerYaw = adapter.yaw || 0;

  const hostiles = [];
  const passives = [];
  const players = [];

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
    const vel = entity.velocity || { x: 0, y: 0, z: 0 };
    const bearing = computeBearing(pos, entity.position, playerYaw);
    const speed = computeSpeed(vel);
    const approach = computeApproach(pos, entity.position, vel);
    const facingUs = computePerEntityFacing(entity, pos);

    const entry = {
      name: nameLower,
      distance: dist,
      bearing,
      speed,
      approach,
      facing_us: facingUs,
      health: entity._health ?? -1,
      max_health: entity._maxHealth ?? 20,
      flags: entity._flags ?? 0,
      hand_state: entity._handState ?? 0,
      is_baby: entity._isBaby ?? false,
      creeper_state: entity._creeperState ?? -1,
      creeper_charged: entity._creeperCharged ?? false,
      equipment_tier: getEquipmentTier(entity),
    };

    if (entity.type === 'player') {
      entry.name = entity.username || nameLower;
      players.push({ entry, raw: entity });
    } else if (HOSTILE_MOBS.has(nameLower)) {
      hostiles.push({ entry, raw: entity });
    } else if (PASSIVE_MOBS.has(nameLower)) {
      passives.push({ entry, raw: entity });
    } else if (nameLower && !nameLower.startsWith('entity_')) {
      passives.push({ entry, raw: entity });
    }
  }

  // Sort by distance, take top N (expanded: 8 hostile, 4 passive, 4 player)
  hostiles.sort((a, b) => a.entry.distance - b.entry.distance);
  passives.sort((a, b) => a.entry.distance - b.entry.distance);
  players.sort((a, b) => a.entry.distance - b.entry.distance);

  // Crowd summary with directional awareness
  const hostileNear = hostiles.filter(h => h.entry.distance <= 8).length;
  const hostileAvgDist = hostiles.length > 0
    ? hostiles.reduce((s, h) => s + h.entry.distance, 0) / hostiles.length : 64;
  const quadDensity = computeQuadrantDensity(hostiles, pos, playerYaw);
  const threatDir = computeThreatDirection(hostiles, pos, playerYaw);

  // Attacker info from trackingState
  let attackerDist = 0, attackerBearing = { sin: 0, cos: 0 }, underAttack = 0;
  if (trackingState && trackingState.lastAttackerEntityId && trackingState.lastAttackerTime) {
    const elapsed = (Date.now() - trackingState.lastAttackerTime) / 1000;
    if (elapsed < 5) {
      underAttack = Math.max(0, 1.0 - elapsed / 5);
      const attacker = adapter._entities
        ? adapter._entities.get(trackingState.lastAttackerEntityId)
        : null;
      if (attacker && attacker.position) {
        const adx = pos.x - attacker.position.x;
        const ady = pos.y - attacker.position.y;
        const adz = pos.z - attacker.position.z;
        attackerDist = Math.sqrt(adx * adx + ady * ady + adz * adz);
        attackerBearing = computeBearing(pos, attacker.position, playerYaw);
      }
    }
  }

  const crowd = {
    quadrant_density: quadDensity,
    hostile_count: hostiles.length,
    hostile_avg_dist: hostileAvgDist,
    hostile_near: hostileNear,
    passive_count: passives.length,
    player_count: players.length,
    threat_direction: threatDir,
    attacker_dist: attackerDist,
    attacker_bearing: attackerBearing,
    under_attack: underAttack,
  };

  // Incoming projectile scan — find nearest approaching projectile
  let incomingProjectile = null;
  for (const entity of entities) {
    if (adapter.isSelf(entity)) continue;
    if (!entity.position) continue;
    const name = (entity.name || '').toLowerCase();
    if (!PROJECTILE_TYPES.has(name)) continue;
    const dx = pos.x - entity.position.x;
    const dy = pos.y - entity.position.y;
    const dz = pos.z - entity.position.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > 32) continue;
    const vel = entity.velocity || { x: 0, y: 0, z: 0 };
    const approach = computeApproach(pos, entity.position, vel);
    if (approach > 0 && dist < (incomingProjectile ? incomingProjectile.distance : Infinity)) {
      incomingProjectile = {
        name,
        distance: dist,
        speed: computeSpeed(vel),
        bearing: computeBearing(pos, entity.position, playerYaw),
      };
    }
  }

  // Nearest hostile acceleration (speed delta detection)
  let nearestHostileAccel = 0;
  if (hostiles.length > 0 && trackingState) {
    const nearest = hostiles[0];
    const eid = nearest.raw.id || nearest.raw.runtimeId;
    if (eid !== undefined) {
      const prevSpeed = (trackingState._prevEntitySpeeds || new Map()).get(eid) || nearest.entry.speed;
      nearestHostileAccel = nearest.entry.speed - prevSpeed;
      if (!trackingState._prevEntitySpeeds) trackingState._prevEntitySpeeds = new Map();
      trackingState._prevEntitySpeeds.set(eid, nearest.entry.speed);
    }
  }

  // Player armor tier (nearest player's average armor)
  const nearestPlayerArmor = players.length > 0 ? getArmorTier(players[0].raw) : 0;

  // Height advantage vs nearest hostile and player
  const heightVsHostile = hostiles.length > 0
    ? pos.y - (hostiles[0].raw.position?.y || pos.y) : 0;
  const heightVsPlayer = players.length > 0
    ? pos.y - (players[0].raw.position?.y || pos.y) : 0;

  return {
    hostiles: hostiles.slice(0, 8).map(h => h.entry),
    passives: passives.slice(0, 4).map(p => p.entry),
    players: players.slice(0, 4).map(p => p.entry),
    crowd,
    incoming_projectile: incomingProjectile,
    nearest_hostile_accel: nearestHostileAccel,
    nearest_player_armor: nearestPlayerArmor,
    height_vs_hostile: heightVsHostile,
    height_vs_player: heightVsPlayer,
  };
}

function summarizeInventory(adapter) {
  const items = adapter.inventoryItems;
  const selectedSlot = adapter.quickBarSlot || 0;

  // Build hotbar array (9 slots, null for empty)
  // Index items by slot — only hotbar slots 0-8
  const hotbarBySlot = new Map();
  let slotsUsed = 0;

  for (const item of items) {
    slotsUsed++;
    if (item.slot >= 0 && item.slot <= 8) {
      hotbarBySlot.set(item.slot, item);
    }
  }

  const hotbar = [];
  for (let s = 0; s < 9; s++) {
    const item = hotbarBySlot.get(s);
    if (!item) {
      hotbar.push(null);
      continue;
    }
    const info = categorizeItem(item.name);
    // Extract durability from metadata if available
    let durabilityFraction = 1.0;
    if (info.maxDurability > 0) {
      const damage = item.metadata?.Damage ?? item.metadata?.damage ?? 0;
      durabilityFraction = Math.max(0, 1.0 - (damage / info.maxDurability));
    }
    hotbar.push({
      category: info.category,
      tier: info.tier,
      durability: durabilityFraction,
      count: item.count || 1,
      max_stack: info.maxStack,
    });
  }

  return {
    slots_used: slotsUsed,
    selected_slot: selectedSlot,
    hotbar,
  };
}

/**
 * Compute self-armor tier from adapter inventory (armor slots).
 * Returns 0.0 (naked) to 1.0 (full netherite).
 */
function computeSelfArmorTier(adapter) {
  const items = adapter.inventoryItems;
  if (!items || items.length === 0) return 0;
  let total = 0, count = 0;
  for (const item of items) {
    // Armor slots: 5=helmet, 6=chest, 7=legs, 8=boots (Java slot numbering)
    if (item.slot < 5 || item.slot > 8) continue;
    if (!item.name) continue;
    count++;
    const name = item.name.toLowerCase();
    if (name.includes('netherite')) total += 1.0;
    else if (name.includes('diamond')) total += 0.8;
    else if (name.includes('iron')) total += 0.6;
    else if (name.includes('chain')) total += 0.5;
    else if (name.includes('gold')) total += 0.3;
    else if (name.includes('leather')) total += 0.2;
  }
  return count > 0 ? total / 4 : 0; // always divide by 4 armor slots
}

function getState(adapter, trackingState) {
  if (trackingState.pendingRespawn || adapter.health <= 0) {
    const kills = trackingState.killsSinceLastState;
    trackingState.killsSinceLastState = 0;
    trackingState.projectileHitLanded = false;
    trackingState.projectilePlayerHitLanded = false;
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
      spatial: DEAD_SPATIAL,
      entities: { hostiles: [], passives: [], players: [] },
      crowd: {
        quadrant_density: [0, 0, 0, 0],
        hostile_count: 0, hostile_avg_dist: 64, hostile_near: 0,
        passive_count: 0, player_count: 0,
        threat_direction: { sin: 0, cos: 0, magnitude: 0 },
        attacker_dist: 0, attacker_bearing: { sin: 0, cos: 0 }, under_attack: 0,
      },
      inventory: { slots_used: 0, selected_slot: 0, hotbar: [null, null, null, null, null, null, null, null, null] },
      alive: false,
      hit_landed: false,
      player_hit_landed: false,
      kills: kills,
      attack_cooldown: 0,
      bot_control_active: adapter.botControlActive || false,
      user_active: adapter.userActive || false,
      is_using_item: adapter._isUsingItem || false,
      // Self-awareness defaults (dead state)
      self_velocity: { x: 0, y: 0, z: 0 },
      health_delta: 0,
      food_delta: 0,
      ticks_airborne: 0,
      self_effects: { speed: 0, strength: 0, resistance: 0, regeneration: 0 },
      incoming_projectile: null,
      self_armor_tier: 0,
      is_thundering: false,
      nearest_hostile_accel: 0,
      nearest_player_armor: 0,
      height_vs_hostile: 0,
      height_vs_player: 0,
      combat_hits_5s: 0,
      combat_damage_5s: 0,
      time_since_hit: 1.0,
      kill_streak: 0,
      strafing: 0,
    };
  }

  const pos = adapter.position;
  const lightLevel = adapter.lightAt(pos);
  const now = Date.now();

  const kills = trackingState.killsSinceLastState;
  trackingState.killsSinceLastState = 0;

  // --- Self-velocity computation ---
  let selfVelocity = { x: 0, y: 0, z: 0 };
  if (trackingState._prevPosition && trackingState._prevPositionTime) {
    const dt = (now - trackingState._prevPositionTime) / 1000;
    if (dt > 0.01 && dt < 5.0) {
      selfVelocity = {
        x: (pos.x - trackingState._prevPosition.x) / dt,
        y: (pos.y - trackingState._prevPosition.y) / dt,
        z: (pos.z - trackingState._prevPosition.z) / dt,
      };
    }
  }
  trackingState._prevPosition = { x: pos.x, y: pos.y, z: pos.z };
  trackingState._prevPositionTime = now;

  // --- Health/food deltas ---
  const healthDelta = adapter.health - (trackingState._prevHealth ?? adapter.health);
  const foodDelta = adapter.food - (trackingState._prevFood ?? adapter.food);
  trackingState._prevHealth = adapter.health;
  trackingState._prevFood = adapter.food;

  // --- Airborne tracking ---
  if (adapter.onGround) {
    trackingState._ticksAirborne = 0;
  } else {
    trackingState._ticksAirborne = (trackingState._ticksAirborne || 0) + 1;
  }

  // --- Combat momentum ring buffers ---
  if (trackingState.lastAttackLanded) {
    trackingState._recentHits.push(now);
  }
  if (healthDelta < 0) {
    trackingState._recentDamage.push({ time: now, amount: -healthDelta });
  }
  if (kills > 0) {
    for (let i = 0; i < kills; i++) trackingState._recentKills.push(now);
  }
  // Prune old entries
  trackingState._recentHits = trackingState._recentHits.filter(t => now - t < 10000);
  trackingState._recentDamage = trackingState._recentDamage.filter(e => now - e.time < 5000);
  trackingState._recentKills = trackingState._recentKills.filter(t => now - t < 30000);

  // Hits in last 5s (subset of 10s buffer)
  const hitsLast5s = trackingState._recentHits.filter(t => now - t < 5000).length;
  const damageLast5s = trackingState._recentDamage.reduce((s, e) => s + e.amount, 0);
  const lastHitTime = trackingState._recentHits.length > 0
    ? trackingState._recentHits[trackingState._recentHits.length - 1] : 0;
  const timeSinceHit = lastHitTime > 0 ? (now - lastHitTime) / 1000 : 10.0; // seconds

  // --- Status effects ---
  const effects = adapter.activeEffects || new Map();
  const selfEffects = {
    speed: effects.has(1) ? (effects.get(1).amplifier + 1) : 0,
    strength: effects.has(5) ? (effects.get(5).amplifier + 1) : 0,
    resistance: effects.has(11) ? (effects.get(11).amplifier + 1) : 0,
    regeneration: effects.has(10) ? (effects.get(10).amplifier + 1) : 0,
  };

  // --- Strafing direction ---
  const yawRad = (adapter.yaw || 0) * Math.PI / 180;
  const fwdX = -Math.sin(yawRad);
  const fwdZ = Math.cos(yawRad);
  const rightX = fwdZ, rightZ = -fwdX; // perpendicular right
  const strafing = selfVelocity.x * rightX + selfVelocity.z * rightZ;

  // --- Self armor tier ---
  const selfArmorTier = computeSelfArmorTier(adapter);

  const entityInfo = getNearbyEntities(adapter, 64, trackingState);

  const state = {
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
    spatial: scanSpatialGrid(adapter),
    entities: {
      hostiles: entityInfo.hostiles,
      passives: entityInfo.passives,
      players: entityInfo.players,
    },
    crowd: entityInfo.crowd,
    inventory: summarizeInventory(adapter),
    alive: true,
    hit_landed: trackingState.lastAttackLanded || trackingState.projectileHitLanded,
    player_hit_landed: trackingState.lastPlayerHitLanded || trackingState.projectilePlayerHitLanded,
    kills: kills,
    attack_cooldown: trackingState.attackCooldown || 0,
    bot_control_active: adapter.botControlActive || false,
    user_active: adapter.userActive || false,
    is_using_item: adapter._isUsingItem || false,
    // Self-awareness fields
    self_velocity: selfVelocity,
    health_delta: healthDelta,
    food_delta: foodDelta,
    ticks_airborne: trackingState._ticksAirborne,
    self_effects: selfEffects,
    incoming_projectile: entityInfo.incoming_projectile,
    self_armor_tier: selfArmorTier,
    is_thundering: adapter.isThundering || false,
    // Threat dynamics fields
    nearest_hostile_accel: entityInfo.nearest_hostile_accel,
    nearest_player_armor: entityInfo.nearest_player_armor,
    height_vs_hostile: entityInfo.height_vs_hostile,
    height_vs_player: entityInfo.height_vs_player,
    combat_hits_5s: hitsLast5s,
    combat_damage_5s: damageLast5s,
    time_since_hit: timeSinceHit,
    kill_streak: trackingState._recentKills.length,
    strafing: strafing,
  };
  // Reset projectile flags after reading (persist across action ticks until consumed)
  trackingState.projectileHitLanded = false;
  trackingState.projectilePlayerHitLanded = false;
  return state;
}

module.exports = { getState, getNearbyEntities, scanSpatialGrid, summarizeInventory };
