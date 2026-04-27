/**
 * Macro-actions — hierarchical action layer with A* pathfinding.
 *
 * Sits between the FPI agent (which selects macro-action IDs 20-24) and
 * the primitive action system. Each macro executes a multi-tick behavior:
 *
 *   20: APPROACH_TARGET  — A* path to nearest hostile/player, stop at 3 blocks
 *   21: FLEE             — Sprint away from nearest threat
 *   22: MINE_BLOCK_BELOW — Look down + break block at feet
 *   23: GO_TO_COORDINATES — A* to explicit {x,y,z} (bridge-only, not in FPI space)
 *   24: APPROACH_PASSIVE  — A* path to nearest passive mob, stop at 3 blocks
 *
 * Uses the block cache for A* pathfinding. Falls back to direct movement
 * when chunks haven't loaded yet.
 */

const { HOSTILE_MOBS, PASSIVE_MOBS, PASSABLE_BLOCKS, DANGER_BLOCKS } = require('./categories');
const { sleep } = require('./utils');

// ---- Macro-action IDs ----

const MACRO_IDS = {
  APPROACH_TARGET: 20,
  FLEE: 21,
  MINE_BLOCK_BELOW: 22,
  GO_TO_COORDINATES: 23,
  APPROACH_PASSIVE: 24,
};

// ---- Block classification for pathfinding ----

function isPassthrough(name) {
  if (!name) return true; // null/unloaded → treat as air
  return PASSABLE_BLOCKS.has(name);
}

function isSolid(name) {
  if (!name) return false;
  return !PASSABLE_BLOCKS.has(name) && name !== 'water';
}

function isDangerous(name) {
  if (!name) return false;
  return DANGER_BLOCKS.has(name);
}

/**
 * Check if a position is walkable: passthrough at feet+head, solid floor, no danger.
 */
function isWalkable(adapter, x, y, z) {
  const feet = adapter.blockAt({ x, y, z });
  const head = adapter.blockAt({ x, y: y + 1, z });
  const floor = adapter.blockAt({ x, y: y - 1, z });

  const feetName = feet ? feet.name : null;
  const headName = head ? head.name : null;
  const floorName = floor ? floor.name : null;

  if (!isPassthrough(feetName)) return false;
  if (!isPassthrough(headName)) return false;
  if (!isSolid(floorName)) return false;
  if (isDangerous(feetName) || isDangerous(floorName)) return false;

  return true;
}

// ---- Binary min-heap for A* open set ----

class MinHeap {
  constructor() {
    this._data = [];
  }
  get size() { return this._data.length; }

  push(key, f) {
    this._data.push({ key, f });
    this._bubbleUp(this._data.length - 1);
  }

  pop() {
    const top = this._data[0];
    const last = this._data.pop();
    if (this._data.length > 0 && last) {
      this._data[0] = last;
      this._sinkDown(0);
    }
    return top;
  }

  _bubbleUp(i) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this._data[i].f >= this._data[parent].f) break;
      [this._data[i], this._data[parent]] = [this._data[parent], this._data[i]];
      i = parent;
    }
  }

  _sinkDown(i) {
    const n = this._data.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      if (l < n && this._data[l].f < this._data[smallest].f) smallest = l;
      if (r < n && this._data[r].f < this._data[smallest].f) smallest = r;
      if (smallest === i) break;
      [this._data[i], this._data[smallest]] = [this._data[smallest], this._data[i]];
      i = smallest;
    }
  }
}

// ---- A* pathfinder ----

// Cardinal directions: dx, dz
const CARDINALS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
// Diagonal directions: dx, dz
const DIAGONALS = [[1, 1], [1, -1], [-1, 1], [-1, -1]];

function heuristic(ax, ay, az, bx, by, bz) {
  const dx = ax - bx;
  const dy = ay - by;
  const dz = az - bz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function posKey(x, y, z) { return `${x},${y},${z}`; }

/**
 * A* pathfinder over the block cache.
 *
 * @param {object} adapter - RelayAdapter with blockAt() method
 * @param {{x:number,y:number,z:number}} start - Start position (floored integers)
 * @param {{x:number,y:number,z:number}} goal - Goal position (floored integers)
 * @param {object} [options] - { maxDistance: 32, maxIterations: 2000 }
 * @returns {Array<{x:number,y:number,z:number}>|null} Waypoints (excluding start) or null
 */
function findPath(adapter, start, goal, options = {}) {
  const maxDist = options.maxDistance || 32;
  const maxIter = options.maxIterations || 2000;

  // Quick checks
  const dx = goal.x - start.x;
  const dy = goal.y - start.y;
  const dz = goal.z - start.z;
  if (Math.sqrt(dx * dx + dy * dy + dz * dz) > maxDist) return null;

  const startKey = posKey(start.x, start.y, start.z);
  const goalKey = posKey(goal.x, goal.y, goal.z);

  if (startKey === goalKey) return [];

  const open = new MinHeap();
  const gScore = new Map(); // key → g cost
  const parent = new Map(); // key → parent key
  const closed = new Set();

  gScore.set(startKey, 0);
  open.push(startKey, heuristic(start.x, start.y, start.z, goal.x, goal.y, goal.z));

  let iterations = 0;

  while (open.size > 0 && iterations < maxIter) {
    iterations++;
    const { key: currentKey } = open.pop();

    if (currentKey === goalKey) {
      // Reconstruct path
      const path = [];
      let k = goalKey;
      while (k !== startKey) {
        const [cx, cy, cz] = k.split(',').map(Number);
        path.push({ x: cx, y: cy, z: cz });
        k = parent.get(k);
        if (!k) break;
      }
      path.reverse();
      return path;
    }

    if (closed.has(currentKey)) continue;
    closed.add(currentKey);

    const [cx, cy, cz] = currentKey.split(',').map(Number);
    const currentG = gScore.get(currentKey);

    // Generate neighbors
    const neighbors = [];

    // Cardinal flat moves (cost 1.0)
    for (const [ndx, ndz] of CARDINALS) {
      const nx = cx + ndx;
      const nz = cz + ndz;
      if (isWalkable(adapter, nx, cy, nz)) {
        neighbors.push({ x: nx, y: cy, z: nz, cost: 1.0 });
      }
    }

    // Diagonal flat moves (cost 1.414) — only if both adjacent cardinals are passable
    for (const [ndx, ndz] of DIAGONALS) {
      const nx = cx + ndx;
      const nz = cz + ndz;
      if (isWalkable(adapter, nx, cy, nz)) {
        // Check that we can pass through both cardinal adjacents (no corner cutting)
        const feet1 = adapter.blockAt({ x: cx + ndx, y: cy, z: cz });
        const feet2 = adapter.blockAt({ x: cx, y: cy, z: cz + ndz });
        if (isPassthrough(feet1 ? feet1.name : null) && isPassthrough(feet2 ? feet2.name : null)) {
          neighbors.push({ x: nx, y: cy, z: nz, cost: 1.414 });
        }
      }
    }

    // Cardinal jump up 1 (cost 2.0): blocked at current level, walkable one above
    for (const [ndx, ndz] of CARDINALS) {
      const nx = cx + ndx;
      const nz = cz + ndz;
      const ny = cy + 1;
      // Must have headroom above current pos (y+2 passable)
      const aboveHead = adapter.blockAt({ x: cx, y: cy + 2, z: cz });
      if (!isPassthrough(aboveHead ? aboveHead.name : null)) continue;
      if (isWalkable(adapter, nx, ny, nz)) {
        neighbors.push({ x: nx, y: ny, z: nz, cost: 2.0 });
      }
    }

    // Cardinal drop down 1 (cost 0.8)
    for (const [ndx, ndz] of CARDINALS) {
      const nx = cx + ndx;
      const nz = cz + ndz;
      const ny = cy - 1;
      if (isWalkable(adapter, nx, ny, nz)) {
        // Must have passthrough at destination head level (ny+1 = cy)
        const destHead = adapter.blockAt({ x: nx, y: cy, z: nz });
        if (isPassthrough(destHead ? destHead.name : null)) {
          neighbors.push({ x: nx, y: ny, z: nz, cost: 0.8 });
        }
      }
    }

    // Drop straight down 2 (cost 0.5)
    if (isWalkable(adapter, cx, cy - 2, cz)) {
      const mid = adapter.blockAt({ x: cx, y: cy - 1, z: cz });
      if (isPassthrough(mid ? mid.name : null)) {
        neighbors.push({ x: cx, y: cy - 2, z: cz, cost: 0.5 });
      }
    }

    // Drop straight down 3 (cost 0.5)
    if (isWalkable(adapter, cx, cy - 3, cz)) {
      const mid1 = adapter.blockAt({ x: cx, y: cy - 1, z: cz });
      const mid2 = adapter.blockAt({ x: cx, y: cy - 2, z: cz });
      if (isPassthrough(mid1 ? mid1.name : null) && isPassthrough(mid2 ? mid2.name : null)) {
        neighbors.push({ x: cx, y: cy - 3, z: cz, cost: 0.5 });
      }
    }

    // Score neighbors
    for (const nb of neighbors) {
      const nbKey = posKey(nb.x, nb.y, nb.z);
      if (closed.has(nbKey)) continue;

      // Distance from start to goal check
      const distFromStart = heuristic(start.x, start.y, start.z, nb.x, nb.y, nb.z);
      if (distFromStart > maxDist) continue;

      const tentativeG = currentG + nb.cost;
      const prevG = gScore.get(nbKey);
      if (prevG !== undefined && tentativeG >= prevG) continue;

      gScore.set(nbKey, tentativeG);
      parent.set(nbKey, currentKey);
      const f = tentativeG + heuristic(nb.x, nb.y, nb.z, goal.x, goal.y, goal.z);
      open.push(nbKey, f);
    }
  }

  return null; // No path found
}

// ---- Movement executor ----

/**
 * Walk along a waypoint path using adapter controls.
 *
 * @param {object} adapter
 * @param {Array<{x:number,y:number,z:number}>} path - Waypoints from findPath()
 * @param {object} trackingState
 * @param {object} [options] - { maxTicks, goalRadius, abortOnDamage }
 * @returns {Promise<string>} "completed" | "aborted" | "timeout"
 */
async function executePath(adapter, path, trackingState, options = {}) {
  const maxTicks = options.maxTicks || 40;
  const goalRadius = options.goalRadius || 1.5;
  const abortOnDamage = options.abortOnDamage !== false;
  const startHealth = adapter.health;

  if (!path || path.length === 0) {
    return 'completed';
  }

  // Copy path so we can shift without mutating caller's array
  const waypoints = path.slice();

  for (let tick = 0; tick < maxTicks; tick++) {
    // Abort on damage
    if (abortOnDamage && adapter.health < startHealth) {
      adapter.clearControlStates();
      return 'aborted';
    }

    const pos = adapter.position;
    const wp = waypoints[0];
    if (!wp) {
      adapter.clearControlStates();
      return 'completed';
    }

    // Check if we've reached this waypoint (2D distance)
    const dx = (wp.x + 0.5) - pos.x;
    const dz = (wp.z + 0.5) - pos.z;
    const hdist = Math.sqrt(dx * dx + dz * dz);
    if (hdist < 0.8) {
      waypoints.shift();
      if (waypoints.length === 0) {
        adapter.clearControlStates();
        return 'completed';
      }
      continue; // Re-evaluate next waypoint this tick
    }

    // Look toward waypoint
    const targetPos = { x: wp.x + 0.5, y: pos.y, z: wp.z + 0.5 };
    await adapter.lookAt(targetPos);

    // Set movement controls
    adapter.clearControlStates();
    adapter.setControlState('forward', true);

    // Jump if waypoint is above us
    const dy = wp.y - Math.floor(pos.y);
    if (dy > 0) {
      adapter.setControlState('jump', true);
    }

    // Sprint if more than 4 blocks from final destination
    const lastWp = waypoints[waypoints.length - 1];
    const distToEnd = Math.sqrt(
      ((lastWp.x + 0.5) - pos.x) ** 2 + ((lastWp.z + 0.5) - pos.z) ** 2,
    );
    if (distToEnd > 4) {
      adapter.setControlState('sprint', true);
    }

    await sleep(50); // One game tick
  }

  adapter.clearControlStates();
  return 'timeout';
}

// ---- Entity helpers ----

function findNearestBySet(adapter, mobSet, maxDist, includePlayer) {
  const entities = adapter.allEntities;
  const pos = adapter.position;
  let best = null;
  let bestDist = Infinity;

  for (const entity of entities) {
    if (adapter.isSelf(entity)) continue;
    if (!entity.position) continue;
    const name = (entity.name || '').toLowerCase();
    const isPlayer = entity.type === 'player';
    if (!isPlayer && !mobSet.has(name)) continue;
    if (isPlayer && !includePlayer) continue;

    const dx = pos.x - entity.position.x;
    const dy = pos.y - entity.position.y;
    const dz = pos.z - entity.position.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < maxDist && dist < bestDist) {
      best = entity;
      bestDist = dist;
    }
  }
  return best;
}

function floorPos(pos) {
  return { x: Math.floor(pos.x), y: Math.floor(pos.y), z: Math.floor(pos.z) };
}

function distanceTo(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Compute a goal position `stopDist` blocks from the target, along the line from us to target.
 */
function approachGoal(ourPos, targetPos, stopDist) {
  const dx = targetPos.x - ourPos.x;
  const dz = targetPos.z - ourPos.z;
  const dist = Math.sqrt(dx * dx + dz * dz);
  if (dist <= stopDist) return floorPos(ourPos);
  const scale = (dist - stopDist) / dist;
  return {
    x: Math.floor(ourPos.x + dx * scale),
    y: Math.floor(targetPos.y),
    z: Math.floor(ourPos.z + dz * scale),
  };
}

// ---- Direct movement fallback (Java or no block cache) ----

/**
 * Walk directly toward a target without pathfinding.
 * Used when block cache is unavailable (Java edition).
 */
async function directApproach(adapter, target, trackingState, options = {}) {
  const maxTicks = options.maxTicks || 40;
  const stopDist = options.stopDist || 3.0;
  const abortOnDamage = options.abortOnDamage !== false;
  const startHealth = adapter.health;

  for (let tick = 0; tick < maxTicks; tick++) {
    if (abortOnDamage && adapter.health < startHealth) {
      adapter.clearControlStates();
      return 'aborted';
    }
    if (!target.position) {
      adapter.clearControlStates();
      return 'no_target';
    }
    const dist = distanceTo(adapter.position, target.position);
    if (dist <= stopDist) {
      adapter.clearControlStates();
      return 'completed';
    }

    await adapter.lookAt(target.position);
    adapter.clearControlStates();
    adapter.setControlState('forward', true);
    if (dist > 6) adapter.setControlState('sprint', true);
    await sleep(50);
  }

  adapter.clearControlStates();
  return 'timeout';
}

/**
 * Sprint directly away from a direction for N ticks.
 */
async function directFlee(adapter, awayFromPos, trackingState, maxTicks) {
  // Look away from threat
  const pos = adapter.position;
  const dx = pos.x - awayFromPos.x;
  const dz = pos.z - awayFromPos.z;
  // Compute a point 10 blocks in the opposite direction and look at it
  const dist = Math.sqrt(dx * dx + dz * dz);
  const nx = dist > 0.01 ? dx / dist : 0;
  const nz = dist > 0.01 ? dz / dist : 1;
  const lookTarget = { x: pos.x + nx * 10, y: pos.y, z: pos.z + nz * 10 };
  await adapter.lookAt(lookTarget);

  adapter.clearControlStates();
  adapter.setControlState('forward', true);
  adapter.setControlState('sprint', true);
  for (let t = 0; t < (maxTicks || 20); t++) {
    await sleep(50);
  }
  adapter.clearControlStates();
  return 'completed';
}

// ---- Macro-action implementations ----

function hasBlockCache(adapter) {
  // Check if adapter has a functional block cache with loaded chunks
  const test = adapter.blockAt({ x: 0, y: 64, z: 0 });
  // Actually, even Bedrock returns null for unloaded chunks.
  // Check if the adapter has a functional block cache by checking stats.
  return adapter._blockCache && adapter._blockCache.stats.chunksLoaded > 0;
}

/**
 * APPROACH_TARGET (20): A* to nearest hostile or player, stop at ~3 blocks.
 */
async function macroApproachTarget(adapter, trackingState, config) {
  const target = findNearestBySet(adapter, HOSTILE_MOBS, 32, true);
  if (!target) return 'no_target';

  const dist = distanceTo(adapter.position, target.position);
  if (dist <= 3.0) return 'completed';

  // No block cache: direct approach
  if (!hasBlockCache(adapter)) {
    return directApproach(adapter, target, trackingState, {
      maxTicks: 40, stopDist: 3.0, abortOnDamage: true,
    });
  }

  const start = floorPos(adapter.position);
  const goal = approachGoal(adapter.position, target.position, 2.5);
  const path = findPath(adapter, start, goal, { maxDistance: 32 });
  if (!path) return 'no_path';

  return executePath(adapter, path, trackingState, {
    maxTicks: 40, goalRadius: 3.0, abortOnDamage: true,
  });
}

/**
 * FLEE (21): Sprint away from nearest threat.
 */
async function macroFlee(adapter, trackingState, config) {
  const threat = findNearestBySet(adapter, HOSTILE_MOBS, 16, true);
  if (!threat) return 'no_target';

  const pos = adapter.position;
  const dx = pos.x - threat.position.x;
  const dz = pos.z - threat.position.z;
  const dist = Math.sqrt(dx * dx + dz * dz);

  // No block cache: direct flee
  if (!hasBlockCache(adapter)) {
    return directFlee(adapter, threat.position, trackingState, 20);
  }

  if (dist < 0.01) {
    // On top of threat — just sprint in a random-ish direction
    return directFlee(adapter, threat.position, trackingState, 20);
  }

  // Flee goal: 10 blocks in opposite direction
  const nx = dx / dist;
  const nz = dz / dist;
  const fleeGoal = {
    x: Math.floor(pos.x + nx * 10),
    y: Math.floor(pos.y),
    z: Math.floor(pos.z + nz * 10),
  };

  const start = floorPos(pos);
  const path = findPath(adapter, start, fleeGoal, { maxDistance: 16 });
  if (!path || path.length === 0) {
    return directFlee(adapter, threat.position, trackingState, 20);
  }

  return executePath(adapter, path, trackingState, {
    maxTicks: 40, abortOnDamage: false, // Keep running even if hit
  });
}

/**
 * MINE_BLOCK_BELOW (22): Look down and break the block at feet level.
 */
async function macroMineBlockBelow(adapter, trackingState, config) {
  const pos = adapter.position;
  const bx = Math.floor(pos.x);
  const by = Math.floor(pos.y) - 1;
  const bz = Math.floor(pos.z);

  const block = adapter.blockAt({ x: bx, y: by, z: bz });
  if (!block || block.name === 'air' || block.name === 'cave_air' || block.name === 'bedrock') {
    return 'no_target';
  }

  // Look straight down
  await adapter.look(adapter.yaw, Math.PI / 2);

  const startHealth = adapter.health;
  const isJava = config.protocol === 'java';
  const loc = { x: bx, y: by, z: bz };

  // Start digging
  if (isJava) {
    if (adapter._upstream) {
      try { adapter._upstream.write('block_dig', { status: 0, location: loc, face: 1 }); }
      catch (_) {}
    }
  } else {
    if (adapter._player && adapter._runtimeEntityId) {
      try {
        adapter._player.upstream.queue('player_action', {
          runtime_entity_id: adapter._runtimeEntityId,
          action: 'start_break',
          position: loc,
          result_position: { x: 0, y: 0, z: 0 },
          face: 1,
        });
      } catch (_) {}
    }
  }

  // Swing for 30 ticks (~1.5s), abort if damaged
  for (let tick = 0; tick < 30; tick++) {
    if (adapter.health < startHealth) {
      adapter.clearControlStates();
      // Abort
      if (isJava) {
        if (adapter._upstream) {
          try { adapter._upstream.write('block_dig', { status: 1, location: loc, face: 1 }); }
          catch (_) {}
        }
      } else {
        if (adapter._player && adapter._runtimeEntityId) {
          try {
            adapter._player.upstream.queue('player_action', {
              runtime_entity_id: adapter._runtimeEntityId,
              action: 'abort_break',
              position: loc,
              result_position: { x: 0, y: 0, z: 0 },
              face: 1,
            });
          } catch (_) {}
        }
      }
      return 'aborted';
    }
    try { await adapter.swingArm(); } catch (_) {}
    await sleep(50);
  }

  // Finish digging
  if (isJava) {
    if (adapter._upstream) {
      try { adapter._upstream.write('block_dig', { status: 2, location: loc, face: 1 }); }
      catch (_) {}
    }
  } else {
    if (adapter._player && adapter._runtimeEntityId) {
      try {
        adapter._player.upstream.queue('player_action', {
          runtime_entity_id: adapter._runtimeEntityId,
          action: 'stop_break',
          position: loc,
          result_position: { x: 0, y: 0, z: 0 },
          face: 1,
        });
      } catch (_) {}
    }
  }

  return 'completed';
}

/**
 * GO_TO_COORDINATES (23): A* to explicit coordinates (bridge-only command).
 */
async function macroGoToCoordinates(adapter, trackingState, config, args) {
  if (!args || args.x === undefined || args.z === undefined) return 'no_target';

  const goal = {
    x: Math.floor(args.x),
    y: Math.floor(args.y !== undefined ? args.y : adapter.position.y),
    z: Math.floor(args.z),
  };
  const start = floorPos(adapter.position);

  if (!hasBlockCache(adapter)) {
    // Direct walk toward coordinates
    const pseudoTarget = { position: { x: goal.x + 0.5, y: goal.y, z: goal.z + 0.5 } };
    return directApproach(adapter, pseudoTarget, trackingState, {
      maxTicks: 60, stopDist: 1.5, abortOnDamage: true,
    });
  }

  const path = findPath(adapter, start, goal, { maxDistance: 32 });
  if (!path) return 'no_path';

  return executePath(adapter, path, trackingState, {
    maxTicks: 60,
    goalRadius: 1.5,
    abortOnDamage: true,
  });
}

/**
 * APPROACH_PASSIVE (24): A* to nearest passive mob, stop at ~3 blocks.
 */
async function macroApproachPassive(adapter, trackingState, config) {
  const target = findNearestBySet(adapter, PASSIVE_MOBS, 32, false);
  if (!target) return 'no_target';

  const dist = distanceTo(adapter.position, target.position);
  if (dist <= 3.0) return 'completed';

  if (!hasBlockCache(adapter)) {
    return directApproach(adapter, target, trackingState, {
      maxTicks: 40, stopDist: 3.0, abortOnDamage: true,
    });
  }

  const start = floorPos(adapter.position);
  const goal = approachGoal(adapter.position, target.position, 2.5);
  const path = findPath(adapter, start, goal, { maxDistance: 32 });
  if (!path) return 'no_path';

  return executePath(adapter, path, trackingState, {
    maxTicks: 40, goalRadius: 3.0, abortOnDamage: true,
  });
}

// ---- Main dispatcher ----

/**
 * Execute a macro-action by ID.
 *
 * @param {number} macroId - 20-24
 * @param {object} adapter
 * @param {object} trackingState
 * @param {object} config - { protocol, tickRate, stealth, ... }
 * @param {object} [macroArgs] - Extra args for GO_TO_COORDINATES
 * @returns {Promise<string>} Status: "completed"|"aborted"|"timeout"|"no_path"|"no_target"
 */
async function executeMacro(macroId, adapter, trackingState, config, macroArgs) {
  switch (macroId) {
    case MACRO_IDS.APPROACH_TARGET:
      return macroApproachTarget(adapter, trackingState, config);
    case MACRO_IDS.FLEE:
      return macroFlee(adapter, trackingState, config);
    case MACRO_IDS.MINE_BLOCK_BELOW:
      return macroMineBlockBelow(adapter, trackingState, config);
    case MACRO_IDS.GO_TO_COORDINATES:
      return macroGoToCoordinates(adapter, trackingState, config, macroArgs || {});
    case MACRO_IDS.APPROACH_PASSIVE:
      return macroApproachPassive(adapter, trackingState, config);
    default:
      console.warn(`[macro] Unknown macro ID: ${macroId}`);
      return 'no_target';
  }
}

module.exports = { MACRO_IDS, executeMacro, findPath, isWalkable };
