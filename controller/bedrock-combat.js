/**
 * Bedrock-specific combat utilities.
 *
 * Key differences from Java Edition PvP:
 * - NO attack cooldown: every hit does full damage (no 1.9+ cooldown mechanic)
 * - Crit-spam: jump + attack continuously while airborne = crits
 * - CPS (clicks per second) matters more than timing precision
 * - No sweeping edge attack
 * - Shield disable with axe works (added 1.18.30)
 * - Reach is 3.0 blocks (vs Java's 3.0-3.5 depending on version)
 *
 * ANTI-CHEAT SAFE: All values tuned for Bedrock Realms.
 * - CPS capped at 8-10 (Realms kicks at ~12-15 CPS)
 * - Attack interval minimum 105-135ms (with jitter)
 * - Reach limited to 3.0 blocks
 * - Jump rate follows natural physics (~590ms per jump cycle)
 * - No impossible packet sequences
 */

const { HOSTILE_MOBS, PASSIVE_MOBS } = require('./categories');
const { sleep, waitTicks } = require('./utils');

// Bedrock Realms-safe combat config
const BEDROCK_COMBAT = {
  attackRange: 3.0,          // Bedrock reach (lower than Java 3.5)
  critRange: 2.5,            // Closer for crit reliability
  // CPS limits: 8-10 is human-like, Realms kicks at ~12-15
  minAttackIntervalMs: 105,  // ~9.5 CPS max (with jitter → effective 8-9 CPS)
  maxAttackIntervalMs: 135,  // ~7.4 CPS min
  // Jump timing (natural Bedrock physics)
  jumpCycleTicks: 12,        // Full jump cycle: ~600ms (jump + land)
  critWindowTicks: 4,        // Ticks after apex where crit registers (falling)
  // W-tap timing
  sprintCancelTicks: 2,      // 2 ticks release (safer than 1 for Bedrock)
  sprintReengageTicks: 2,    // 2 ticks to re-engage before hit
  // Multi-hit: how many attacks in a combo sequence
  comboHitsMax: 3,           // Max hits per combat action (keeps CPS in check)
  shieldDisableEnabled: true,
};

/**
 * Get random attack delay within Realms-safe CPS range.
 * Adds human-like jitter so timing isn't perfectly mechanical.
 */
function getAttackInterval() {
  const { minAttackIntervalMs, maxAttackIntervalMs } = BEDROCK_COMBAT;
  // Uniform random in safe range + slight Gaussian-ish jitter
  const base = minAttackIntervalMs + Math.random() * (maxAttackIntervalMs - minAttackIntervalMs);
  const jitter = (Math.random() - 0.5) * 20; // ±10ms noise
  return Math.max(minAttackIntervalMs, Math.round(base + jitter));
}

/**
 * Get AABB for a Bedrock entity.
 * Same as Java version but used with Bedrock's 3.0 range.
 */
function getEntityAABB(entity) {
  const pos = entity.position;
  if (!pos) return null;
  const width = entity.width || 0.6;
  const height = entity.height || 1.8;
  const halfWidth = width / 2;
  return {
    minX: pos.x - halfWidth,
    minY: pos.y,
    minZ: pos.z - halfWidth,
    maxX: pos.x + halfWidth,
    maxY: pos.y + height,
    maxZ: pos.z + halfWidth,
  };
}

/**
 * Get eye position for look-at targeting.
 */
function getEyePosition(entity) {
  const pos = entity.position;
  if (!pos) return null;
  const eyeHeight = entity.type === 'player' ? 1.62 : (entity.height || 1.8) * 0.85;
  return { x: pos.x, y: pos.y + eyeHeight, z: pos.z };
}

/**
 * Distance from bot to nearest point on entity AABB.
 */
function getAABBDistance(adapter, entity) {
  if (!entity || !entity.position) return Infinity;
  const botPos = adapter.position;
  const eyePos = { x: botPos.x, y: botPos.y + 1.62, z: botPos.z };
  const aabb = getEntityAABB(entity);
  if (!aabb) return Infinity;

  const cx = Math.max(aabb.minX, Math.min(eyePos.x, aabb.maxX));
  const cy = Math.max(aabb.minY, Math.min(eyePos.y, aabb.maxY));
  const cz = Math.max(aabb.minZ, Math.min(eyePos.z, aabb.maxZ));

  const dx = eyePos.x - cx;
  const dy = eyePos.y - cy;
  const dz = eyePos.z - cz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Check if target is shielding (same logic as Java, works on Bedrock 1.18.30+).
 */
function isShielding(target) {
  if (!target || !target.metadata) return false;
  const handState = target.metadata[8];
  if (typeof handState === 'number' && (handState & 0x01)) {
    if (target.equipment && target.equipment[1]) {
      const offhand = target.equipment[1];
      if (offhand && offhand.name && offhand.name.includes('shield')) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Find nearest attackable entity within Bedrock's 3.0 block reach.
 */
function findNearestTargetBedrock(adapter, range = BEDROCK_COMBAT.attackRange) {
  const entities = adapter.allEntities;
  let best = null;
  let bestDist = Infinity;

  for (const entity of entities) {
    if (adapter.isSelf(entity)) continue;
    if (!entity.position) continue;

    if (entity.type === 'player') {
      // OK — attack players
    } else {
      const name = (entity.name || '').toLowerCase();
      if (!HOSTILE_MOBS.has(name) && !PASSIVE_MOBS.has(name)) continue;
    }

    const dist = getAABBDistance(adapter, entity);
    if (dist < range && dist < bestDist) {
      best = entity;
      bestDist = dist;
    }
  }

  return best;
}

/**
 * Execute a Bedrock attack: look at target + hit.
 * No cooldown management needed — every hit is full damage.
 * Tracks hit for reward signal but does NOT set attackCooldown.
 */
async function bedrockAttack(adapter, target, trackingState) {
  const eyePos = getEyePosition(target);
  if (eyePos) {
    await adapter.lookAt(eyePos);
  }
  await adapter.attack(target);
  trackingState.lastAttackLanded = true;
  // NO cooldown on Bedrock — this is the key difference
  trackingState.attackCooldown = 0;
  if (target.type === 'player') {
    trackingState.lastPlayerHitLanded = true;
  }
  trackingState.attackedEntities.add(target.id || target.runtimeId);
}

/**
 * Execute Bedrock basic attack.
 * Spam-hits the target 1-3 times within safe CPS limits.
 * This is the Bedrock equivalent of Java's single timed attack.
 *
 * @param {object} adapter - Bot adapter (BedrockBot)
 * @param {object} target - Target entity
 * @param {object} trackingState - Shared tracking state
 * @param {number} tickRate - Server tick rate
 */
async function executeBedrockAttack(adapter, target, trackingState, tickRate) {
  // Shield disable takes priority (same as Java)
  if (isShielding(target)) {
    await executeBedrockShieldDisable(adapter, target, trackingState, tickRate);
    return;
  }

  // Bedrock: hit 1-2 times with safe CPS interval
  // (keeping combo short so each "attack action" from FPI is quick)
  const hits = 1 + Math.floor(Math.random() * 2); // 1-2 hits per action

  for (let i = 0; i < hits; i++) {
    const dist = getAABBDistance(adapter, target);
    if (dist > BEDROCK_COMBAT.attackRange) break; // Target moved out of range

    try {
      await bedrockAttack(adapter, target, trackingState);
    } catch (_) { break; }

    if (i < hits - 1) {
      // Wait between hits (Realms-safe interval)
      await sleep(getAttackInterval());
    }
  }
}

/**
 * Execute Bedrock crit-spam.
 * The dominant Bedrock PvP technique: continuously jump + attack.
 * Every hit while airborne = critical hit (1.5x damage).
 *
 * Timing (Realms-safe):
 * - Jump → wait 3-4 ticks (reach apex/start falling) → attack
 * - Land → immediately jump again → repeat
 * - Total cycle: ~12 ticks (600ms) with 1-2 hits per jump
 *
 * This gives ~3-4 crits per second (well under CPS limits).
 */
async function executeBedrockCritSpam(adapter, target, trackingState, tickRate) {
  // Sprint toward target
  adapter.setControlState('forward', true);
  adapter.setControlState('sprint', true);

  // One crit-spam cycle: jump + hit while airborne
  // Jump
  adapter.setControlState('jump', true);
  await waitTicks(tickRate, 1);
  adapter.setControlState('jump', false);

  // Wait for rising phase (2-3 ticks) — still counts as crit if not on ground
  await waitTicks(tickRate, 3);

  // Attack during airborne phase (= critical hit)
  const dist = getAABBDistance(adapter, target);
  if (dist <= BEDROCK_COMBAT.critRange + 0.5) {
    try {
      await bedrockAttack(adapter, target, trackingState);
    } catch (_) {}

    // Optional second hit on the way down (safe CPS)
    await sleep(getAttackInterval());
    const dist2 = getAABBDistance(adapter, target);
    if (dist2 <= BEDROCK_COMBAT.attackRange) {
      try {
        await bedrockAttack(adapter, target, trackingState);
      } catch (_) {}
    }
  }

  // Wait for landing
  await waitTicks(tickRate, 5);
  adapter.clearControlStates();
}

/**
 * Execute Bedrock W-tap.
 * Sprint-cancel → re-engage → multi-hit with fresh sprint momentum.
 * Same concept as Java but faster cycle since no cooldown.
 *
 * Gives fresh sprint knockback on the first hit after re-engage,
 * then follows up with 1-2 more hits at safe CPS.
 */
async function executeBedrockWTap(adapter, target, trackingState, tickRate) {
  // Release sprint (the "cancel")
  adapter.setControlState('forward', false);
  adapter.setControlState('sprint', false);
  await waitTicks(tickRate, BEDROCK_COMBAT.sprintCancelTicks);

  // Re-engage sprint (fresh sprint = more knockback on next hit)
  adapter.setControlState('forward', true);
  adapter.setControlState('sprint', true);
  await waitTicks(tickRate, BEDROCK_COMBAT.sprintReengageTicks);

  // Attack with fresh sprint momentum (first hit gets bonus knockback)
  const dist = getAABBDistance(adapter, target);
  if (dist <= BEDROCK_COMBAT.attackRange) {
    try {
      await bedrockAttack(adapter, target, trackingState);
    } catch (_) {}

    // Follow-up hit (still sprinting = more knockback chain)
    await sleep(getAttackInterval());
    const dist2 = getAABBDistance(adapter, target);
    if (dist2 <= BEDROCK_COMBAT.attackRange) {
      try {
        await bedrockAttack(adapter, target, trackingState);
      } catch (_) {}
    }
  }

  // Continue sprinting briefly
  await waitTicks(tickRate, 3);
  adapter.clearControlStates();
}

/**
 * Execute shield disable on Bedrock.
 * Same as Java: switch to axe → hit → switch back.
 * Works on Bedrock 1.18.30+.
 */
async function executeBedrockShieldDisable(adapter, target, trackingState, tickRate) {
  const items = adapter.inventoryItems;
  let axeSlot = -1;
  const originalSlot = adapter.quickBarSlot;

  for (let i = 0; i < items.length; i++) {
    const name = items[i]?.name || '';
    if (name.includes('_axe')) {
      axeSlot = i;
      break;
    }
  }

  if (axeSlot < 0) {
    // No axe — just attack normally
    try {
      await bedrockAttack(adapter, target, trackingState);
    } catch (_) {}
    return;
  }

  // Switch to axe
  if (axeSlot < 9) {
    adapter.setQuickBarSlot(axeSlot);
  }
  await waitTicks(tickRate, 2); // Slightly longer slot switch for Bedrock safety

  // Attack with axe (disables shield)
  try {
    await bedrockAttack(adapter, target, trackingState);
  } catch (_) {}

  // Switch back
  await waitTicks(tickRate, 2);
  adapter.setQuickBarSlot(originalSlot);
}

module.exports = {
  BEDROCK_COMBAT,
  getAttackInterval,
  getEntityAABB,
  getEyePosition,
  getAABBDistance,
  isShielding,
  findNearestTargetBedrock,
  executeBedrockAttack,
  executeBedrockCritSpam,
  executeBedrockWTap,
  executeBedrockShieldDisable,
};
