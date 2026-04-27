/**
 * Java Edition combat utilities.
 *
 * Key differences from Bedrock PvP:
 * - 1.9+ attack cooldown: sword = 625ms (1.6/sec), axe = 1000ms (1.0/sec)
 * - Crits: must be falling (past apex), NOT on ground, cooldown >= 90% charged
 * - Sweeping edge: sword hits nearby mobs (1.9+, not explicitly handled here)
 * - Shield disable: axe attack disables shield for 5 seconds (100 ticks)
 * - Reach: 3.0 blocks (same as Bedrock)
 * - W-tap: 1-tick sprint cancel works in Java (faster than Bedrock's 2-tick)
 *
 * Supports two PvP styles via --pvp-style flag:
 * - 'cooldown' (default, 1.9+): single well-timed hits with full cooldown
 * - 'spam' (1.8 servers): click-spam like Bedrock, no cooldown
 */

const { HOSTILE_MOBS, PASSIVE_MOBS } = require('./categories');
const { sleep, waitTicks } = require('./utils');
const { getEntityAABB, getEyePosition, getAABBDistance, isShielding } = require('./combat-utils');

const JAVA_COMBAT = {
  attackRange: 3.0,
  critRange: 2.5,
  // 1.9+ cooldown
  swordCooldownMs: 625,     // 1.6 attacks/sec
  axeCooldownMs: 1000,      // 1.0 attacks/sec
  // 1.8 spam timing — wider range to defeat profiling
  minAttackIntervalMs: 95,
  maxAttackIntervalMs: 195,
  // Crit timing — varied at runtime
  critAscentTicksMin: 5,     // Earliest tick to attack after jump
  critAscentTicksMax: 8,     // Latest tick (gives 5, 6, 7, or 8)
  // W-tap (faster than Bedrock)
  sprintCancelTicks: 1,
  sprintReengageTicks: 1,
  comboHitsMax: 1,           // 1 hit per cooldown cycle (1.9+)
  shieldDisableEnabled: true,
  shieldDisableTicks: 100,   // 5 seconds
};

/**
 * Get attack delay based on PvP style.
 * @param {string} pvpStyle - 'cooldown' or 'spam'
 */
function getAttackInterval(pvpStyle) {
  if (pvpStyle === 'spam') {
    const { minAttackIntervalMs, maxAttackIntervalMs } = JAVA_COMBAT;
    // Triangular distribution (clusters around center, wider than uniform)
    const u1 = Math.random();
    const u2 = Math.random();
    const t = (u1 + u2) / 2;
    const base = minAttackIntervalMs + t * (maxAttackIntervalMs - minAttackIntervalMs);
    const pause = (Math.random() < 0.04) ? 30 + Math.random() * 80 : 0;
    return Math.max(minAttackIntervalMs, Math.round(base + pause));
  }
  // 1.9+: full sword cooldown + wider human-like jitter (±75ms)
  // Sum of 3 uniforms → roughly bell-shaped around center
  const u1 = Math.random();
  const u2 = Math.random();
  const u3 = Math.random();
  const jitter = ((u1 + u2 + u3) / 3 - 0.5) * 150; // ±75ms, bell-shaped
  const pause = (Math.random() < 0.03) ? 40 + Math.random() * 60 : 0;
  return Math.max(JAVA_COMBAT.swordCooldownMs * 0.92, Math.round(JAVA_COMBAT.swordCooldownMs + jitter + pause));
}

// Combat geometry utils imported from combat-utils.js

function findNearestTargetJava(adapter, range) {
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
 * Single attack: look at target + hit.
 * Sets cooldown ticks for 1.9+ mode.
 */
async function javaAttack(adapter, target, trackingState, pvpStyle) {
  const eyePos = getEyePosition(target);
  if (eyePos) await adapter.lookAt(eyePos);
  await adapter.attack(target);
  trackingState.lastAttackLanded = true;
  if (pvpStyle === 'spam') {
    trackingState.attackCooldown = 0;
  } else {
    trackingState.attackCooldown = Math.ceil(JAVA_COMBAT.swordCooldownMs / 50);
  }
  if (target.type === 'player') {
    trackingState.lastPlayerHitLanded = true;
  }
  trackingState.attackedEntities.add(target.id || target.runtimeId);
}

/**
 * Basic attack. Respects 1.9+ cooldown or spams in 1.8 mode.
 */
async function executeJavaAttack(adapter, target, trackingState, tickRate, pvpStyle) {
  if (isShielding(target)) {
    await executeJavaShieldDisable(adapter, target, trackingState, tickRate, pvpStyle);
    return;
  }

  if (pvpStyle === 'spam') {
    // 1.8 style: spam hits
    const hits = 1 + Math.floor(Math.random() * 2);
    for (let i = 0; i < hits; i++) {
      const dist = getAABBDistance(adapter, target);
      if (dist > JAVA_COMBAT.attackRange) break;
      try { await javaAttack(adapter, target, trackingState, pvpStyle); }
      catch (_) { break; }
      if (i < hits - 1) await sleep(getAttackInterval('spam'));
    }
  } else {
    // 1.9+: wait for full cooldown, then single precise hit
    if (trackingState.attackCooldown > 0) {
      await sleep(trackingState.attackCooldown * 50);
    }
    const dist = getAABBDistance(adapter, target);
    if (dist <= JAVA_COMBAT.attackRange) {
      try { await javaAttack(adapter, target, trackingState, pvpStyle); }
      catch (_) {}
    }
  }
}

/**
 * Java crit attack: jump → wait for fall (past apex) → attack.
 * In 1.9+, must have full cooldown to deal crit damage.
 */
async function executeJavaCritSpam(adapter, target, trackingState, tickRate, pvpStyle) {
  adapter.setControlState('forward', true);
  adapter.setControlState('sprint', true);

  // Jump
  adapter.setControlState('jump', true);
  await waitTicks(tickRate, 1);
  adapter.setControlState('jump', false);

  // Wait to reach apex and start falling — vary 5-8 ticks (not always the same)
  const critTicks = JAVA_COMBAT.critAscentTicksMin +
    Math.floor(Math.random() * (JAVA_COMBAT.critAscentTicksMax - JAVA_COMBAT.critAscentTicksMin + 1));
  await waitTicks(tickRate, critTicks);

  // Attack while falling = critical hit
  const dist = getAABBDistance(adapter, target);
  if (dist <= JAVA_COMBAT.critRange + 0.5) {
    if (pvpStyle !== 'spam' && trackingState.attackCooldown > 0) {
      await sleep(trackingState.attackCooldown * 50);
    }
    try { await javaAttack(adapter, target, trackingState, pvpStyle); }
    catch (_) {}
  }

  // Wait for landing
  await waitTicks(tickRate, 6);
  adapter.clearControlStates();
}

/**
 * Java W-tap: sprint cancel → re-engage → attack with fresh sprint.
 * Java allows 1-tick cancel (faster than Bedrock's 2-tick).
 */
async function executeJavaWTap(adapter, target, trackingState, tickRate, pvpStyle) {
  // Release sprint
  adapter.setControlState('forward', false);
  adapter.setControlState('sprint', false);
  await waitTicks(tickRate, JAVA_COMBAT.sprintCancelTicks);

  // Re-engage
  adapter.setControlState('forward', true);
  adapter.setControlState('sprint', true);
  await waitTicks(tickRate, JAVA_COMBAT.sprintReengageTicks);

  // Attack with fresh sprint momentum
  const dist = getAABBDistance(adapter, target);
  if (dist <= JAVA_COMBAT.attackRange) {
    if (pvpStyle !== 'spam' && trackingState.attackCooldown > 0) {
      await sleep(trackingState.attackCooldown * 50);
    }
    try { await javaAttack(adapter, target, trackingState, pvpStyle); }
    catch (_) {}

    if (pvpStyle === 'spam') {
      await sleep(getAttackInterval('spam'));
      const dist2 = getAABBDistance(adapter, target);
      if (dist2 <= JAVA_COMBAT.attackRange) {
        try { await javaAttack(adapter, target, trackingState, pvpStyle); }
        catch (_) {}
      }
    }
  }

  await waitTicks(tickRate, 3);
  adapter.clearControlStates();
}

/**
 * Shield disable: switch to axe → attack → switch back.
 * Disables shield for 5 seconds (100 ticks) in Java 1.9+.
 */
async function executeJavaShieldDisable(adapter, target, trackingState, tickRate, pvpStyle) {
  const items = adapter.inventoryItems;
  let axeSlot = -1;
  const originalSlot = adapter.quickBarSlot;

  for (let i = 0; i < items.length; i++) {
    const name = items[i]?.name || '';
    if (name.includes('_axe')) { axeSlot = i; break; }
  }

  if (axeSlot < 0) {
    try { await javaAttack(adapter, target, trackingState, pvpStyle); }
    catch (_) {}
    return;
  }

  if (axeSlot < 9) adapter.setQuickBarSlot(axeSlot);
  await waitTicks(tickRate, 1);

  try { await javaAttack(adapter, target, trackingState, pvpStyle); }
  catch (_) {}

  await waitTicks(tickRate, 1);
  adapter.setQuickBarSlot(originalSlot);
}

module.exports = {
  JAVA_COMBAT,
  getAttackInterval,
  getEntityAABB,
  getEyePosition,
  getAABBDistance,
  isShielding,
  findNearestTargetJava,
  executeJavaAttack,
  executeJavaCritSpam,
  executeJavaWTap,
  executeJavaShieldDisable,
};
