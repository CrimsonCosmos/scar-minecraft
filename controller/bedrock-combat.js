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
const { getEntityAABB, getEyePosition, getAABBDistance, isShielding } = require('./combat-utils');

// Bedrock Realms-safe combat config
const BEDROCK_COMBAT = {
  attackRange: 3.0,          // Bedrock reach (lower than Java 3.5)
  critRange: 2.5,            // Closer for crit reliability
  // CPS limits: wider range mimics real human variance
  // Realms kicks at ~12-15 CPS, so min interval stays >= 95ms (~10.5 CPS)
  minAttackIntervalMs: 95,   // ~10.5 CPS hard floor (safe margin)
  maxAttackIntervalMs: 195,  // ~5.1 CPS (slow end — humans vary widely)
  // Jump timing (natural Bedrock physics)
  jumpCycleTicks: 12,        // Full jump cycle: ~600ms (jump + land)
  critWindowStartTick: 2,    // Earliest tick to attack after jump (rising)
  critWindowEndTick: 5,      // Latest tick to attack (falling)
  // W-tap timing
  sprintCancelTicks: 2,      // 2 ticks release (safer than 1 for Bedrock)
  sprintReengageTicks: 2,    // 2 ticks to re-engage before hit
  // Multi-hit: how many attacks in a combo sequence
  comboHitsMax: 3,           // Max hits per combat action (keeps CPS in check)
  shieldDisableEnabled: true,
};

/**
 * Get random attack delay within Realms-safe CPS range.
 * Uses sum-of-uniforms (triangular-ish) distribution that clusters around
 * the center (~145ms / ~7 CPS) with natural tails, defeating statistical
 * profiling that detects narrow uniform distributions.
 */
function getAttackInterval() {
  const { minAttackIntervalMs, maxAttackIntervalMs } = BEDROCK_COMBAT;
  // Sum of 2 uniforms → triangular distribution (clusters around center)
  const u1 = Math.random();
  const u2 = Math.random();
  const t = (u1 + u2) / 2; // 0-1, peaked at 0.5
  const base = minAttackIntervalMs + t * (maxAttackIntervalMs - minAttackIntervalMs);

  // 4% chance of micro-pause (human hesitation / repositioning)
  const pause = (Math.random() < 0.04) ? 30 + Math.random() * 80 : 0;

  return Math.max(minAttackIntervalMs, Math.round(base + pause));
}

// Combat geometry utils imported from combat-utils.js

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

  // Wait for airborne phase — vary between 2-4 ticks (not always the same)
  const critWaitTicks = BEDROCK_COMBAT.critWindowStartTick +
    Math.floor(Math.random() * (BEDROCK_COMBAT.critWindowEndTick - BEDROCK_COMBAT.critWindowStartTick + 1));
  await waitTicks(tickRate, critWaitTicks);

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
