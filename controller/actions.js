/**
 * Action execution — all 20 discrete actions for Relay adapter.
 *
 * Supports both Bedrock and Java protocols. Uses config.protocol to
 * dispatch to the appropriate combat functions.
 *
 * The adapter interface is identical across protocols, so movement,
 * look, and utility actions work unchanged. Only combat actions
 * (11, 18, 19) and composite combat need protocol-specific dispatch.
 */

const { HOSTILE_MOBS, PASSIVE_MOBS, FOOD_NAMES } = require('./categories');
const { sleep, waitTicks } = require('./utils');
const {
  findNearestTargetBedrock, executeBedrockAttack,
  executeBedrockCritSpam, executeBedrockWTap, BEDROCK_COMBAT,
  getEyePosition,
} = require('./bedrock-combat');

// Lazy-load Java combat (only needed when protocol === 'java')
let _javaCombat = null;
function javaCombat() {
  if (!_javaCombat) _javaCombat = require('./java-combat');
  return _javaCombat;
}

// Lazy-load macro-actions (only needed for Phase 4)
let _macroActions = null;
function macroActions() {
  if (!_macroActions) _macroActions = require('./macro-actions');
  return _macroActions;
}

async function executeAction(adapter, actionId, trackingState, config) {
  const { tickRate = 0, actionDurationTicks = 4 } = config;
  const stealth = config.stealth || null;

  if (stealth) {
    await stealth.preActionDelay();
  }

  if (stealth && stealth.shouldRandomIdle()) {
    await sleep(stealth.getIdleDurationMs());
  }

  if (trackingState.knockbackCooldown > 0 && actionId <= 6) {
    adapter.clearControlStates();
    await waitTicks(tickRate, actionDurationTicks);
    return;
  }

  if (actionId <= 6) {
    adapter.clearControlStates();
  }
  trackingState.lastAttackLanded = false;
  trackingState.lastPlayerHitLanded = false;

  const durationTicks = stealth ? stealth.getActionTicks(actionDurationTicks) : actionDurationTicks;

  switch (actionId) {
    case 0: // Forward
      adapter.setControlState('forward', true);
      break;
    case 1: // Backward
      adapter.setControlState('back', true);
      break;
    case 2: // Strafe left
      adapter.setControlState('left', true);
      break;
    case 3: // Strafe right
      adapter.setControlState('right', true);
      break;
    case 4: // Jump
      adapter.setControlState('jump', true);
      break;
    case 5: // Forward + jump
      adapter.setControlState('forward', true);
      adapter.setControlState('jump', true);
      break;
    case 6: // Sprint forward
      adapter.setControlState('forward', true);
      adapter.setControlState('sprint', true);
      break;
    case 7: { // Look left 45 deg
      const targetYaw = adapter.yaw + 45;
      if (stealth) {
        const steps = stealth.getLookSteps(adapter.yaw, targetYaw);
        for (const yaw of steps) {
          await adapter.look(yaw, adapter.pitch);
          await sleep(30);
        }
      } else {
        await adapter.look(targetYaw, adapter.pitch);
      }
      break;
    }
    case 8: { // Look right 45 deg
      const targetYaw = adapter.yaw - 45;
      if (stealth) {
        const steps = stealth.getLookSteps(adapter.yaw, targetYaw);
        for (const yaw of steps) {
          await adapter.look(yaw, adapter.pitch);
          await sleep(30);
        }
      } else {
        await adapter.look(targetYaw, adapter.pitch);
      }
      break;
    }
    case 9: // Look up 30 deg
      await adapter.look(adapter.yaw, Math.max(-90, adapter.pitch - 30));
      break;
    case 10: // Look down 30 deg
      await adapter.look(adapter.yaw, Math.min(90, adapter.pitch + 30));
      break;
    case 11: { // Attack nearest entity
      if (config.protocol === 'java') {
        const jc = javaCombat();
        const target = jc.findNearestTargetJava(adapter, jc.JAVA_COMBAT.attackRange);
        if (target) {
          await jc.executeJavaAttack(adapter, target, trackingState, tickRate, config.pvpStyle);
        } else {
          try { adapter.swingArm(); } catch (_) {}
        }
      } else {
        const target = findNearestTargetBedrock(adapter, BEDROCK_COMBAT.attackRange);
        if (target) {
          await executeBedrockAttack(adapter, target, trackingState, tickRate);
        } else {
          try { adapter.swingArm(); } catch (_) {}
        }
      }
      break;
    }
    case 12: // Idle / no-op
      break;
    case 13: { // Use item
      try { await adapter.activateItem(); } catch (_) {}
      break;
    }
    case 14: { // Select next hotbar slot
      const current = adapter.quickBarSlot;
      adapter.setQuickBarSlot((current + 1) % 9);
      break;
    }
    case 15: { // Select prev hotbar slot
      const current = adapter.quickBarSlot;
      adapter.setQuickBarSlot((current + 8) % 9);
      break;
    }
    case 16: { // Craft planks
      try { await adapter.craftPlanks(); } catch (_) {}
      break;
    }
    case 17: { // Craft sticks/pickaxe
      try { await adapter.craftToolOrSticks(); } catch (_) {}
      break;
    }
    case 18: { // Sprint-crit
      if (config.protocol === 'java') {
        const jc = javaCombat();
        const critTarget = jc.findNearestTargetJava(adapter, jc.JAVA_COMBAT.critRange + 1.0);
        if (critTarget) {
          await jc.executeJavaCritSpam(adapter, critTarget, trackingState, tickRate, config.pvpStyle);
        } else {
          adapter.setControlState('forward', true);
          adapter.setControlState('sprint', true);
          adapter.setControlState('jump', true);
          await waitTicks(tickRate, 2);
          adapter.setControlState('jump', false);
          await waitTicks(tickRate, durationTicks);
          adapter.clearControlStates();
        }
      } else {
        const critTarget = findNearestTargetBedrock(adapter, BEDROCK_COMBAT.critRange + 1.0);
        if (critTarget) {
          await executeBedrockCritSpam(adapter, critTarget, trackingState, tickRate);
        } else {
          adapter.setControlState('forward', true);
          adapter.setControlState('sprint', true);
          adapter.setControlState('jump', true);
          await waitTicks(tickRate, 2);
          adapter.setControlState('jump', false);
          await waitTicks(tickRate, durationTicks);
          adapter.clearControlStates();
        }
      }
      break;
    }
    case 19: { // W-tap
      if (config.protocol === 'java') {
        const jc = javaCombat();
        const wtapTarget = jc.findNearestTargetJava(adapter, jc.JAVA_COMBAT.attackRange + 0.5);
        if (wtapTarget) {
          await jc.executeJavaWTap(adapter, wtapTarget, trackingState, tickRate, config.pvpStyle);
        } else {
          adapter.setControlState('forward', false);
          adapter.setControlState('sprint', false);
          await waitTicks(tickRate, 1);
          adapter.setControlState('forward', true);
          adapter.setControlState('sprint', true);
          await waitTicks(tickRate, durationTicks);
          adapter.clearControlStates();
        }
      } else {
        const wtapTarget = findNearestTargetBedrock(adapter, BEDROCK_COMBAT.attackRange + 0.5);
        if (wtapTarget) {
          await executeBedrockWTap(adapter, wtapTarget, trackingState, tickRate);
        } else {
          adapter.setControlState('forward', false);
          adapter.setControlState('sprint', false);
          await waitTicks(tickRate, 2);
          adapter.setControlState('forward', true);
          adapter.setControlState('sprint', true);
          await waitTicks(tickRate, durationTicks);
          adapter.clearControlStates();
        }
      }
      break;
    }
    default:
      // Macro-actions: IDs 20+
      if (actionId >= 20 && actionId <= 24) {
        const ma = macroActions();
        const macroArgs = config._macroArgs || {};
        const status = await ma.executeMacro(actionId, adapter, trackingState, config, macroArgs);
        trackingState.lastMacroStatus = status;
      } else {
        console.warn(`[actions] Unknown action: ${actionId}`);
      }
  }

  if (actionId <= 6) {
    await waitTicks(tickRate, durationTicks);
    adapter.clearControlStates();
  } else {
    await waitTicks(tickRate, durationTicks);
  }
}

// ---- Composite action (movement + look + combat in parallel) ----

function _applyMovement(adapter, movement) {
  adapter.clearControlStates();
  switch (movement) {
    case 0: break;
    case 1: adapter.setControlState('forward', true); break;
    case 2: adapter.setControlState('back', true); break;
    case 3: adapter.setControlState('left', true); break;
    case 4: adapter.setControlState('right', true); break;
    case 5:
      adapter.setControlState('forward', true);
      adapter.setControlState('jump', true);
      break;
    case 6:
      adapter.setControlState('forward', true);
      adapter.setControlState('sprint', true);
      break;
  }
}

async function _executeLookAxis(adapter, look, stealth, config) {
  switch (look) {
    case 0: break;
    case 1: { // track target
      let target;
      let eyePosFn = getEyePosition;
      if (config && config.protocol === 'java') {
        const jc = javaCombat();
        target = jc.findNearestTargetJava(adapter, 16);
        eyePosFn = jc.getEyePosition;
      } else {
        target = findNearestTargetBedrock(adapter, 16);
      }
      if (target) {
        const eyePos = eyePosFn(target);
        if (eyePos) await adapter.lookAt(eyePos);
      }
      break;
    }
    case 2: { // look left
      const targetYaw = adapter.yaw + 45;
      if (stealth) {
        for (const yaw of stealth.getLookSteps(adapter.yaw, targetYaw)) {
          await adapter.look(yaw, adapter.pitch);
          await sleep(30);
        }
      } else {
        await adapter.look(targetYaw, adapter.pitch);
      }
      break;
    }
    case 3: { // look right
      const targetYaw = adapter.yaw - 45;
      if (stealth) {
        for (const yaw of stealth.getLookSteps(adapter.yaw, targetYaw)) {
          await adapter.look(yaw, adapter.pitch);
          await sleep(30);
        }
      } else {
        await adapter.look(targetYaw, adapter.pitch);
      }
      break;
    }
    case 4:
      await adapter.look(adapter.yaw, Math.max(-90, adapter.pitch - 30));
      break;
    case 5:
      await adapter.look(adapter.yaw, Math.min(90, adapter.pitch + 30));
      break;
  }
}

async function _executeAttackAxis(adapter, trackingState, tickRate, config) {
  if (config && config.protocol === 'java') {
    const jc = javaCombat();
    const target = jc.findNearestTargetJava(adapter, jc.JAVA_COMBAT.attackRange);
    if (target) {
      await jc.executeJavaAttack(adapter, target, trackingState, tickRate, config.pvpStyle);
    } else {
      try { adapter.swingArm(); } catch (_) {}
    }
  } else {
    const target = findNearestTargetBedrock(adapter, BEDROCK_COMBAT.attackRange);
    if (target) {
      await executeBedrockAttack(adapter, target, trackingState, tickRate);
    } else {
      try { adapter.swingArm(); } catch (_) {}
    }
  }
}

async function executeCompositeAction(adapter, movement, look, combat, trackingState, config) {
  const { tickRate = 0, actionDurationTicks = 4 } = config;
  const stealth = config.stealth || null;

  if (stealth) await stealth.preActionDelay();
  if (stealth && stealth.shouldRandomIdle()) {
    await sleep(stealth.getIdleDurationMs());
  }

  const durationTicks = stealth ? stealth.getActionTicks(actionDurationTicks) : actionDurationTicks;

  trackingState.lastAttackLanded = false;
  trackingState.lastPlayerHitLanded = false;

  const knockbackActive = trackingState.knockbackCooldown > 0;

  // Motor program overrides (crit/wtap consume whole tick)
  if (combat === 2) {
    adapter.clearControlStates();
    if (config.protocol === 'java') {
      const jc = javaCombat();
      const target = jc.findNearestTargetJava(adapter, jc.JAVA_COMBAT.critRange + 1.0);
      if (target) {
        await jc.executeJavaCritSpam(adapter, target, trackingState, tickRate, config.pvpStyle);
      } else {
        adapter.setControlState('forward', true);
        adapter.setControlState('sprint', true);
        adapter.setControlState('jump', true);
        await waitTicks(tickRate, 2);
        adapter.setControlState('jump', false);
        await waitTicks(tickRate, durationTicks);
        adapter.clearControlStates();
      }
    } else {
      const target = findNearestTargetBedrock(adapter, BEDROCK_COMBAT.critRange + 1.0);
      if (target) {
        await executeBedrockCritSpam(adapter, target, trackingState, tickRate);
      } else {
        adapter.setControlState('forward', true);
        adapter.setControlState('sprint', true);
        adapter.setControlState('jump', true);
        await waitTicks(tickRate, 2);
        adapter.setControlState('jump', false);
        await waitTicks(tickRate, durationTicks);
        adapter.clearControlStates();
      }
    }
    await _executeLookAxis(adapter, look, stealth, config);
    return;
  }

  if (combat === 3) {
    adapter.clearControlStates();
    if (config.protocol === 'java') {
      const jc = javaCombat();
      const target = jc.findNearestTargetJava(adapter, jc.JAVA_COMBAT.attackRange + 0.5);
      if (target) {
        await jc.executeJavaWTap(adapter, target, trackingState, tickRate, config.pvpStyle);
      } else {
        adapter.setControlState('forward', false);
        adapter.setControlState('sprint', false);
        await waitTicks(tickRate, 1);
        adapter.setControlState('forward', true);
        adapter.setControlState('sprint', true);
        await waitTicks(tickRate, durationTicks);
        adapter.clearControlStates();
      }
    } else {
      const target = findNearestTargetBedrock(adapter, BEDROCK_COMBAT.attackRange + 0.5);
      if (target) {
        await executeBedrockWTap(adapter, target, trackingState, tickRate);
      } else {
        adapter.setControlState('forward', false);
        adapter.setControlState('sprint', false);
        await waitTicks(tickRate, 2);
        adapter.setControlState('forward', true);
        adapter.setControlState('sprint', true);
        await waitTicks(tickRate, durationTicks);
        adapter.clearControlStates();
      }
    }
    await _executeLookAxis(adapter, look, stealth, config);
    return;
  }

  // Parallel: movement + look + combat primitive
  if (!knockbackActive) {
    _applyMovement(adapter, movement);
  } else {
    adapter.clearControlStates();
  }

  await _executeLookAxis(adapter, look, stealth);

  if (combat === 1) {
    await _executeAttackAxis(adapter, trackingState, tickRate, config);
  } else if (combat === 4) {
    // use_start: press and hold right-click (charge bow, raise shield, eat)
    try { await adapter.pressUseItem(); } catch (_) {}
  } else if (combat === 5) {
    // use_stop: release right-click (fire bow, lower shield)
    try { await adapter.releaseUseItem(); } catch (_) {}
  } else if (combat === 6) {
    // hotbar_next: cycle to next hotbar slot
    const current = adapter.quickBarSlot;
    adapter.setQuickBarSlot((current + 1) % 9);
  }

  await waitTicks(tickRate, durationTicks);
  adapter.clearControlStates();
}

// ---- Auto-eat ----

// Food items — single source of truth in categories.js
const FOOD_ITEMS = FOOD_NAMES;

async function tryAutoEat(adapter, trackingState) {
  if (adapter.food >= 2) return false;

  const entities = adapter.allEntities;
  const pos = adapter.position;
  for (const entity of entities) {
    if (adapter.isSelf(entity)) continue;
    if (!entity.position) continue;
    const name = (entity.name || entity.displayName || '').toLowerCase();
    if (entity.type === 'player' || HOSTILE_MOBS.has(name)) {
      const dx = pos.x - entity.position.x;
      const dy = pos.y - entity.position.y;
      const dz = pos.z - entity.position.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < 16) return false;
    }
  }

  if (trackingState.knockbackCooldown > 0) return false;

  const items = adapter.inventoryItems;

  // Find food in hotbar (slots 0-8) first, then anywhere in inventory
  let foodItem = items.find(i => FOOD_ITEMS.has(i.name) && i.slot !== undefined && i.slot <= 8);
  if (!foodItem) {
    foodItem = items.find(i => FOOD_ITEMS.has(i.name));
  }
  if (!foodItem) return false;

  try {
    const prevSlot = adapter.quickBarSlot;

    // Switch to the food slot if it's in the hotbar
    if (foodItem.slot !== undefined && foodItem.slot <= 8) {
      adapter.setQuickBarSlot(foodItem.slot);
      await sleep(100);
    }

    await adapter.activateItem();
    await sleep(1700);
    for (let i = 0; i < 10 && adapter.food < 20; i++) {
      const remaining = adapter.inventoryItems.find(it =>
        FOOD_ITEMS.has(it.name) && it.slot !== undefined && it.slot <= 8
      ) || adapter.inventoryItems.find(it => FOOD_ITEMS.has(it.name));
      if (!remaining) break;
      if (remaining.slot !== undefined && remaining.slot <= 8) {
        adapter.setQuickBarSlot(remaining.slot);
        await sleep(100);
      }
      await adapter.activateItem();
      await sleep(1700);
    }

    // Restore previous slot
    adapter.setQuickBarSlot(prevSlot);
  } catch (_) {}

  return true;
}

module.exports = { executeAction, executeCompositeAction, tryAutoEat };
