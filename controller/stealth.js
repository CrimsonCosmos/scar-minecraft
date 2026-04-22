/**
 * StealthEngine — humanizes bot behavior to evade anti-cheat.
 *
 * Activated with --stealth flag or stealth: true in config.
 * Adds timing jitter, smooth rotation, varied action durations,
 * and random idle insertions to break detectable patterns.
 */

const { sleep } = require('./utils');

class StealthEngine {
  constructor(config = {}) {
    this.enabled = config.enabled !== false;
    this.actionJitter = config.actionJitterMs || [30, 150];   // ms range for pre-action delay
    this.lookSteps = config.lookSteps || 3;                   // smooth rotation step count
    this.attackVariance = config.attackCooldownVariance || 0.2;
    this.idleChance = config.idleChance || 0.05;              // 5% random pauses
  }

  /**
   * Random delay before action execution (simulates human reaction time).
   */
  async preActionDelay() {
    if (!this.enabled) return;
    const [min, max] = this.actionJitter;
    await sleep(min + Math.random() * (max - min));
  }

  /**
   * Multi-step smooth rotation (humans don't snap-look).
   * Returns array of intermediate yaw values.
   */
  getLookSteps(currentYaw, targetYaw) {
    if (!this.enabled) return [targetYaw];
    const steps = [];
    for (let i = 1; i <= this.lookSteps; i++) {
      const t = i / this.lookSteps;
      // Add slight overshoot/undershoot for realism
      const jitter = (Math.random() - 0.5) * 0.05;
      steps.push(currentYaw + (targetYaw - currentYaw) * (t + jitter));
    }
    return steps;
  }

  /**
   * Vary action duration (don't hold keys for exact same time).
   */
  getActionTicks(baseTicks) {
    if (!this.enabled) return baseTicks;
    const variance = Math.floor(Math.random() * 3) - 1; // -1, 0, or +1
    return Math.max(2, baseTicks + variance);
  }

  /**
   * Should we insert a random idle? (breaks pattern detection).
   */
  shouldRandomIdle() {
    return this.enabled && Math.random() < this.idleChance;
  }

  /**
   * Attack cooldown with human-like variance.
   */
  getAttackDelay(baseCooldownMs) {
    if (!this.enabled) return baseCooldownMs;
    const variance = this.attackVariance;
    return baseCooldownMs * (1 + (Math.random() - 0.5) * 2 * variance) + Math.random() * 50;
  }
}

module.exports = { StealthEngine };
