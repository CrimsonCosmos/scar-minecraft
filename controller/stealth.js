/**
 * StealthEngine — humanizes bot behavior to evade anti-cheat.
 *
 * Activated with --stealth flag or stealth: true in config.
 *
 * Core insight: real human timing is CORRELATED, not i.i.d. random.
 * A player who clicks at 120ms will click at ~118ms or ~125ms next,
 * not randomly jump to 200ms then back to 90ms. This module uses
 * exponentially-weighted drift to produce realistic timing patterns.
 *
 * Features:
 * - Correlated timing drift (EWMA random walk) for all intervals
 * - Fatigue model: timing gradually degrades, then "resets" (player refocuses)
 * - Micro-stutters: occasional 10-60ms lag spikes (simulates OS/network jitter)
 * - Burst-rest combat rhythm: attacks come in bursts, then brief repositioning
 * - Smooth multi-step camera rotation with overshoot correction
 * - Random idle insertions with variable duration
 * - Movement tick jitter: 38-62ms (simulates variable client framerate)
 */

const { sleep } = require('./utils');

/**
 * Correlated timing generator.
 * Produces humanized delays using a random walk with momentum,
 * so consecutive values are similar (autocorrelated) like real human input.
 */
class HumanTiming {
  constructor() {
    this._drift = 0;           // Current timing drift fraction (-0.25 to +0.25)
    this._momentum = 0;        // Drift velocity (smooths changes)
    this._fatigue = 0;         // Accumulated fatigue 0-1 (slows timing)
    this._ticksSinceReset = 0; // Ticks since last fatigue reset
    this._burstCount = 0;      // Actions in current burst
    this._burstLimit = 3 + Math.floor(Math.random() * 5); // 3-7 actions per burst
  }

  /**
   * Advance the internal state by one step.
   * Call this before generating each delay.
   */
  _tick() {
    // Random walk with momentum (produces correlated drift)
    this._momentum += (Math.random() - 0.5) * 0.25;
    this._momentum *= 0.82; // Decay toward zero
    this._drift += this._momentum;
    // Clamp drift to ±25% of base timing
    this._drift = Math.max(-0.25, Math.min(0.25, this._drift));

    // Fatigue accumulation
    this._ticksSinceReset++;
    if (this._ticksSinceReset > 60 + Math.random() * 120) {
      this._fatigue = Math.min(1.0, this._fatigue + 0.08 + Math.random() * 0.07);
      // Player "refocuses" when fatigue gets high
      if (this._fatigue > 0.6 && Math.random() < 0.25) {
        this._fatigue = Math.random() * 0.15; // Reset to near-zero
        this._ticksSinceReset = 0;
      }
    }
  }

  /**
   * Generate a humanized delay from a base value.
   * Returns ms with correlated drift, fatigue, and occasional micro-stutters.
   */
  nextDelay(baseMs) {
    this._tick();

    const fatigueScale = 1 + this._fatigue * 0.18; // Up to 18% slower when fatigued
    let delay = baseMs * (1 + this._drift) * fatigueScale;

    // Micro-stutter: 2.5% chance of an extra 10-60ms spike
    if (Math.random() < 0.025) {
      delay += 10 + Math.random() * 50;
    }

    // Floor at 70% of base to avoid impossibly fast timing
    return Math.max(baseMs * 0.7, Math.round(delay));
  }

  /**
   * Movement tick timing (base ~50ms / 20 TPS).
   * Varies 38-62ms to simulate variable client framerate.
   */
  nextMovementTick() {
    this._tick();
    const fatigueScale = 1 + this._fatigue * 0.05; // Slight fatigue effect on ticks
    let interval = 50 * (1 + this._drift * 0.5) * fatigueScale; // ±12.5% of 50ms

    // Occasional frame drop: 1.5% chance of a doubled tick
    if (Math.random() < 0.015) {
      interval += 20 + Math.random() * 30;
    }

    return Math.max(38, Math.min(65, Math.round(interval)));
  }

  /**
   * Attack delay with burst-rest pattern.
   * Humans attack in bursts of 3-7 hits, then briefly pause to reposition.
   */
  nextAttackDelay(baseMs) {
    this._burstCount++;

    // End of burst: insert a repositioning pause
    if (this._burstCount >= this._burstLimit) {
      this._burstCount = 0;
      this._burstLimit = 3 + Math.floor(Math.random() * 5);
      // Brief reposition pause: 150-400ms extra
      return this.nextDelay(baseMs) + 150 + Math.random() * 250;
    }

    // Within burst: slight anticipation effect (3% chance of early click)
    const anticipation = (Math.random() < 0.03) ? -baseMs * 0.06 : 0;
    return Math.max(baseMs * 0.7, this.nextDelay(baseMs) + anticipation);
  }

  /**
   * Reset burst counter (e.g., when target changes or combat pauses).
   */
  resetBurst() {
    this._burstCount = 0;
    this._burstLimit = 3 + Math.floor(Math.random() * 5);
  }
}


class StealthEngine {
  constructor(config = {}) {
    this.enabled = config.enabled !== false;
    this.actionJitter = config.actionJitterMs || [30, 150];   // ms range for pre-action delay
    this.lookSteps = config.lookSteps || 3;                   // smooth rotation step count
    this.attackVariance = config.attackCooldownVariance || 0.2;
    this.idleChance = config.idleChance || 0.05;              // 5% random pauses
    this.idleDurationMs = config.idleDurationMs || [80, 400]; // idle pause range

    // Correlated timing generator
    this.timing = new HumanTiming();
  }

  /**
   * Random delay before action execution (simulates human reaction time).
   * Uses correlated timing instead of pure uniform random.
   */
  async preActionDelay() {
    if (!this.enabled) return;
    const [min, max] = this.actionJitter;
    const center = (min + max) / 2;
    const delay = this.timing.nextDelay(center);
    // Clamp to configured range with some overflow allowed
    await sleep(Math.max(min * 0.8, Math.min(max * 1.3, delay)));
  }

  /**
   * Multi-step smooth rotation (humans don't snap-look).
   * Returns array of intermediate yaw values with realistic overshoot.
   */
  getLookSteps(currentYaw, targetYaw) {
    if (!this.enabled) return [targetYaw];

    // Vary step count: 2-4 steps based on angle magnitude
    let delta = targetYaw - currentYaw;
    // Normalize to [-180, 180]
    while (delta > 180) delta -= 360;
    while (delta < -180) delta += 360;

    const absDelta = Math.abs(delta);
    const steps = absDelta > 45 ? 4 : (absDelta > 15 ? 3 : 2);
    const result = [];

    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      // Ease-out curve: fast start, slow finish (like real mouse movement)
      const eased = 1 - Math.pow(1 - t, 2.2);
      // Per-step jitter: ±3% of remaining delta
      const jitter = (Math.random() - 0.5) * 0.06 * delta;
      // Slight overshoot on the last step (human micro-correction)
      const overshoot = (i === steps && absDelta > 20)
        ? (Math.random() - 0.5) * 0.04 * delta
        : 0;
      result.push(currentYaw + delta * eased + jitter + overshoot);
    }
    return result;
  }

  /**
   * Vary action duration (don't hold keys for exact same time).
   * Uses correlated timing for consistent feel.
   */
  getActionTicks(baseTicks) {
    if (!this.enabled) return baseTicks;
    // ±1 tick, biased by current drift
    const variance = Math.round(this.timing._drift * 3);
    return Math.max(2, baseTicks + Math.max(-1, Math.min(1, variance)));
  }

  /**
   * Should we insert a random idle? (breaks pattern detection).
   * Probability scales with fatigue (tired players pause more).
   */
  shouldRandomIdle() {
    if (!this.enabled) return false;
    const fatigueBoost = this.timing._fatigue * 0.04; // Up to +4% when fatigued
    return Math.random() < (this.idleChance + fatigueBoost);
  }

  /**
   * Get random idle duration (variable, not fixed).
   */
  getIdleDurationMs() {
    const [min, max] = this.idleDurationMs;
    return this.timing.nextDelay((min + max) / 2);
  }

  /**
   * Attack cooldown with human-like variance using correlated timing.
   */
  getAttackDelay(baseCooldownMs) {
    if (!this.enabled) return baseCooldownMs;
    return this.timing.nextAttackDelay(baseCooldownMs);
  }

  /**
   * Movement tick interval (for Bedrock relay).
   * Varies 38-65ms to simulate real client framerate.
   */
  getMovementTickMs() {
    if (!this.enabled) return 50;
    return this.timing.nextMovementTick();
  }
}

module.exports = { StealthEngine, HumanTiming };
