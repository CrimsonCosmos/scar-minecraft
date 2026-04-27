/**
 * State dict contract tests — validates golden_state.json against
 * the schema expected by both state.js (JS) and env.py/encoder.py (Python).
 *
 * Run: node --test tests/test_state_contract.js
 */

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const goldenState = require('./golden_state.json');

// ── env.py REQUIRED_STATE_KEYS (must match exactly) ────────────────
const REQUIRED_STATE_KEYS = new Set([
  'health', 'food', 'alive',
  'time_of_day', 'light_level',
  'yaw', 'pitch', 'on_ground', 'is_in_water', 'is_raining', 'altitude',
  'spatial', 'entities', 'inventory',
  'xp_level', 'xp_points',
  'attack_cooldown', 'hit_landed', 'player_hit_landed', 'kills',
]);

// ���─ env.py OPTIONAL_STATE_KEYS ─��───────────────────────────���───────
const OPTIONAL_STATE_KEYS = new Set([
  'food_saturation', 'position',
  'bot_control_active', 'user_active', 'is_using_item',
  'type',
  'macro_status',
  'crowd',
  'self_velocity', 'health_delta', 'food_delta', 'ticks_airborne',
  'self_effects', 'incoming_projectile', 'self_armor_tier', 'is_thundering',
  'nearest_hostile_accel', 'nearest_player_armor', 'height_vs_hostile',
  'height_vs_player', 'combat_hits_5s', 'combat_damage_5s',
  'time_since_hit', 'kill_streak', 'strafing',
]);

// ── Spatial grid groups (from encoder.py SPATIAL_GROUPS) ───────────
const SPATIAL_GROUPS = ['body_clear', 'drop_depth', 'overhead', 'danger', 'composition', 'immediate'];

// ── Entity field expectations ──────────────────────────────────────
const ENTITY_REQUIRED_FIELDS = ['name', 'distance', 'health', 'max_health', 'flags', 'hand_state'];
const ENTITY_OPTIONAL_FIELDS = ['is_baby', 'creeper_state', 'creeper_charged', 'bearing', 'speed', 'approach', 'facing_us', 'equipment_tier', 'velocity'];


describe('golden_state.json — required keys', () => {
  for (const key of REQUIRED_STATE_KEYS) {
    it(`has required key "${key}"`, () => {
      assert.ok(key in goldenState, `Missing required key: ${key}`);
    });
  }
});


describe('golden_state.json — optional keys present', () => {
  // Golden state should include ALL optional keys except 'type' and 'macro_status'
  // (those are only injected in specific contexts)
  const expectedOptional = [...OPTIONAL_STATE_KEYS].filter(k => k !== 'type' && k !== 'macro_status');

  for (const key of expectedOptional) {
    it(`has optional key "${key}"`, () => {
      assert.ok(key in goldenState, `Missing optional key: ${key}`);
    });
  }
});


describe('golden_state.json — no unknown keys', () => {
  it('all keys are known', () => {
    const allKnown = new Set([...REQUIRED_STATE_KEYS, ...OPTIONAL_STATE_KEYS, '_comment']);
    const unknownKeys = Object.keys(goldenState).filter(k => !allKnown.has(k));
    assert.deepStrictEqual(unknownKeys, [], `Unknown keys: ${unknownKeys.join(', ')}`);
  });
});


describe('golden_state.json — field types', () => {
  it('health is a number', () => assert.strictEqual(typeof goldenState.health, 'number'));
  it('food is a number', () => assert.strictEqual(typeof goldenState.food, 'number'));
  it('alive is a boolean', () => assert.strictEqual(typeof goldenState.alive, 'boolean'));
  it('yaw is a number', () => assert.strictEqual(typeof goldenState.yaw, 'number'));
  it('pitch is a number', () => assert.strictEqual(typeof goldenState.pitch, 'number'));
  it('on_ground is a boolean', () => assert.strictEqual(typeof goldenState.on_ground, 'boolean'));
  it('altitude is a number', () => assert.strictEqual(typeof goldenState.altitude, 'number'));
  it('time_of_day is a number', () => assert.strictEqual(typeof goldenState.time_of_day, 'number'));
  it('kills is a number', () => assert.strictEqual(typeof goldenState.kills, 'number'));
  it('attack_cooldown is a number', () => assert.strictEqual(typeof goldenState.attack_cooldown, 'number'));
  it('hit_landed is a boolean', () => assert.strictEqual(typeof goldenState.hit_landed, 'boolean'));
  it('player_hit_landed is a boolean', () => assert.strictEqual(typeof goldenState.player_hit_landed, 'boolean'));
  it('bot_control_active is a boolean', () => assert.strictEqual(typeof goldenState.bot_control_active, 'boolean'));
  it('spatial is an object', () => assert.strictEqual(typeof goldenState.spatial, 'object'));
  it('entities is an object', () => assert.strictEqual(typeof goldenState.entities, 'object'));
  it('inventory is an object', () => assert.strictEqual(typeof goldenState.inventory, 'object'));
  it('position is an object', () => assert.strictEqual(typeof goldenState.position, 'object'));
});


describe('golden_state.json — position structure', () => {
  it('position has x, y, z', () => {
    const pos = goldenState.position;
    assert.strictEqual(typeof pos.x, 'number');
    assert.strictEqual(typeof pos.y, 'number');
    assert.strictEqual(typeof pos.z, 'number');
  });

  it('altitude matches position.y', () => {
    assert.strictEqual(goldenState.altitude, goldenState.position.y);
  });
});


describe('golden_state.json — spatial structure', () => {
  for (const group of SPATIAL_GROUPS) {
    it(`spatial.${group} is a 4-element array`, () => {
      const arr = goldenState.spatial[group];
      assert.ok(Array.isArray(arr), `spatial.${group} should be an array`);
      assert.strictEqual(arr.length, 4, `spatial.${group} should have 4 elements`);
      for (const v of arr) {
        assert.strictEqual(typeof v, 'number', `spatial.${group} elements should be numbers`);
      }
    });
  }

  it('spatial has exactly 6 groups', () => {
    const groups = Object.keys(goldenState.spatial);
    assert.strictEqual(groups.length, 6);
  });
});


describe('golden_state.json — entity structure', () => {
  it('entities has hostiles, passives, players arrays', () => {
    assert.ok(Array.isArray(goldenState.entities.hostiles));
    assert.ok(Array.isArray(goldenState.entities.passives));
    assert.ok(Array.isArray(goldenState.entities.players));
  });

  const allEntities = [
    ...goldenState.entities.hostiles,
    ...goldenState.entities.passives,
    ...goldenState.entities.players,
  ];

  for (const field of ENTITY_REQUIRED_FIELDS) {
    it(`all entities have "${field}"`, () => {
      for (const ent of allEntities) {
        assert.ok(field in ent, `Entity "${ent.name}" missing field "${field}"`);
      }
    });
  }

  it('entity distances are positive', () => {
    for (const ent of allEntities) {
      assert.ok(ent.distance > 0, `Entity "${ent.name}" has non-positive distance: ${ent.distance}`);
    }
  });

  it('entities have speed and facing_us fields', () => {
    for (const ent of allEntities) {
      assert.strictEqual(typeof ent.speed, 'number', `Entity "${ent.name}" missing speed`);
      assert.strictEqual(typeof ent.facing_us, 'number', `Entity "${ent.name}" missing facing_us`);
    }
  });
});


describe('golden_state.json — crowd structure', () => {
  it('crowd has count fields', () => {
    const crowd = goldenState.crowd;
    assert.strictEqual(typeof crowd.hostile_count, 'number');
    assert.strictEqual(typeof crowd.hostile_avg_dist, 'number');
    assert.strictEqual(typeof crowd.hostile_near, 'number');
    assert.strictEqual(typeof crowd.passive_count, 'number');
    assert.strictEqual(typeof crowd.player_count, 'number');
  });

  it('crowd has directional fields', () => {
    const crowd = goldenState.crowd;
    assert.ok(Array.isArray(crowd.quadrant_density));
    assert.strictEqual(crowd.quadrant_density.length, 4);
    assert.strictEqual(typeof crowd.threat_direction.sin, 'number');
    assert.strictEqual(typeof crowd.threat_direction.cos, 'number');
    assert.strictEqual(typeof crowd.threat_direction.magnitude, 'number');
  });

  it('crowd has attacker fields', () => {
    const crowd = goldenState.crowd;
    assert.strictEqual(typeof crowd.attacker_dist, 'number');
    assert.strictEqual(typeof crowd.attacker_bearing.sin, 'number');
    assert.strictEqual(typeof crowd.attacker_bearing.cos, 'number');
    assert.strictEqual(typeof crowd.under_attack, 'number');
  });
});


describe('golden_state.json — inventory structure', () => {
  it('inventory has slots_used, selected_slot, hotbar', () => {
    const inv = goldenState.inventory;
    assert.strictEqual(typeof inv.slots_used, 'number');
    assert.strictEqual(typeof inv.selected_slot, 'number');
    assert.ok(Array.isArray(inv.hotbar));
  });

  it('hotbar has exactly 9 slots', () => {
    assert.strictEqual(goldenState.inventory.hotbar.length, 9);
  });

  it('non-null hotbar items have category, tier, durability, count, max_stack', () => {
    for (const slot of goldenState.inventory.hotbar) {
      if (slot === null) continue;
      assert.strictEqual(typeof slot.category, 'number', 'category should be number');
      assert.strictEqual(typeof slot.tier, 'number', 'tier should be number');
      assert.strictEqual(typeof slot.durability, 'number', 'durability should be number');
      assert.strictEqual(typeof slot.count, 'number', 'count should be number');
      assert.strictEqual(typeof slot.max_stack, 'number', 'max_stack should be number');
    }
  });
});


describe('golden_state.json — per-entity facing and bearing', () => {
  it('hostiles have bearing, speed, approach, facing_us', () => {
    for (const h of goldenState.entities.hostiles) {
      assert.strictEqual(typeof h.bearing.sin, 'number', `${h.name} missing bearing.sin`);
      assert.strictEqual(typeof h.bearing.cos, 'number', `${h.name} missing bearing.cos`);
      assert.strictEqual(typeof h.speed, 'number', `${h.name} missing speed`);
      assert.strictEqual(typeof h.approach, 'number', `${h.name} missing approach`);
      assert.strictEqual(typeof h.facing_us, 'number', `${h.name} missing facing_us`);
    }
  });

  it('players have bearing, speed, approach, facing_us, equipment_tier', () => {
    for (const p of goldenState.entities.players) {
      assert.strictEqual(typeof p.bearing.sin, 'number', `${p.name} missing bearing.sin`);
      assert.strictEqual(typeof p.speed, 'number', `${p.name} missing speed`);
      assert.strictEqual(typeof p.approach, 'number', `${p.name} missing approach`);
      assert.strictEqual(typeof p.facing_us, 'number', `${p.name} missing facing_us`);
      assert.strictEqual(typeof p.equipment_tier, 'number', `${p.name} missing equipment_tier`);
    }
  });

  it('passives have speed and facing_us', () => {
    for (const p of goldenState.entities.passives) {
      assert.strictEqual(typeof p.speed, 'number', `${p.name} missing speed`);
      assert.strictEqual(typeof p.facing_us, 'number', `${p.name} missing facing_us`);
    }
  });
});


describe('golden_state.json — bridge roundtrip', () => {
  it('JSON.parse(JSON.stringify(state)) preserves all fields', () => {
    const roundtripped = JSON.parse(JSON.stringify(goldenState));
    // Remove _comment for comparison
    delete roundtripped._comment;
    const original = { ...goldenState };
    delete original._comment;
    assert.deepStrictEqual(roundtripped, original);
  });

  it('no undefined values (would be lost in JSON)', () => {
    const json = JSON.stringify(goldenState);
    const parsed = JSON.parse(json);
    // Count non-_comment keys
    const origKeys = Object.keys(goldenState).filter(k => k !== '_comment');
    const parsedKeys = Object.keys(parsed).filter(k => k !== '_comment');
    assert.strictEqual(origKeys.length, parsedKeys.length,
      'Key count changed after JSON roundtrip — possible undefined value');
  });
});


describe('golden_state.json — value ranges', () => {
  it('health in [0, 20]', () => {
    assert.ok(goldenState.health >= 0 && goldenState.health <= 20);
  });

  it('food in [0, 20]', () => {
    assert.ok(goldenState.food >= 0 && goldenState.food <= 20);
  });

  it('light_level in [0, 15]', () => {
    assert.ok(goldenState.light_level >= 0 && goldenState.light_level <= 15);
  });

  it('time_of_day in [0, 24000]', () => {
    assert.ok(goldenState.time_of_day >= 0 && goldenState.time_of_day <= 24000);
  });

  it('altitude is reasonable', () => {
    assert.ok(goldenState.altitude >= -64 && goldenState.altitude <= 320);
  });
});


describe('golden_state.json — self-awareness fields', () => {
  it('self_velocity has x, y, z', () => {
    const v = goldenState.self_velocity;
    assert.strictEqual(typeof v.x, 'number');
    assert.strictEqual(typeof v.y, 'number');
    assert.strictEqual(typeof v.z, 'number');
  });

  it('health_delta is a number', () => assert.strictEqual(typeof goldenState.health_delta, 'number'));
  it('food_delta is a number', () => assert.strictEqual(typeof goldenState.food_delta, 'number'));
  it('ticks_airborne is a number', () => assert.strictEqual(typeof goldenState.ticks_airborne, 'number'));
  it('self_armor_tier is a number', () => assert.strictEqual(typeof goldenState.self_armor_tier, 'number'));
  it('is_thundering is a boolean', () => assert.strictEqual(typeof goldenState.is_thundering, 'boolean'));
  it('strafing is a number', () => assert.strictEqual(typeof goldenState.strafing, 'number'));

  it('self_effects has speed, strength, resistance, regeneration', () => {
    const e = goldenState.self_effects;
    assert.strictEqual(typeof e.speed, 'number');
    assert.strictEqual(typeof e.strength, 'number');
    assert.strictEqual(typeof e.resistance, 'number');
    assert.strictEqual(typeof e.regeneration, 'number');
  });
});


describe('golden_state.json — threat dynamics fields', () => {
  it('incoming_projectile has expected structure when present', () => {
    const p = goldenState.incoming_projectile;
    if (p !== null) {
      assert.strictEqual(typeof p.name, 'string');
      assert.strictEqual(typeof p.distance, 'number');
      assert.strictEqual(typeof p.speed, 'number');
      assert.strictEqual(typeof p.bearing.sin, 'number');
      assert.strictEqual(typeof p.bearing.cos, 'number');
    }
  });

  it('nearest_hostile_accel is a number', () => assert.strictEqual(typeof goldenState.nearest_hostile_accel, 'number'));
  it('nearest_player_armor is a number', () => assert.strictEqual(typeof goldenState.nearest_player_armor, 'number'));
  it('height_vs_hostile is a number', () => assert.strictEqual(typeof goldenState.height_vs_hostile, 'number'));
  it('height_vs_player is a number', () => assert.strictEqual(typeof goldenState.height_vs_player, 'number'));
  it('combat_hits_5s is a number', () => assert.strictEqual(typeof goldenState.combat_hits_5s, 'number'));
  it('combat_damage_5s is a number', () => assert.strictEqual(typeof goldenState.combat_damage_5s, 'number'));
  it('time_since_hit is a number', () => assert.strictEqual(typeof goldenState.time_since_hit, 'number'));
  it('kill_streak is a number', () => assert.strictEqual(typeof goldenState.kill_streak, 'number'));
});
