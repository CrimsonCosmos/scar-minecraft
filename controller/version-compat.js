/**
 * Version compatibility — runtime patches for unsupported Minecraft versions.
 *
 * Minecraft renamed versions from "1.x.x" to plain numbers (26.1, 26.1.1, etc.)
 * but minecraft-protocol / minecraft-data haven't added support yet.
 *
 * This module patches minecraft-data at runtime by:
 *   1. Using the real 26.1 protocol.json (correct packet IDs for config/play states)
 *   2. Falling back to 1.21.11 for non-protocol data (blocks, entities, items, etc.)
 *
 * This is necessary because simple version aliasing breaks the login sequence —
 * new packets in the configuration state shift ALL packet IDs.
 *
 * When official support is added, the patch becomes a no-op automatically.
 */

const path = require('path');
const fs = require('fs');

/**
 * Patch play-state packet IDs for protocol 775+ (Minecraft 26.1+).
 *
 * 26.1 added new packets in the middle of the play-state ID space,
 * shifting all subsequent IDs. PrismarineJS auto-generator copied the
 * 1.21.11 (protocol 774) mapping unchanged, so we fix it at runtime.
 *
 * Clientbound shifts (verified by raw packet capture against 26.1.2):
 *   0x00-0x26: unchanged
 *   0x27: NEW packet (inserted)
 *   old 0x27-0x31 → new 0x28-0x32 (+1)
 *   0x33: NEW packet (inserted)
 *   old 0x32-0x8a → new 0x34-0x8c (+2)
 *
 * Serverbound shifts (verified by probing multiple packets):
 *   0x00: unchanged (teleport_confirm)
 *   0x01: NEW packet "attack" (inserted)
 *   old 0x01-0x41 → new 0x02-0x42 (+1)
 */
function patchPlayPacketIds(protocol) {
  // Smart detection: independently check if CB and SB need shifting.
  // This handles the case where bundled protocol.json has SB already
  // shifted but CB still at 1.21.11 positions.

  // --- Clientbound play ---
  const cbRef = protocol.play.toClient.types.packet[1][0].type[1];

  // In 1.21.11 (unshifted), keep_alive is at 0x2b.
  // In 26.1 (shifted), keep_alive should be at 0x2d.
  const cbNeedsShift = cbRef.mappings['0x2b'] === 'keep_alive';

  if (cbNeedsShift) {
    const oldCbMap = { ...cbRef.mappings };
    const newCbMap = {};

    for (const [hex, name] of Object.entries(oldCbMap)) {
      const oldId = parseInt(hex, 16);
      let newId;
      if (oldId <= 0x26) {
        newId = oldId;
      } else if (oldId <= 0x31) {
        newId = oldId + 1;
      } else {
        newId = oldId + 2;
      }
      newCbMap['0x' + newId.toString(16).padStart(2, '0')] = name;
    }

    // New packets as opaque restBuffer containers
    newCbMap['0x27'] = 'unknown_cb_27';
    newCbMap['0x33'] = 'unknown_cb_33';
    cbRef.mappings = newCbMap;

    protocol.play.toClient.types['packet_unknown_cb_27'] = [
      'container', [{ name: 'data', type: 'restBuffer' }],
    ];
    protocol.play.toClient.types['packet_unknown_cb_33'] = [
      'container', [{ name: 'data', type: 'restBuffer' }],
    ];

    console.log(`[version-compat] Patched CB play packet IDs: ${Object.keys(newCbMap).length} (was ${Object.keys(oldCbMap).length})`);
  }

  // --- Serverbound play ---
  const sbRef = protocol.play.toServer.types.packet[1][0].type[1];

  // If "attack" is already in the SB mappings, SB is already shifted
  const sbHasAttack = Object.values(sbRef.mappings).includes('attack');

  if (!sbHasAttack) {
    const oldSbMap = { ...sbRef.mappings };
    const newSbMap = {};

    for (const [hex, name] of Object.entries(oldSbMap)) {
      const oldId = parseInt(hex, 16);
      let newId;
      if (oldId <= 0x00) {
        newId = oldId;           // teleport_confirm stays at 0x00
      } else {
        newId = oldId + 1;       // new "attack" packet inserted at 0x01
      }
      newSbMap['0x' + newId.toString(16).padStart(2, '0')] = name;
    }

    newSbMap['0x01'] = 'attack';
    sbRef.mappings = newSbMap;

    protocol.play.toServer.types['packet_attack'] = [
      'container', [{ name: 'data', type: 'restBuffer' }],
    ];

    console.log(`[version-compat] Patched SB play packet IDs: ${Object.keys(newSbMap).length} (was ${Object.keys(oldSbMap).length})`);
  }

  // 26.1 changed update_time format (VarLong instead of i64, extra fields).
  // Make it opaque — adapters parse raw data manually.
  protocol.play.toClient.types['packet_update_time'] = [
    'container', [{ name: 'data', type: 'restBuffer' }],
  ];

  // Same for declare_recipes and recipe_book_add — format changed in 26.1
  protocol.play.toClient.types['packet_declare_recipes'] = [
    'container', [{ name: 'data', type: 'restBuffer' }],
  ];
  if (protocol.play.toClient.types['packet_recipe_book_add']) {
    protocol.play.toClient.types['packet_recipe_book_add'] = [
      'container', [{ name: 'data', type: 'restBuffer' }],
    ];
  }

  protocol._packetIdsPatched = true;

  if (cbNeedsShift || !sbHasAttack) {
    console.log(`[version-compat] Packet ID patch: CB ${cbNeedsShift ? 'shifted' : 'already correct'}, SB ${sbHasAttack ? 'already correct' : 'shifted'}`);
  }
}

/**
 * Find the closest supported release version that has actual protocol data.
 */
function _findClosestSupported() {
  const mcData = require('minecraft-data');
  const allVersions = mcData.versions.pc;

  const supported = allVersions
    .filter(v => {
      if (v.releaseType !== 'release') return false;
      if (v.version >= 0x40000000) return false;
      try { return mcData(v.minecraftVersion) !== null; } catch (_) { return false; }
    })
    .sort((a, b) => b.version - a.version);

  return supported[0] || null;
}

/**
 * Load the bundled 26.1 protocol.json from our data/ directory.
 * Returns null if not available.
 */
function _loadBundledProtocol(majorVersion) {
  const protoPath = path.join(__dirname, 'data', majorVersion, 'protocol.json');
  try {
    if (fs.existsSync(protoPath)) {
      return require(protoPath);
    }
  } catch (_) {}
  return null;
}

/**
 * Ensure minecraft-data has protocol data for the given version.
 * If the version is already supported, this is a no-op.
 *
 * Handles two cases:
 *   1. Version is in protocolVersions.json but has no data (e.g., "26.1")
 *   2. Version is completely unknown (e.g., "26.1.2")
 *
 * @param {string} targetVersion - e.g., '26.1' or '26.1.2'
 * @param {number} [protocolNum] - optional protocol number from server ping
 * @returns {string} The version string to pass to mc.createClient()
 */
function patchVersionSupport(targetVersion, protocolNum) {
  if (!targetVersion) return targetVersion;

  const mcData = require('minecraft-data');

  // Already supported — nothing to do
  try { if (mcData(targetVersion)) return targetVersion; } catch (_) {}

  const allVersions = mcData.versions.pc;
  let target = allVersions.find(v => v.minecraftVersion === targetVersion);

  // Case 2: version not in protocolVersions.json (e.g., "26.1.2")
  if (!target) {
    const parts = targetVersion.split('.');
    let matched = null;
    for (let i = parts.length - 1; i >= 1; i--) {
      const prefix = parts.slice(0, i).join('.');
      matched = allVersions.find(v =>
        v.minecraftVersion === prefix &&
        v.releaseType === 'release' &&
        v.version < 0x40000000
      );
      if (matched) break;
    }

    if (!matched) {
      console.warn(`[version-compat] Unknown version "${targetVersion}" — no matching prefix in protocolVersions.json`);
      return targetVersion;
    }

    const synthProtocol = protocolNum || matched.version;
    target = {
      minecraftVersion: targetVersion,
      version: synthProtocol,
      dataVersion: (matched.dataVersion || 0) + 1,
      majorVersion: matched.majorVersion,
      releaseType: 'release',
    };

    console.log(`[version-compat] "${targetVersion}" not in registry. Using prefix "${matched.minecraftVersion}" (protocol ${synthProtocol}).`);

    // Register in lookup tables
    const vbmv = mcData.versionsByMinecraftVersion;
    if (vbmv && vbmv.pc) {
      vbmv.pc[targetVersion] = {
        minecraftVersion: target.minecraftVersion,
        version: target.version,
        dataVersion: target.dataVersion,
        usesNetty: true,
        majorVersion: target.majorVersion,
        releaseType: 'release',
      };
    }

    const pnvbpv = mcData.postNettyVersionsByProtocolVersion;
    if (pnvbpv && pnvbpv.pc) {
      const key = String(target.version);
      if (!pnvbpv.pc[key]) pnvbpv.pc[key] = [];
      pnvbpv.pc[key].push({
        minecraftVersion: target.minecraftVersion,
        version: target.version,
        dataVersion: target.dataVersion,
        majorVersion: target.majorVersion,
        releaseType: 'release',
      });
    }
  }

  // Find closest supported version for non-protocol data (blocks, entities, etc.)
  const closest = _findClosestSupported();
  if (!closest) {
    console.warn('[version-compat] No supported versions found — cannot patch');
    return targetVersion;
  }

  const protocolDiff = Math.abs(target.version - closest.version);
  console.log(`[version-compat] ${targetVersion} (protocol ${target.version}) not natively supported.`);
  console.log(`[version-compat] Using ${closest.minecraftVersion} as base (protocol ${closest.version}, diff=${protocolDiff}).`);

  // Patch minecraft-data's internal data map
  const dataModule = require('minecraft-data/data');
  // Use the specific sub-version data (e.g., '1.21.11') not just the base
  // major version (e.g., '1.21'). Entity IDs shift between minor versions
  // (1.21.3 added ~20 entities), so base '1.21' has wrong entity/block IDs.
  const closestData = dataModule.pc[closest.minecraftVersion] || dataModule.pc[closest.majorVersion];
  if (!closestData) {
    console.warn(`[version-compat] Could not load data for ${closest.majorVersion}`);
    return targetVersion;
  }

  // Only patch if this majorVersion doesn't already have data
  if (dataModule.pc[target.majorVersion]) {
    // Already patched (e.g., 26.1 already done, now doing 26.1.2 with same majorVersion)
    try {
      const test = mcData(targetVersion);
      if (test) {
        console.log(`[version-compat] ${targetVersion} already resolved via ${target.majorVersion} data.`);
        return targetVersion;
      }
    } catch (_) {}
  }

  // Create patched entry: real protocol.json + fallback data from closest version
  const bundledProtocol = _loadBundledProtocol(target.majorVersion);

  // Apply patchPlayPacketIds to whichever protocol is used.
  // The function uses smart detection (checks keep_alive position for CB,
  // checks for "attack" presence for SB) so it's safe on both bundled
  // and fallback protocols — it only shifts what actually needs shifting.

  const patchedEntry = {};

  for (const key of Object.getOwnPropertyNames(closestData)) {
    if (key === 'version') {
      // Override version to report correct protocol number
      Object.defineProperty(patchedEntry, 'version', {
        get() {
          return {
            minecraftVersion: target.minecraftVersion,
            version: target.version,
            dataVersion: target.dataVersion || 0,
            majorVersion: target.majorVersion,
            type: 'pc',
          };
        },
        enumerable: true,
        configurable: true,
      });
    } else if (key === 'protocol' && bundledProtocol) {
      // Use real protocol.json with correct packet IDs
      Object.defineProperty(patchedEntry, 'protocol', {
        get() { return bundledProtocol; },
        enumerable: true,
        configurable: true,
      });
    } else {
      // Copy everything else from closest version (blocks, entities, items, etc.)
      const descriptor = Object.getOwnPropertyDescriptor(closestData, key);
      Object.defineProperty(patchedEntry, key, descriptor);
    }
  }

  // Patch play-state packet IDs for 26.1+ — the function auto-detects
  // what needs shifting (CB, SB, or both) so it's safe for any state.
  if (target.version >= 775) {
    try {
      const proto = patchedEntry.protocol;
      if (proto) {
        patchPlayPacketIds(proto);
      }
    } catch (_) {}
  }

  dataModule.pc[target.majorVersion] = patchedEntry;

  // Verify
  try {
    const test = mcData(targetVersion);
    if (test) {
      const hasRealProto = !!bundledProtocol;
      console.log(`[version-compat] Patch successful. mcData("${targetVersion}") OK (protocol ${test.version.version}, real protocol.json: ${hasRealProto}).`);
    } else {
      console.warn(`[version-compat] Patch failed — mcData("${targetVersion}") returns null.`);
    }
  } catch (e) {
    console.warn(`[version-compat] Patch verification error:`, e.message);
  }

  return targetVersion;
}

/**
 * Preemptively patch ALL unsupported release versions in minecraft-data.
 * Call ONCE at startup, before any minecraft-protocol operations.
 *
 * @returns {string[]} List of version strings that were patched.
 */
function patchAllUnsupported() {
  const mcData = require('minecraft-data');
  const allVersions = mcData.versions.pc;

  const unsupported = allVersions.filter(v => {
    if (v.releaseType !== 'release') return false;
    if (v.version >= 0x40000000) return false;
    try { return mcData(v.minecraftVersion) === null; } catch (_) { return true; }
  });

  if (unsupported.length === 0) return [];

  const patched = [];
  for (const ver of unsupported) {
    const result = patchVersionSupport(ver.minecraftVersion);
    if (result) patched.push(result);
  }
  return patched;
}

module.exports = { patchVersionSupport, patchAllUnsupported, patchPlayPacketIds };
