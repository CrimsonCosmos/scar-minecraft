#!/usr/bin/env node
/**
 * SCAR-Minecraft Controller — main entry point.
 *
 * Starts a bedrock-protocol Relay (transparent proxy) and a TCP bridge
 * for the Python FPI agent.
 *
 * Usage:
 *   # Connect to a Realm:
 *   node controller/main.js --realm-invite "https://realms.gg/CODE" --bridge-port 3001
 *
 *   # Connect to a direct server:
 *   node controller/main.js --server-host play.example.com --server-port 19132
 *
 *   # With stealth mode:
 *   node controller/main.js --realm-invite "https://realms.gg/CODE" --stealth
 *
 *   # Custom relay listen port (if 19132 is taken):
 *   node controller/main.js --realm-invite "..." --listen-port 19133
 *
 * The real Minecraft Bedrock client connects to localhost:<listen-port>.
 * The Python FPI agent connects to localhost:<bridge-port>.
 *
 * Bot control starts DISABLED — user plays normally while FPI observes.
 * Python agent sends { cmd: "bot_control", enabled: true } to take over.
 */

const { RelayAdapter } = require('./relay');
const { createBridge } = require('./bridge');
const { StealthEngine } = require('./stealth');
const { parseArgs } = require('./utils');

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const config = {
    // Relay settings
    listenPort: parseInt(args['listen-port'] || '19132', 10),
    serverHost: args['server-host'] || args['host'] || 'localhost',
    serverPort: parseInt(args['server-port'] || args['port'] || '19132', 10),
    realmInvite: args['realm-invite'] || null,
    realmId: args['realm-id'] || null,
    authCache: args['auth-cache'] || './auth_cache',
    logPackets: args['log-packets'] === 'true',

    // Bridge settings
    bridgePort: parseInt(args['bridge-port'] || '3001', 10),
  };

  // Stealth engine
  const stealth = args.stealth === 'true' ? new StealthEngine() : null;
  if (stealth) {
    console.log('[main] Stealth mode enabled.');
  }

  // Action config (Bedrock only)
  const actionConfig = {
    tickRate: 20,
    actionDurationTicks: 4,
    stealth: stealth,
    protocol: 'bedrock',
  };

  // Tracking state (shared between bridge and adapter)
  const trackingState = {
    pendingRespawn: false,
    lastAttackLanded: false,
    lastPlayerHitLanded: false,
    killsSinceLastState: 0,
    attackedEntities: new Set(),
    attackCooldown: 0,
    knockbackCooldown: 0,
  };

  // Decrement knockback cooldown each tick
  setInterval(() => {
    if (trackingState.knockbackCooldown > 0) {
      trackingState.knockbackCooldown--;
    }
    if (trackingState.attackCooldown > 0) {
      trackingState.attackCooldown--;
    }
  }, 50);

  // Create relay adapter
  const adapter = new RelayAdapter(config);

  // Start bridge server (Python connects here)
  const bridgeServer = createBridge(config.bridgePort, adapter, trackingState, actionConfig);

  // Start relay (waits for real client to connect)
  console.log('[main] Starting relay proxy...');
  console.log('[main] Waiting for real Minecraft client to connect...');

  try {
    await adapter.start(trackingState);
    console.log('[main] Client connected. System ready.');
    console.log('[main] Bot control is OFF — user plays normally.');
    console.log('[main] Python agent can send { cmd: "bot_control", enabled: true } to take over.');
  } catch (err) {
    console.error('[main] Failed to start relay:', err.message);
    process.exit(1);
  }

  // Graceful shutdown
  process.on('SIGINT', () => {
    console.log('\n[main] Shutting down...');
    adapter.disconnect();
    bridgeServer.close();
    process.exit(0);
  });
}

main().catch(err => {
  console.error('[main] Fatal error:', err);
  process.exit(1);
});
