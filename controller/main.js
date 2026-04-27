#!/usr/bin/env node
/**
 * SCAR-Minecraft Controller — CLI entry point.
 *
 * Starts a protocol relay (transparent proxy) and a TCP bridge
 * for the Python FPI agent. Supports both Bedrock and Java Edition.
 *
 * Usage:
 *   # Java — Use your own singleplayer world (easiest):
 *   node controller/main.js --world "MyWorld"
 *   node controller/main.js --world "~/Library/Application Support/minecraft/saves/MyWorld"
 *
 *   # Bedrock — Connect to a Realm:
 *   node controller/main.js --realm-invite "https://realms.gg/CODE"
 *
 *   # Bedrock — Direct server:
 *   node controller/main.js --server-host play.example.com --server-port 19132
 *
 *   # Java Edition — Direct server:
 *   node controller/main.js --protocol java --server-host play.example.com --server-port 25565
 *
 *   # Java — With Microsoft auth (for online-mode servers):
 *   node controller/main.js --protocol java --server-host mc.server.com --online-mode
 *
 *   # Java — 1.8 PvP mode (no attack cooldown):
 *   node controller/main.js --protocol java --server-host mc.server.com --pvp-style spam
 *
 *   # With stealth mode:
 *   node controller/main.js --realm-invite "https://realms.gg/CODE" --stealth
 *
 *   # With keyboard fallback:
 *   node controller/main.js --realm-invite "..." --keyboard-fallback
 *
 *   # Custom ports:
 *   node controller/main.js --listen-port 19133 --bridge-port 3002
 *
 * The real Minecraft client connects to localhost:<listen-port>.
 * The Python FPI agent connects to localhost:<bridge-port>.
 *
 * Bot control starts DISABLED — user plays normally while FPI observes.
 * Python agent sends { cmd: "bot_control", enabled: true } to take over.
 */

const { ScarLauncher } = require('./launcher');
const { parseArgs } = require('./utils');

async function main() {
  const args = parseArgs(process.argv.slice(2));

  // --world implies Java protocol
  const worldPath = args['world'] || null;
  const protocol = worldPath ? 'java' : (args.protocol || 'bedrock');

  // Handle --world list
  if (worldPath === 'list') {
    const { listWorlds } = require('./auto-server');
    const worlds = listWorlds();
    if (worlds.length === 0) {
      console.log('No world saves found in default Minecraft directory.');
    } else {
      console.log('Available worlds:');
      for (const w of worlds) {
        console.log(`  ${w.name}  →  ${w.path}`);
      }
    }
    process.exit(0);
  }

  const launcher = new ScarLauncher({
    protocol,
    worldPath: worldPath || undefined,
    realmInvite: args['realm-invite'] || undefined,
    serverHost: args['server-host'] || args['host'] || undefined,
    serverPort: parseInt(args['server-port'] || args['port'] || '0', 10) || undefined,
    listenPort: parseInt(args['listen-port'] || '0', 10) || undefined,
    bridgePort: parseInt(args['bridge-port'] || '3001', 10),
    stealth: args.stealth === 'true',
    keyboardFallback: args['keyboard-fallback'] === 'true',
    mouseSensitivity: parseInt(args['mouse-sensitivity'] || '400', 10),
    pvpStyle: args['pvp-style'] || 'cooldown',
    onlineMode: args['online-mode'] === 'true',
    version: args.version || undefined,
    phase: parseInt(args.phase || '3', 10),
  });

  launcher.on('log', ({ source, message }) => {
    console.log(`[${source}] ${message}`);
  });

  await launcher.start();

  process.on('SIGINT', async () => {
    console.log('\n[main] Shutting down...');
    await launcher.shutdown();
    process.exit(0);
  });
}

main().catch(err => {
  console.error('[main] Fatal error:', err);
  process.exit(1);
});
