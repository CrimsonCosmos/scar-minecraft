#!/usr/bin/env node
/**
 * Diagnostic: connect to LAN server, try switching to spectator mode via
 * change_gamemode packet and chat_command, log responses.
 *
 * Usage: node tests/diag-spectator.js [port]
 *
 * To test with spectator as LAN default gamemode:
 *   1. In MC: Settings > Open to LAN > Game Mode: Spectator > Allow Commands: ON
 *   2. Run this script
 */

const mc = require('minecraft-protocol');
const { patchVersionSupport } = require('../controller/version-compat');

const PORT = parseInt(process.argv[2]) || 52083;

async function main() {
  console.log(`Connecting to localhost:${PORT}...`);

  const pingResult = await mc.ping({ host: 'localhost', port: PORT });
  const serverVer = pingResult.version?.name || 'unknown';
  const serverProto = pingResult.version?.protocol || 0;
  console.log(`Server: ${serverVer} (protocol ${serverProto})`);

  const patchedVer = patchVersionSupport(serverVer, serverProto);

  const client = mc.createClient({
    host: 'localhost',
    port: PORT,
    username: 'SpectatorTest',
    version: patchedVer,
    auth: 'offline',
    keepAlive: true,
    hideErrors: false,
  });

  let spectatorConfirmed = false;
  let spectatorSent = false;

  client.on('login', (data) => {
    console.log(`\nLogged in — entity ID: ${data.entityId}`);
    const gm = data.worldState?.gamemode;
    console.log(`  Initial gamemode: ${gm} (${typeof gm})`);
    if (gm === 'spectator' || gm === 3) {
      console.log('  >>> SPECTATOR from login! No command needed.');
      spectatorConfirmed = true;
    }
  });

  client.on('packet', (data, meta) => {
    if (meta.name === 'game_state_change') {
      console.log(`[game_state_change] reason=${data.reason} gameMode=${data.gameMode}`);
      if ((data.reason === 3 || data.reason === 'change_game_mode') &&
          (data.gameMode === 3 || data.gameMode === 3.0)) {
        console.log('  >>> SPECTATOR confirmed via game_state_change!');
        spectatorConfirmed = true;
      }
    }
    if (meta.name === 'system_chat') {
      const content = data.content;
      let msg = typeof content === 'string' ? content : JSON.stringify(content);
      console.log(`[system_chat] ${msg}`);
    }
    if (meta.name === 'abilities') {
      console.log(`[abilities] flags=${data.flags} fly=${data.flyingSpeed} walk=${data.walkingSpeed}`);
    }
    if (meta.name === 'kick_disconnect' || meta.name === 'disconnect') {
      console.log(`[KICK] ${JSON.stringify(data)}`);
    }
  });

  client.on('position', (data) => {
    console.log(`[position] x=${data.x?.toFixed(1)} y=${data.y?.toFixed(1)} z=${data.z?.toFixed(1)}`);
    try {
      client.write('teleport_confirm', { teleportId: data.teleportId || 0 });
    } catch (_) {}

    if (!spectatorSent && !spectatorConfirmed) {
      spectatorSent = true;
      console.log('\n--- Trying spectator in 2s ---');
      setTimeout(() => {
        console.log('\n[1] change_gamemode { mode: 3 }');
        try {
          client.write('change_gamemode', { mode: 3 });
          console.log('  sent OK');
        } catch (e) { console.log('  FAILED:', e.message); }

        setTimeout(() => {
          if (!spectatorConfirmed) {
            console.log('\n[2] chat_command "gamemode spectator"');
            try {
              client.write('chat_command', { command: 'gamemode spectator' });
              console.log('  sent OK');
            } catch (e) { console.log('  FAILED:', e.message); }
          }

          setTimeout(() => {
            console.log(`\n--- Result: spectator=${spectatorConfirmed} ---`);
            if (!spectatorConfirmed) {
              console.log('FAILED. To fix: re-open LAN with Game Mode: Spectator');
            }
            client.end();
            process.exit(spectatorConfirmed ? 0 : 1);
          }, 5000);
        }, 2000);
      }, 2000);
    }
  });

  client.on('chunk_batch_finished', () => {
    try { client.write('chunk_batch_received', { chunksPerTick: 20.0 }); } catch (_) {}
  });

  client.on('error', (err) => console.error('Error:', err.message));
  client.on('end', (reason) => { console.log('Disconnected:', reason); process.exit(1); });
  setTimeout(() => { console.log('Timeout.'); client.end(); process.exit(1); }, 25000);
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
