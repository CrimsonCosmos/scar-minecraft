/**
 * TCP bridge server — newline-delimited JSON protocol.
 *
 * Same protocol as fpi-minecraft: Python FPI agent connects and sends
 * { cmd: "get_state" | "action" | "respawn" | "disconnect" | "bot_control" }
 *
 * New commands for relay mode:
 *   { cmd: "bot_control", enabled: true/false } — toggle FPI agent control
 *   State responses include bot_control_active: boolean
 */

const net = require('net');
const { getState } = require('./state');
const { executeAction, executeCompositeAction, tryAutoEat } = require('./actions');
const { sleep } = require('./utils');

function send(socket, obj) {
  try {
    socket.write(JSON.stringify(obj) + '\n');
  } catch (e) {
    console.error('[bridge] Failed to send:', e.message);
  }
}

function createBridge(port, adapter, trackingState, actionConfig) {
  const server = net.createServer((socket) => {
    console.log('[bridge] Python client connected.');
    let buffer = '';

    socket.on('data', (data) => {
      buffer += data.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        handleMessage(socket, line.trim(), adapter, trackingState, actionConfig);
      }
    });

    socket.on('close', () => {
      console.log('[bridge] Python client disconnected.');
    });

    socket.on('error', (err) => {
      console.error('[bridge] Socket error:', err.message);
    });
  });

  server.listen(port, () => {
    console.log(`[bridge] TCP bridge listening on port ${port}`);
  });

  return server;
}

async function handleMessage(socket, rawMsg, adapter, trackingState, actionConfig) {
  let msg;
  try {
    msg = JSON.parse(rawMsg);
  } catch (e) {
    send(socket, { type: 'error', message: 'Invalid JSON' });
    return;
  }

  if (!adapter.ready) {
    send(socket, { type: 'error', message: 'Client not connected to relay' });
    return;
  }

  switch (msg.cmd) {
    case 'get_state': {
      const st = getState(adapter, trackingState);
      // Log entity counts every ~100 requests
      if (!trackingState._stateReqCount) trackingState._stateReqCount = 0;
      trackingState._stateReqCount++;
      if (trackingState._stateReqCount % 100 === 1) {
        const h = st.entities?.hostiles?.length || 0;
        const p = st.entities?.passives?.length || 0;
        const pl = st.entities?.players?.length || 0;
        const c = st.crowd || {};
        console.log(`[bridge] Entities: ${h} hostile, ${p} passive, ${pl} player | crowd: ${c.hostile_count || 0}h ${c.passive_count || 0}p | total_tracked: ${adapter._entities?.size || '?'}`);
      }
      send(socket, { type: 'state', ...st });
      break;
    }

    case 'action':
      try {
        // If bot control is not active, just return state without executing
        if (!adapter.botControlActive) {
          send(socket, { type: 'state', ...getState(adapter, trackingState) });
          break;
        }

        if (trackingState.pendingRespawn && adapter.health > 0) {
          trackingState.pendingRespawn = false;
        }
        if (trackingState.pendingRespawn || adapter.health <= 0) {
          send(socket, { type: 'state', ...getState(adapter, trackingState) });
          break;
        }

        if (msg.movement !== undefined) {
          await executeCompositeAction(
            adapter, msg.movement, msg.look, msg.combat,
            trackingState, actionConfig,
          );
        } else {
          // Pass macroArgs for GO_TO_COORDINATES (id 23)
          if (msg.id >= 20 && msg.macroArgs) {
            actionConfig._macroArgs = msg.macroArgs;
          }
          await executeAction(adapter, msg.id, trackingState, actionConfig);
          delete actionConfig._macroArgs;
        }

        await tryAutoEat(adapter, trackingState);

        // Include macro_status if a macro-action was executed
        const state = getState(adapter, trackingState);
        if (trackingState.lastMacroStatus) {
          state.macro_status = trackingState.lastMacroStatus;
          trackingState.lastMacroStatus = null;
        }
        send(socket, { type: 'state', ...state });
      } catch (err) {
        try {
          const state = getState(adapter, trackingState);
          if (trackingState.lastMacroStatus) {
            state.macro_status = trackingState.lastMacroStatus;
            trackingState.lastMacroStatus = null;
          }
          send(socket, { type: 'state', ...state });
        } catch (_) {
          send(socket, { type: 'error', message: err.message });
        }
      }
      break;

    case 'respawn':
      if (trackingState.pendingRespawn) {
        if (typeof adapter.respawn === 'function') {
          // Attach mode: click Respawn button
          try { await adapter.respawn(); } catch (_) {}
        } else {
          // Relay mode: chat command
          try { adapter.chat('/kill'); } catch (_) {}
        }
        await sleep(3000);
        if (adapter.health > 0) {
          trackingState.pendingRespawn = false;
        }
      }
      send(socket, { type: 'state', ...getState(adapter, trackingState) });
      break;

    case 'bot_control':
      if (msg.enabled) {
        adapter.enableBotControl();
      } else {
        adapter.disableBotControl();
      }
      send(socket, { type: 'ack', bot_control_active: adapter.botControlActive });
      break;

    case 'disconnect':
      send(socket, { type: 'ack' });
      socket.end();
      break;

    default:
      send(socket, { type: 'error', message: `Unknown command: ${msg.cmd}` });
  }
}

module.exports = { createBridge, send };
