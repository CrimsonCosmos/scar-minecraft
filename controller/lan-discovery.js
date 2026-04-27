/**
 * Minecraft Java Edition LAN game discovery.
 *
 * Two detection methods:
 *   1. Log-file parsing (primary, most reliable on macOS)
 *      Reads ~/Library/Application Support/minecraft/logs/latest.log for
 *      "Local game hosted on port XXXXX"
 *
 *   2. UDP multicast (fallback)
 *      Listens on 224.0.2.60:4445 for [MOTD]...[/MOTD][AD]port[/AD]
 *
 * discoverLanGame() races both methods and resolves whichever finds it first.
 */

const EventEmitter = require('events');
const dgram = require('dgram');
const net = require('net');
const fs = require('fs');
const path = require('path');
const os = require('os');

const MULTICAST_GROUP = '224.0.2.60';
const MULTICAST_PORT = 4445;
const PAYLOAD_RE = /\[MOTD\](.*?)\[\/MOTD\]\[AD\](\d+)\[\/AD\]/;
const LOG_PORT_RE = /Local game hosted on port \[?(\d+)\]?/;

/** Resolve the Minecraft log file path for the current OS. */
function getLogPath() {
  switch (process.platform) {
    case 'darwin':
      return path.join(os.homedir(), 'Library/Application Support/minecraft/logs/latest.log');
    case 'win32':
      return path.join(process.env.APPDATA || '', '.minecraft/logs/latest.log');
    default:
      return path.join(os.homedir(), '.minecraft/logs/latest.log');
  }
}

// ──────────────────────── Port verification ────────────────────────

/**
 * Verify a port is actually reachable via TCP connect.
 * Returns a promise that resolves if connectable, rejects if not.
 */
function verifyPort(host, port, timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection({ host, port }, () => {
      socket.destroy();
      resolve();
    });
    socket.on('error', (err) => {
      socket.destroy();
      reject(err);
    });
    socket.setTimeout(timeoutMs, () => {
      socket.destroy();
      reject(new Error('timeout'));
    });
  });
}

// ──────────────────────── Log-file discovery ────────────────────────

/**
 * Discover a LAN game by watching the Minecraft log file.
 * First checks existing content (last 50 lines), then polls for new entries.
 * Verifies the port is reachable before returning (catches stale log entries).
 *
 * @param {number} [timeoutMs=10000]
 * @returns {Promise<{host: string, port: number}>}
 */
function discoverLanFromLog(timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    const logPath = getLogPath();
    let done = false;
    // Track ports we've already tried and found stale
    const stalePorts = new Set();
    const inFlightPorts = new Set();

    const finish = (result) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      clearInterval(poller);
      if (result) resolve(result);
      else reject(new Error('[lan-discovery] no LAN port found in log within ' + timeoutMs + 'ms'));
    };

    // Check if log file exists
    if (!fs.existsSync(logPath)) {
      reject(new Error('[lan-discovery] Minecraft log not found: ' + logPath));
      return;
    }

    // Declare BEFORE finish() can access them (avoids Temporal Dead Zone)
    let poller = null;
    let timer = null;
    let lastSize = 0;

    const checkLog = () => {
      if (done) return;
      try {
        const stat = fs.statSync(logPath);
        const content = fs.readFileSync(logPath, 'utf8');
        const lines = content.split('\n');
        // Search ALL lines from the end to find the MOST RECENT LAN port
        for (let i = lines.length - 1; i >= 0; i--) {
          const match = lines[i].match(LOG_PORT_RE);
          if (match) {
            const port = parseInt(match[1], 10);
            if (stalePorts.has(port) || inFlightPorts.has(port)) continue;
            console.log('[lan-discovery] Found LAN port %d in log file (line %d/%d) — verifying...', port, i + 1, lines.length);
            // Verify port is actually reachable before returning
            inFlightPorts.add(port);
            verifyPort('localhost', port, 2000).then(() => {
              inFlightPorts.delete(port);
              console.log('[lan-discovery] Port %d verified reachable.', port);
              finish({ host: 'localhost', port });
            }).catch(() => {
              inFlightPorts.delete(port);
              console.log('[lan-discovery] Port %d from log is stale (ECONNREFUSED) — skipping.', port);
              stalePorts.add(port);
            });
            return; // Wait for verification, don't search further this pass
          }
        }
        lastSize = stat.size;
      } catch (_) {}
    };

    // Initial check
    checkLog();
    // Don't check `done` here — verification is async, need poller to continue

    // Poll every 500ms for new log entries
    poller = setInterval(checkLog, 500);
    timer = setTimeout(() => finish(null), timeoutMs);
  });
}

// ──────────────────────── UDP multicast discovery ────────────────────────

/** Listens for Minecraft LAN game broadcasts on the local network. */
class LanDiscovery extends EventEmitter {
  constructor() {
    super();
    this._socket = dgram.createSocket({ type: 'udp4', reuseAddr: true });
    this._seen = new Set();
    this._socket.on('error', (err) => {
      console.log('[lan-discovery] socket error:', err.message);
    });
  }

  start() {
    this._socket.bind(MULTICAST_PORT, '0.0.0.0', () => {
      try {
        this._socket.addMembership(MULTICAST_GROUP);
      } catch (e) {
        console.warn('[lan-discovery] Failed to join multicast group:', e.message);
      }
      console.log('[lan-discovery] listening on', MULTICAST_GROUP + ':' + MULTICAST_PORT);
    });

    this._socket.on('message', (msg, rinfo) => {
      const match = msg.toString('utf8').match(PAYLOAD_RE);
      if (!match) return;

      const motd = match[1];
      const port = parseInt(match[2], 10);
      const key = rinfo.address + ':' + port;

      if (this._seen.has(key)) return;
      this._seen.add(key);

      console.log('[lan-discovery] found LAN game at %s:%d — %s', rinfo.address, port, motd);
      this.emit('found', { host: rinfo.address, port, motd });
    });
  }

  stop() {
    try { this._socket.dropMembership(MULTICAST_GROUP); } catch (_) {}
    try { this._socket.close(); } catch (_) {}
  }
}

function discoverLanFromMulticast(timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    const discovery = new LanDiscovery();
    let timer;

    const cleanup = () => {
      clearTimeout(timer);
      discovery.stop();
    };

    discovery.once('found', (info) => {
      cleanup();
      resolve(info);
    });

    timer = setTimeout(() => {
      cleanup();
      reject(new Error('[lan-discovery] no LAN game found via multicast within ' + timeoutMs + 'ms'));
    }, timeoutMs);

    discovery.start();
  });
}

// ──────────────────────── Process-based discovery ────────────────────────

/**
 * Discover a LAN game by finding Java processes with listening TCP ports,
 * then pinging each to check if it's a Minecraft server.
 *
 * MC 26.1+ may not log "Local game hosted on port" and multicast can be
 * blocked by firewall. This method is the most reliable fallback.
 *
 * @param {number} [timeoutMs=10000]
 * @returns {Promise<{host: string, port: number}>}
 */
function discoverLanFromProcess(timeoutMs = 10000) {
  const { execSync } = require('child_process');
  const mc = require('minecraft-protocol');

  return new Promise((resolve, reject) => {
    let done = false;
    const timer = setTimeout(() => {
      if (!done) { done = true; reject(new Error('[lan-discovery] no LAN game found via process scan')); }
    }, timeoutMs);

    // Find Java listening ports
    let ports = [];
    try {
      if (process.platform === 'win32') {
        const out = execSync('netstat -ano -p tcp', { timeout: 3000 }).toString();
        const javaLines = out.split('\n').filter(l => l.includes('LISTENING'));
        for (const line of javaLines) {
          const m = line.match(/:(\d+)\s+.*LISTENING/);
          if (m) ports.push(parseInt(m[1], 10));
        }
      } else {
        // macOS / Linux: lsof to find Java listening ports
        const out = execSync('lsof -i -P -n 2>/dev/null | grep java | grep LISTEN', { timeout: 3000 }).toString();
        for (const line of out.split('\n')) {
          const m = line.match(/:(\d+)\s+\(LISTEN\)/);
          if (m) ports.push(parseInt(m[1], 10));
        }
      }
    } catch (_) {}

    // Deduplicate and filter out unlikely ports (< 1024 or common non-MC ports)
    ports = [...new Set(ports)].filter(p => p >= 1024 && p !== 3001);

    if (ports.length === 0) {
      clearTimeout(timer);
      reject(new Error('[lan-discovery] no Java listening ports found'));
      return;
    }

    console.log('[lan-discovery] Found Java listening ports:', ports.join(', '));

    // Ping each port to check if it's a Minecraft server
    let pending = ports.length;
    for (const port of ports) {
      mc.ping({ host: 'localhost', port, closeTimeout: 3000 }, (err, result) => {
        if (done) return;
        pending--;
        if (!err && result && result.version) {
          done = true;
          clearTimeout(timer);
          console.log('[lan-discovery] Found MC server on port %d via process scan (%s)', port, result.version.name);
          resolve({ host: 'localhost', port });
        } else if (pending === 0) {
          done = true;
          clearTimeout(timer);
          reject(new Error('[lan-discovery] pinged ' + ports.length + ' Java ports, none responded as MC'));
        }
      });
    }
  });
}

// ──────────────────────── Combined discovery ────────────────────────

/**
 * Discover a LAN game using log-file, UDP multicast, and process scan.
 * Resolves as soon as any method finds a game.
 *
 * @param {number} [timeoutMs=10000]
 * @returns {Promise<{host: string, port: number}>}
 */
function discoverLanGame(timeoutMs = 10000) {
  return Promise.any([
    discoverLanFromLog(timeoutMs),
    discoverLanFromMulticast(timeoutMs),
    discoverLanFromProcess(timeoutMs),
  ]).catch(() => {
    throw new Error('[lan-discovery] no LAN game found within ' + timeoutMs + 'ms (tried log file + multicast + process scan)');
  });
}

module.exports = { LanDiscovery, discoverLanGame, discoverLanFromLog, discoverLanFromMulticast, discoverLanFromProcess, getLogPath };
