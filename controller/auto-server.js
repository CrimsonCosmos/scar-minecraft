/**
 * AutoServer — automatically starts a dedicated Minecraft server from
 * the user's singleplayer world save.
 *
 * Usage:
 *   node controller/main.js --protocol java --world "~/.minecraft/saves/MyWorld"
 *   node controller/main.js --protocol java --world "MyWorld"  (searches default saves dir)
 *
 * Flow:
 *   1. Reads level.dat to detect the Minecraft version
 *   2. Downloads the matching server.jar from Mojang (cached)
 *   3. Symlinks the world into a temp server directory
 *   4. Configures server.properties (offline-mode, etc.)
 *   5. Starts the server on a random available port
 *   6. Waits for "Done" in stdout
 *   7. Returns { host, port, process, stop() }
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const zlib = require('zlib');
const https = require('https');
const http = require('http');
const { spawn } = require('child_process');
const nbt = require('prismarine-nbt');

// Default Minecraft saves directory per platform
function defaultSavesDir() {
  switch (process.platform) {
    case 'darwin':
      return path.join(os.homedir(), 'Library', 'Application Support', 'minecraft', 'saves');
    case 'win32':
      return path.join(process.env.APPDATA || os.homedir(), '.minecraft', 'saves');
    default: // linux
      return path.join(os.homedir(), '.minecraft', 'saves');
  }
}

/**
 * Resolve a world path. Accepts:
 *   - Absolute path to world directory
 *   - Relative path from CWD
 *   - Just the world name (searches default saves dir)
 */
function resolveWorldPath(input) {
  // Expand ~ to homedir
  if (input.startsWith('~')) {
    input = path.join(os.homedir(), input.slice(1));
  }

  // Absolute or relative path
  const resolved = path.resolve(input);
  if (fs.existsSync(path.join(resolved, 'level.dat'))) {
    return resolved;
  }

  // Try default saves directory
  const inSaves = path.join(defaultSavesDir(), input);
  if (fs.existsSync(path.join(inSaves, 'level.dat'))) {
    return inSaves;
  }

  throw new Error(
    `World not found: "${input}"\n` +
    `Searched:\n  ${resolved}\n  ${inSaves}\n` +
    `Make sure the path contains a level.dat file.`
  );
}

/**
 * Read level.dat and extract the Minecraft version name.
 */
async function readWorldVersion(worldPath) {
  const levelDat = path.join(worldPath, 'level.dat');
  const compressed = fs.readFileSync(levelDat);

  const buf = await new Promise((res, rej) => {
    zlib.gunzip(compressed, (err, data) => err ? rej(err) : res(data));
  });

  const parsed = await new Promise((res, rej) => {
    nbt.parse(buf, (err, data) => err ? rej(err) : res(data));
  });

  const root = parsed.parsed || parsed;
  const d = root.value.Data ? root.value.Data.value : root.value;

  if (d.Version && d.Version.value && d.Version.value.Name) {
    return d.Version.value.Name.value;
  }

  throw new Error('Could not determine Minecraft version from level.dat');
}

/**
 * Download a file from a URL (follows redirects).
 */
function download(url, destPath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);
    const get = url.startsWith('https') ? https.get : http.get;

    get(url, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        file.close();
        fs.unlinkSync(destPath);
        return download(res.headers.location, destPath).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        file.close();
        fs.unlinkSync(destPath);
        return reject(new Error(`Download failed: HTTP ${res.statusCode}`));
      }
      res.pipe(file);
      file.on('finish', () => { file.close(); resolve(); });
      file.on('error', reject);
    }).on('error', (err) => {
      file.close();
      try { fs.unlinkSync(destPath); } catch (_) {}
      reject(err);
    });
  });
}

/**
 * Fetch JSON from a URL.
 */
function fetchJSON(url) {
  return new Promise((resolve, reject) => {
    const get = url.startsWith('https') ? https.get : http.get;
    get(url, (res) => {
      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode} from ${url}`));
      }
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(e); }
      });
    }).on('error', reject);
  });
}

/**
 * Get the server.jar download URL for a specific Minecraft version.
 */
async function getServerJarUrl(version) {
  console.log(`[auto-server] Looking up server jar for Minecraft ${version}...`);

  const manifest = await fetchJSON(
    'https://launchermeta.mojang.com/mc/game/version_manifest.json'
  );

  const versionEntry = manifest.versions.find(v => v.id === version);
  if (!versionEntry) {
    throw new Error(
      `Minecraft version "${version}" not found in Mojang's version manifest.\n` +
      `Available recent versions: ${manifest.versions.slice(0, 10).map(v => v.id).join(', ')}`
    );
  }

  const versionMeta = await fetchJSON(versionEntry.url);
  const serverDl = versionMeta.downloads && versionMeta.downloads.server;
  if (!serverDl || !serverDl.url) {
    throw new Error(`No server download available for Minecraft ${version}`);
  }

  return serverDl.url;
}

/**
 * Ensure we have the server.jar cached for this version.
 * Caches in <project>/.server-cache/<version>/server.jar
 */
async function ensureServerJar(version) {
  const cacheDir = path.join(__dirname, '..', '.server-cache', version);
  const jarPath = path.join(cacheDir, 'server.jar');

  if (fs.existsSync(jarPath)) {
    console.log(`[auto-server] Using cached server jar: ${jarPath}`);
    return jarPath;
  }

  fs.mkdirSync(cacheDir, { recursive: true });

  const url = await getServerJarUrl(version);
  console.log(`[auto-server] Downloading server.jar for ${version}...`);
  await download(url, jarPath);
  console.log(`[auto-server] Downloaded: ${jarPath}`);

  return jarPath;
}

/**
 * Find a Java executable. Prefers JDK 21+ for modern Minecraft.
 */
function findJava() {
  // Check project-local JDK first
  const localJdkDir = path.join(__dirname, '..', '.jdk');
  if (fs.existsSync(localJdkDir)) {
    const entries = fs.readdirSync(localJdkDir);
    for (const entry of entries) {
      const javaBin = path.join(localJdkDir, entry, 'bin', 'java');
      if (fs.existsSync(javaBin)) {
        return javaBin;
      }
      // macOS JDK structure: Contents/Home/bin/java
      const macJavaBin = path.join(localJdkDir, entry, 'Contents', 'Home', 'bin', 'java');
      if (fs.existsSync(macJavaBin)) {
        return macJavaBin;
      }
    }
  }

  // Fall back to system Java
  return 'java';
}

/**
 * Find an available port by binding to port 0.
 */
function findAvailablePort() {
  return new Promise((resolve, reject) => {
    const net = require('net');
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
    srv.on('error', reject);
  });
}

/**
 * Start a Minecraft server from a world save.
 *
 * @param {string} worldPath - Absolute path to the world save directory
 * @param {object} opts
 * @param {number} opts.port - Port to use (0 = auto)
 * @param {string} opts.java - Path to java binary
 * @param {number} opts.timeoutMs - How long to wait for server startup (default 60s)
 * @returns {Promise<{host: string, port: number, version: string, process: ChildProcess, stop: Function}>}
 */
async function startAutoServer(worldPath, opts = {}) {
  worldPath = resolveWorldPath(worldPath);
  const worldName = path.basename(worldPath);

  console.log(`[auto-server] World: ${worldName} (${worldPath})`);

  // Read version from level.dat
  const version = await readWorldVersion(worldPath);
  console.log(`[auto-server] Minecraft version: ${version}`);

  // Ensure server jar is cached
  const jarPath = await ensureServerJar(version);

  // Find Java
  const javaBin = opts.java || findJava();
  console.log(`[auto-server] Java: ${javaBin}`);

  // Pick a port
  const port = opts.port || await findAvailablePort();

  // Create temp server directory
  const serverDir = path.join(os.tmpdir(), `scar-server-${Date.now()}`);
  fs.mkdirSync(serverDir, { recursive: true });

  // Symlink the world save as "world" in the server directory
  const worldLink = path.join(serverDir, 'world');
  fs.symlinkSync(worldPath, worldLink);

  // Copy server.jar
  fs.copyFileSync(jarPath, path.join(serverDir, 'server.jar'));

  // Accept EULA
  fs.writeFileSync(path.join(serverDir, 'eula.txt'), 'eula=true\n');

  // Grant SCAR_Observer op permissions (level 2 = /gamemode, /tp)
  // Compute offline-mode UUID (same algorithm as MC server: UUID v3 from "OfflinePlayer:<name>")
  const { createHash } = require('crypto');
  const md5 = createHash('md5').update('OfflinePlayer:SCAR_Observer').digest();
  md5[6] = (md5[6] & 0x0f) | 0x30; // UUID v3
  md5[8] = (md5[8] & 0x3f) | 0x80;
  const hex = md5.toString('hex');
  const observerUuid = `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
  fs.writeFileSync(path.join(serverDir, 'ops.json'), JSON.stringify([{
    uuid: observerUuid, name: 'SCAR_Observer', level: 2, bypassesPlayerLimit: false,
  }], null, 2) + '\n');

  // Write server.properties
  fs.writeFileSync(path.join(serverDir, 'server.properties'), [
    `server-port=${port}`,
    'server-ip=127.0.0.1',
    'online-mode=false',
    'spawn-protection=0',
    'enable-command-block=true',
    'max-players=2',
    'view-distance=10',
    'simulation-distance=10',
    'level-name=world',
    'gamemode=survival',
    'difficulty=normal',
    'pvp=true',
    'spawn-monsters=true',
    'spawn-animals=true',
    'enable-rcon=false',
    'sync-chunk-writes=true',
    'motd=SCAR Auto Server',
  ].join('\n') + '\n');

  // Start the server
  console.log(`[auto-server] Starting server on port ${port}...`);
  const timeoutMs = opts.timeoutMs || 90000;

  const serverProcess = spawn(javaBin, [
    '-Xmx1G',
    '-Xms512M',
    '-jar', 'server.jar',
    'nogui',
  ], {
    cwd: serverDir,
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  // Wait for "Done" in stdout
  await new Promise((resolve, reject) => {
    let stdout = '';
    const timer = setTimeout(() => {
      reject(new Error(`Server did not start within ${timeoutMs / 1000}s`));
    }, timeoutMs);

    serverProcess.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      stdout += text;
      // Print server output with prefix
      for (const line of text.split('\n').filter(l => l.trim())) {
        console.log(`[mc-server] ${line}`);
      }
      if (stdout.includes('Done')) {
        clearTimeout(timer);
        resolve();
      }
    });

    serverProcess.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      for (const line of text.split('\n').filter(l => l.trim())) {
        console.log(`[mc-server:err] ${line}`);
      }
    });

    serverProcess.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });

    serverProcess.on('exit', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`Server exited with code ${code}`));
      }
    });
  });

  console.log(`[auto-server] Server ready on 127.0.0.1:${port}`);

  return {
    host: '127.0.0.1',
    port,
    version,
    serverDir,
    process: serverProcess,
    stop() {
      console.log('[auto-server] Stopping server...');
      serverProcess.stdin.write('stop\n');
      // Force kill after 10s if graceful shutdown fails
      setTimeout(() => {
        try { serverProcess.kill('SIGKILL'); } catch (_) {}
      }, 10000);
    },
  };
}

/**
 * List available world saves in the default Minecraft directory.
 */
function listWorlds() {
  const savesDir = defaultSavesDir();
  if (!fs.existsSync(savesDir)) return [];

  return fs.readdirSync(savesDir)
    .filter(name => {
      const levelDat = path.join(savesDir, name, 'level.dat');
      return fs.existsSync(levelDat);
    })
    .map(name => ({
      name,
      path: path.join(savesDir, name),
    }));
}

module.exports = { startAutoServer, resolveWorldPath, readWorldVersion, listWorlds };
