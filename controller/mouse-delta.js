/**
 * NativeMouseDelta — Node.js wrapper for the macOS CGEvent relative mouse helper.
 *
 * Spawns controller/native/mouse-delta as a persistent subprocess and pipes
 * "dx dy\n" commands. The helper posts HID-level mouse events with delta fields
 * that survive Minecraft's pointer lock.
 *
 * macOS-only. On other platforms, start() is a no-op and move() silently returns.
 */

const path = require('path');
const fs = require('fs');
const { spawn, execSync } = require('child_process');

class NativeMouseDelta {
  constructor(opts = {}) {
    this.sensitivity = opts.mouseSensitivity || 400;
    this._proc = null;
    this._available = false;
  }

  start() {
    if (process.platform !== 'darwin') {
      console.warn('[mouse-delta] Native mouse delta only supported on macOS.');
      return;
    }

    const binaryPath = path.join(__dirname, 'native', 'mouse-delta');
    const sourcePath = binaryPath + '.c';

    if (!fs.existsSync(binaryPath)) {
      if (!fs.existsSync(sourcePath)) {
        console.warn('[mouse-delta] Source not found:', sourcePath);
        return;
      }
      try {
        console.log('[mouse-delta] Compiling native helper...');
        execSync(`cc -O2 -o "${binaryPath}" "${sourcePath}" -framework ApplicationServices`);
        console.log('[mouse-delta] Compiled successfully.');
      } catch (e) {
        console.warn('[mouse-delta] Could not compile native helper:', e.message);
        return;
      }
    }

    this._proc = spawn(binaryPath, [], { stdio: ['pipe', 'pipe', 'ignore'] });
    this._proc.on('error', (err) => {
      console.warn('[mouse-delta] Process error:', err.message);
      this._available = false;
    });
    this._proc.on('exit', (code) => {
      if (this._available) {
        console.warn('[mouse-delta] Process exited with code', code);
      }
      this._available = false;
    });
    this._available = true;
    console.log('[mouse-delta] Native mouse delta ready.');
  }

  /**
   * Send a relative mouse movement.
   * @param {number} deltaYawRad - Yaw delta in radians (positive = right)
   * @param {number} deltaPitchRad - Pitch delta in radians (positive = down)
   */
  move(deltaYawRad, deltaPitchRad) {
    if (!this._available || !this._proc) return;
    const dx = Math.round(deltaYawRad * this.sensitivity);
    const dy = Math.round(deltaPitchRad * this.sensitivity);
    if (dx === 0 && dy === 0) return;
    this._proc.stdin.write(`${dx} ${dy}\n`);
  }

  stop() {
    this._available = false;
    if (this._proc) {
      this._proc.kill();
      this._proc = null;
    }
  }
}

module.exports = { NativeMouseDelta };
