/**
 * LAN Automation — automates the "Open to LAN" flow in Minecraft Java Edition.
 *
 * Uses mouse clicks at calculated screen positions to navigate the pause menu.
 * Falls back to prompting the user for manual action if automation fails.
 *
 * Minecraft Java 1.20.x Pause Menu layout (in GUI units, relative to center):
 *   "Back to Game"                    — full width, y = h/4 + 24 - 16
 *   "Advancements" | "Statistics"     — half width,  y = h/4 + 48 - 16
 *   "Options..."   | "Open to LAN"   — half width,  y = h/4 + 72 - 16
 *   "Save and Quit to Title"         — full width,  y = h/4 + 96 - 16
 *
 * The "Open to LAN" button is the RIGHT button of the 3rd row.
 * Its center in GUI units: (w/2 + 53, h/4 + 66)
 * In window pixels: multiply by GUI scale (typically 2 on Retina Mac).
 */

const { execSync } = require('child_process');
const { sleep } = require('./utils');

// Lazy-load nut-js
let keyboard, mouse, Key, Button, screen;

function loadNutJs() {
  if (keyboard) return;
  const nut = require('@nut-tree-fork/nut-js');
  keyboard = nut.keyboard;
  mouse = nut.mouse;
  Key = nut.Key;
  Button = nut.Button;
  screen = nut.screen;
  keyboard.config.autoDelayMs = 0;
  mouse.config.autoDelayMs = 0;
  mouse.config.mouseSpeed = 8000;
}

/**
 * Focus the Minecraft window.
 * On macOS: uses AppleScript for reliable window activation.
 * On other platforms: uses nut-js getWindows.
 */
async function focusMinecraft() {
  if (process.platform === 'darwin') {
    try {
      // Try activating by window title containing "Minecraft"
      execSync(`osascript -e '
        tell application "System Events"
          set mcProcs to every process whose name contains "java"
          repeat with p in mcProcs
            set wins to every window of p
            repeat with w in wins
              if name of w contains "Minecraft" then
                set frontmost of p to true
                return
              end if
            end repeat
          end repeat
        end tell
      '`, { timeout: 3000 });
      console.log('[lan-automation] Focused Minecraft via AppleScript.');
      return true;
    } catch (_) {
      // Try by app name
      try {
        execSync(`osascript -e 'tell application "Minecraft" to activate'`, { timeout: 2000 });
        console.log('[lan-automation] Focused Minecraft by app name.');
        return true;
      } catch (_2) {}
    }
  }

  // Fallback: nut-js getWindows
  try {
    loadNutJs();
    const nut = require('@nut-tree-fork/nut-js');
    const getWindows = nut.getWindows;
    if (!getWindows) return false;
    const windows = await getWindows();
    for (const win of windows) {
      const title = await win.getTitle();
      if (title && title.toLowerCase().includes('minecraft')) {
        await win.focus();
        console.log('[lan-automation] Focused Minecraft via nut-js.');
        return true;
      }
    }
  } catch (_) {}

  console.warn('[lan-automation] Could not focus Minecraft window.');
  return false;
}

/**
 * Get the Minecraft window bounds on macOS.
 * Returns { x, y, width, height } or null.
 */
function getMcWindowBounds() {
  if (process.platform !== 'darwin') return null;
  try {
    const result = execSync(`osascript -e '
      tell application "System Events"
        set mcProcs to every process whose name contains "java"
        repeat with p in mcProcs
          set wins to every window of p
          repeat with w in wins
            if name of w contains "Minecraft" then
              set pos to position of w
              set sz to size of w
              return "" & (item 1 of pos) & "," & (item 2 of pos) & "," & (item 1 of sz) & "," & (item 2 of sz)
            end if
          end repeat
        end repeat
      end tell
    '`, { timeout: 3000 }).toString().trim();
    const parts = result.split(',').map(Number);
    if (parts.length === 4 && parts.every(n => !isNaN(n))) {
      return { x: parts[0], y: parts[1], width: parts[2], height: parts[3] };
    }
  } catch (_) {}
  return null;
}

/**
 * Automate the pause menu → Open to LAN → Start LAN World flow using mouse clicks.
 *
 * Strategy:
 *  1. Focus Minecraft window
 *  2. Press Escape to open pause menu
 *  3. Click "Open to LAN" button at calculated position
 *  4. Click "Start LAN World" button
 *  5. Press Escape to dismiss chat notification
 */
async function openToLan() {
  loadNutJs();

  console.log('[lan-automation] Focusing Minecraft...');
  const focused = await focusMinecraft();
  await sleep(400);

  // Get window bounds for accurate click positioning
  const bounds = getMcWindowBounds();
  let cx, cy, guiScale;

  if (bounds) {
    cx = bounds.x + Math.round(bounds.width / 2);
    cy = bounds.y + Math.round(bounds.height / 2);
    // Estimate GUI scale (MC auto picks largest integer that fits)
    // MC needs at least 320px per GUI unit, so scale = floor(min(w,h) / 240)
    guiScale = Math.max(1, Math.min(4, Math.floor(Math.min(bounds.width, bounds.height) / 240)));
    console.log(`[lan-automation] Window: ${bounds.width}x${bounds.height} at (${bounds.x},${bounds.y}), GUI scale ~${guiScale}`);
  } else {
    // Fallback: use screen center
    const sw = await screen.width();
    const sh = await screen.height();
    cx = Math.round(sw / 2);
    cy = Math.round(sh / 2);
    guiScale = 2;
    console.log(`[lan-automation] Using screen center (${cx}, ${cy}), guessing GUI scale ${guiScale}`);
  }

  // --- Open pause menu ---
  console.log('[lan-automation] Opening pause menu (Escape)...');
  await keyboard.type(Key.Escape);
  await sleep(600);

  // --- Click "Open to LAN" ---
  // In MC GUI units: x = width/2 + 53 (center of right half-button), y = height/4 + 66
  // The window height in GUI units = window_height_px / guiScale
  // So in pixels: y = bounds_center_y + (66 - height_gui/4) * guiScale ... this gets complicated.
  //
  // Simpler: from screen center, the "Open to LAN" button offset depends on GUI scale:
  //   MC centers the menu vertically. The menu spans from h/4+8 to h/4+116 in GUI units.
  //   Menu center in GUI units = h/4 + 62 = screen_center_gui + (h/4 + 62 - h/2)
  //                            = screen_center_gui - h/4 + 62
  //
  // "Open to LAN" center in GUI units from screen center:
  //   y_offset_gui = -(h_gui/4) + 66   (where h_gui = window_height / guiScale)
  //   x_offset_gui = +53
  //
  // In pixels:
  //   y_offset_px = (-(window_height / guiScale / 4) + 66) * guiScale
  //               = -window_height/4 + 66*guiScale
  //   x_offset_px = 53 * guiScale

  const h = bounds ? bounds.height : cy * 2;
  const yOffsetPx = Math.round(-h / 4 + 66 * guiScale);
  const xOffsetPx = Math.round(53 * guiScale);

  const lanBtnX = cx + xOffsetPx;
  const lanBtnY = cy + yOffsetPx;

  console.log(`[lan-automation] Clicking "Open to LAN" at (${lanBtnX}, ${lanBtnY})...`);
  await mouse.setPosition({ x: lanBtnX, y: lanBtnY });
  await sleep(100);
  await mouse.click(Button.LEFT);
  await sleep(700);

  // --- Toggle "Allow Cheats" to ON ---
  // On the LAN settings screen, "Allow Cheats: OFF" is at y ~ -24 GUI units from center
  const cheatsY = cy + Math.round(-24 * guiScale);
  console.log(`[lan-automation] Clicking "Allow Cheats" at (${cx}, ${cheatsY})...`);
  await mouse.setPosition({ x: cx, y: cheatsY });
  await sleep(100);
  await mouse.click(Button.LEFT);
  await sleep(300);

  // --- Click "Start LAN World" ---
  // Layout (GUI units from screen center):
  //   "Allow Cheats: ON"   — y ~ -24 (just toggled)
  //   "Game Mode: Survival" — y ~ 0
  //   "Start LAN World"    — y ~ +40 (full width, centered)
  const startBtnY = cy + Math.round(40 * guiScale);
  console.log(`[lan-automation] Clicking "Start LAN World" at (${cx}, ${startBtnY})...`);
  await mouse.setPosition({ x: cx, y: startBtnY });
  await sleep(100);
  await mouse.click(Button.LEFT);
  await sleep(500);

  // Dismiss chat notification by pressing Escape
  console.log('[lan-automation] Dismissing chat (Escape)...');
  await keyboard.type(Key.Escape);
  await sleep(300);
}

/**
 * Open to LAN with automatic verification + manual fallback.
 *
 * @param {function} discoverFn - Discovery function (discoverLanGame)
 * @param {number} [maxAttempts=3] - Auto attempts before prompting user
 * @returns {Promise<{host: string, port: number}>}
 */
/**
 * @param {function} discoverFn - Discovery function (discoverLanGame)
 * @param {object} [options]
 * @param {function} [options.log] - Logger function(msg). Defaults to console.log.
 * @returns {Promise<{host: string, port: number}>}
 */
async function openToLanWithRetry(discoverFn, options = {}) {
  const log = options.log || console.log.bind(console);

  // First, check if LAN is already open (10s check — generous for log parsing)
  log('Checking if LAN is already open...');
  try {
    const existing = await discoverFn(10000);
    // Double-check port is actually reachable (catches stale log entries)
    const net = require('net');
    await new Promise((resolve, reject) => {
      const sock = net.createConnection({ host: existing.host || 'localhost', port: existing.port }, () => {
        sock.destroy();
        resolve();
      });
      sock.on('error', (err) => { sock.destroy(); reject(err); });
      sock.setTimeout(2000, () => { sock.destroy(); reject(new Error('timeout')); });
    });
    log(`LAN already open on port ${existing.port}!`);
    return existing;
  } catch (_) {
    log('No existing LAN game found (or port is stale).');
  }

  // Try ONE automated attempt (pressing Escape is risky — skip if it fails)
  log('Trying automated Open to LAN (1 attempt)...');
  try {
    await openToLan();
    const result = await discoverFn(5000);
    log(`LAN game found on port ${result.port}`);
    return result;
  } catch (e) {
    log(`Automated attempt failed: ${e.message}`);
    // Press Escape to close any menus we may have opened
    try {
      loadNutJs();
      await keyboard.type(Key.Escape);
      await sleep(300);
    } catch (_) {}
  }

  // Manual fallback — don't press Escape anymore, just ask the user
  log('--- Please open your world to LAN manually ---');
  log('1. In Minecraft, press Escape to open the pause menu');
  log('2. Click "Open to LAN"');
  log('3. Leave the port blank (let MC auto-pick)');
  log('4. Click "Start LAN World"');
  log('SCAR will detect it automatically once LAN is open.');
  log('Waiting for LAN open (60s timeout)...');

  try {
    const result = await discoverFn(60000);
    log(`LAN game detected on port ${result.port}`);
    return result;
  } catch (_) {
    throw new Error('LAN game was not detected. Please ensure Minecraft is open and the world is opened to LAN.');
  }
}

module.exports = { openToLan, openToLanWithRetry, focusMinecraft, getMcWindowBounds };
