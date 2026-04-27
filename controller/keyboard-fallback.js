/**
 * KeyboardFallback — OS-level keyboard/mouse simulation via @nut-tree-fork/nut-js.
 *
 * Safety net for edge cases where packet injection fails (upstream disconnects,
 * malformed packets rejected, etc.). Simulates real keyboard/mouse input to the
 * Minecraft client window.
 *
 * Works with both Java Edition and Bedrock Edition — same default keybinds.
 * Pass keyOverrides to customize for non-default key configurations.
 *
 * Requires:
 * - macOS: Terminal must have Accessibility access (System Preferences → Privacy)
 * - Windows: Run as administrator (optional, usually works without)
 * - Linux: xdotool or similar backend
 *
 * Enabled with --keyboard-fallback CLI flag. Never used unless packet injection
 * actually throws an error.
 */

// Lazy-load nut-js so the module doesn't fail if not installed
// (only needed when --keyboard-fallback is enabled)
let keyboard, mouse, Key, Button, screen, getWindows;

function loadNutJs() {
  if (keyboard) return; // Already loaded
  try {
    const nut = require('@nut-tree-fork/nut-js');
    keyboard = nut.keyboard;
    mouse = nut.mouse;
    Key = nut.Key;
    Button = nut.Button;
    screen = nut.screen;
    // Window management
    getWindows = nut.getWindows || null;
  } catch (e) {
    throw new Error(
      `@nut-tree-fork/nut-js is required for keyboard fallback. Install with: npm install @nut-tree-fork/nut-js\n${e.message}`
    );
  }
}

// Default key mappings (same for Java and Bedrock)
const DEFAULT_KEY_MAP = {
  forward: 'W',
  back: 'S',
  left: 'A',
  right: 'D',
  jump: 'Space',
  sprint: 'LeftControl',
  sneak: 'LeftShift',
  dropItem: 'Q',
  swapOffhand: 'F',
  openInventory: 'E',
};

// Hotbar slot (0-8) → number key (1-9)
const HOTBAR_KEYS = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7', 'Num8', 'Num9'];

class KeyboardFallback {
  /**
   * @param {object} opts
   * @param {number} opts.mouseSensitivity - Pixels per radian for mouse look (default 400)
   * @param {string} opts.protocol - 'bedrock' or 'java' (default 'bedrock')
   * @param {object} opts.keyOverrides - Override default key mappings, e.g. { sprint: 'LeftAlt' }
   */
  constructor(opts = {}) {
    this.mouseSensitivity = opts.mouseSensitivity || 400;
    this.protocol = opts.protocol || 'bedrock';

    // Merge key overrides onto defaults
    this._keyMap = { ...DEFAULT_KEY_MAP, ...(opts.keyOverrides || {}) };

    // Track which keys are currently held down
    this._heldKeys = new Set();

    // Track which mouse buttons are currently held down
    this._heldButtons = new Set();

    // Load nut-js eagerly so we fail fast if not installed
    loadNutJs();

    // Configure nut-js for low-latency input
    keyboard.config.autoDelayMs = 0;
    mouse.config.autoDelayMs = 0;
    mouse.config.mouseSpeed = 10000; // Very fast mouse moves (pixels/sec)

    console.log(`[keyboard-fallback] Initialized (${this.protocol}). Mouse sensitivity: ${this.mouseSensitivity} px/rad`);
    console.log('[keyboard-fallback] NOTE: On macOS, ensure Accessibility access is granted to your terminal app.');
  }

  /**
   * Set a movement control state (hold or release a key).
   * @param {string} key - Control name: forward, back, left, right, jump, sprint, sneak
   * @param {boolean} val - true to hold, false to release
   */
  async setControlState(key, val) {
    const keyName = this._keyMap[key];
    if (!keyName || !Key[keyName]) {
      console.warn(`[keyboard-fallback] Unknown control key: ${key}`);
      return;
    }

    try {
      if (val) {
        if (!this._heldKeys.has(keyName)) {
          await keyboard.pressKey(Key[keyName]);
          this._heldKeys.add(keyName);
        }
      } else {
        if (this._heldKeys.has(keyName)) {
          await keyboard.releaseKey(Key[keyName]);
          this._heldKeys.delete(keyName);
        }
      }
    } catch (e) {
      console.warn(`[keyboard-fallback] setControlState(${key}, ${val}) failed:`, e.message);
    }
  }

  /**
   * Release all held keys and mouse buttons.
   */
  async clearControlStates() {
    for (const keyName of this._heldKeys) {
      try {
        await keyboard.releaseKey(Key[keyName]);
      } catch (_) {}
    }
    this._heldKeys.clear();
    for (const btn of this._heldButtons) {
      try {
        await mouse.releaseButton(btn);
      } catch (_) {}
    }
    this._heldButtons.clear();
  }

  /**
   * Left-click attack.
   */
  async attack() {
    try {
      await mouse.click(Button.LEFT);
    } catch (e) {
      console.warn('[keyboard-fallback] attack() click failed:', e.message);
    }
  }

  /**
   * Swing arm (left click, no entity targeting).
   */
  async swingArm() {
    try {
      await mouse.click(Button.LEFT);
    } catch (e) {
      console.warn('[keyboard-fallback] swingArm() click failed:', e.message);
    }
  }

  /**
   * Use/activate item (right click tap).
   */
  async activateItem() {
    try {
      await mouse.click(Button.RIGHT);
    } catch (e) {
      console.warn('[keyboard-fallback] activateItem() click failed:', e.message);
    }
  }

  /**
   * Press and hold right-click (start using item: charge bow, raise shield, eat).
   */
  async pressUseItem() {
    try {
      if (!this._heldButtons.has(Button.RIGHT)) {
        await mouse.pressButton(Button.RIGHT);
        this._heldButtons.add(Button.RIGHT);
      }
    } catch (e) {
      console.warn('[keyboard-fallback] pressUseItem() failed:', e.message);
    }
  }

  /**
   * Release right-click (fire bow, lower shield, stop eating).
   */
  async releaseUseItem() {
    try {
      if (this._heldButtons.has(Button.RIGHT)) {
        await mouse.releaseButton(Button.RIGHT);
        this._heldButtons.delete(Button.RIGHT);
      }
    } catch (e) {
      console.warn('[keyboard-fallback] releaseUseItem() failed:', e.message);
    }
  }

  /**
   * Relative mouse move for look direction.
   * Converts delta yaw/pitch (radians) to pixel offsets.
   *
   * @param {number} deltaYaw - Yaw change in radians (positive = right)
   * @param {number} deltaPitch - Pitch change in radians (positive = down)
   */
  async look(deltaYaw, deltaPitch) {
    const dx = Math.round(deltaYaw * this.mouseSensitivity);
    const dy = Math.round(deltaPitch * this.mouseSensitivity);

    if (dx === 0 && dy === 0) return;

    try {
      const currentPos = await mouse.getPosition();
      await mouse.setPosition({
        x: currentPos.x + dx,
        y: currentPos.y + dy,
      });
    } catch (e) {
      console.warn('[keyboard-fallback] look() mouse move failed:', e.message);
    }
  }

  /**
   * Select a hotbar slot by pressing number key 1-9.
   * @param {number} slot - Slot index 0-8
   */
  async setQuickBarSlot(slot) {
    if (slot < 0 || slot > 8) return;
    const keyName = HOTBAR_KEYS[slot];
    if (!keyName || !Key[keyName]) return;

    try {
      await keyboard.type(Key[keyName]);
    } catch (e) {
      console.warn(`[keyboard-fallback] setQuickBarSlot(${slot}) failed:`, e.message);
    }
  }

  /**
   * Drop the currently held item (Q key).
   * Works on both Java and Bedrock.
   */
  async dropItem() {
    const keyName = this._keyMap.dropItem;
    if (!keyName || !Key[keyName]) return;
    try {
      await keyboard.type(Key[keyName]);
    } catch (e) {
      console.warn('[keyboard-fallback] dropItem() failed:', e.message);
    }
  }

  /**
   * Swap item to offhand (F key).
   * Java Edition feature (Bedrock added offhand but no swap key by default).
   */
  async swapOffhand() {
    const keyName = this._keyMap.swapOffhand;
    if (!keyName || !Key[keyName]) return;
    try {
      await keyboard.type(Key[keyName]);
    } catch (e) {
      console.warn('[keyboard-fallback] swapOffhand() failed:', e.message);
    }
  }

  /**
   * Open/close inventory (E key).
   * Works on both Java and Bedrock.
   */
  async openInventory() {
    const keyName = this._keyMap.openInventory;
    if (!keyName || !Key[keyName]) return;
    try {
      await keyboard.type(Key[keyName]);
    } catch (e) {
      console.warn('[keyboard-fallback] openInventory() failed:', e.message);
    }
  }

  /**
   * Attempt to bring the Minecraft window to the foreground.
   * Required because keyboard/mouse input goes to the focused window.
   * Searches for both Java ("Minecraft") and Bedrock ("Minecraft") window titles.
   */
  async focusWindow() {
    try {
      if (!getWindows) {
        // nut-js window API may not be available on all platforms
        return true;
      }
      const windows = await getWindows();
      for (const win of windows) {
        const title = await win.getTitle();
        if (title && title.toLowerCase().includes('minecraft')) {
          await win.focus();
          return true;
        }
      }
      console.warn('[keyboard-fallback] Could not find Minecraft window to focus.');
      return false;
    } catch (e) {
      console.warn('[keyboard-fallback] focusWindow() failed:', e.message);
      return false;
    }
  }
}

module.exports = { KeyboardFallback };
