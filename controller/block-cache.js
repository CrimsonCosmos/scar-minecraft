/**
 * Block cache — parses Bedrock sub-chunk data to provide blockAt().
 *
 * Loads the canonical block-state→runtime-ID table from minecraft-data
 * (15k+ entries), then parses level_chunk and update_block packets to
 * build a spatial block index.  Chunks far from the player are pruned
 * automatically.
 *
 * Bedrock sub-chunk format (version 8/9):
 *   u8 version
 *   u8 storage_count
 *   Per storage layer:
 *     u8 header = (bits_per_block << 1) | palette_type
 *     Padded bit array (4096 blocks)
 *     Palette: unsigned varint count, then signed varints (runtime IDs)
 */

const mcData = require('minecraft-data');
const bpOptions = require('bedrock-protocol/src/options');

class BlockCache {
  constructor(options = {}) {
    // Runtime ID → block name.
    // Loaded from minecraft-data for the protocol version bedrock-protocol targets.
    this._palette = new Map();

    // Chunk columns: "cx,cz" → Array(24) of SubChunk | null
    // 24 sub-chunks cover y [-64, 320) in modern Bedrock (overworld)
    this._chunks = new Map();

    // Individual block overrides from update_block packets
    this._blockOverrides = new Map(); // "x,y,z" → blockName

    // Prune chunks beyond this radius (in chunk coords)
    this._pruneRadius = options.pruneRadius || 8;
    this._lastPruneChunk = null;

    this._ready = false;

    // Load palette immediately from minecraft-data
    this._loadPaletteFromMcData();
  }

  /**
   * Load the canonical runtime block palette from minecraft-data.
   * The blockStates array index IS the runtime ID.
   */
  _loadPaletteFromMcData() {
    // Try versions in preference order: current protocol → recent fallbacks
    const currentVer = bpOptions.CURRENT_VERSION;
    const candidates = [
      `bedrock_${currentVer}`,
      'bedrock_1.21.80',
      'bedrock_1.21.60',
      'bedrock_1.21.0',
      'bedrock_1.20.80',
    ];

    for (const ver of candidates) {
      try {
        const data = mcData(ver);
        if (data && data.blockStates && data.blockStates.length > 0) {
          for (let i = 0; i < data.blockStates.length; i++) {
            const name = (data.blockStates[i].name || '').replace('minecraft:', '');
            this._palette.set(i, name || 'unknown');
          }
          this._ready = true;
          console.log(`[block-cache] Loaded ${this._palette.size} block states from minecraft-data (${ver}).`);
          return;
        }
      } catch (_) {}
    }
    console.warn('[block-cache] Could not load block palette from minecraft-data. blockAt() will not work until start_game.');
  }

  /**
   * Extend palette with any custom blocks from start_game packet.
   * Vanilla Realms typically send no custom blocks, but modded servers might.
   */
  handleStartGame(params) {
    const entries = params.block_properties || [];
    let added = 0;
    for (const entry of entries) {
      const name = (entry.name || '').replace('minecraft:', '');
      if (!name) continue;
      // Custom blocks get runtime IDs after the standard palette
      const rid = this._palette.size;
      this._palette.set(rid, name);
      added++;
    }
    if (added > 0) {
      console.log(`[block-cache] Added ${added} custom blocks from start_game. Total: ${this._palette.size}.`);
    }
    this._ready = true;
  }

  /**
   * Handle update_block packet — single block change.
   */
  handleUpdateBlock(params) {
    if (!params.position) return;
    const { x, y, z } = params.position;
    const rid = params.block_runtime_id;
    const name = this._palette.get(rid) || 'unknown';
    this._blockOverrides.set(`${x},${y},${z}`, name);
  }

  /**
   * Handle level_chunk packet — full chunk column.
   */
  handleLevelChunk(params) {
    if (!this._ready) return;
    try {
      const cx = params.x;
      const cz = params.z;
      const subChunkCount = params.sub_chunk_count || 0;
      const payload = params.payload || params.data;

      // sub_chunk_count <= 0 means sub-chunk request system is in use
      if (subChunkCount <= 0 || !Buffer.isBuffer(payload) || payload.length === 0) return;

      const column = new Array(24).fill(null);
      let offset = 0;

      for (let i = 0; i < subChunkCount && offset < payload.length; i++) {
        const result = this._parseSubChunk(payload, offset);
        if (result) {
          column[i] = result.subChunk;
          offset = result.offset;
        } else {
          break;
        }
      }

      this._chunks.set(`${cx},${cz}`, column);
    } catch (_) {
      // Graceful fallback — chunk parsing is best-effort
    }
  }

  /**
   * Handle sub_chunk packet (1.18+ sub-chunk request responses).
   */
  handleSubChunk(params) {
    if (!this._ready) return;
    try {
      const centerX = params.center_x || 0;
      const centerZ = params.center_z || 0;

      for (const entry of (params.entries || [])) {
        // result=1 or 'success' means data is valid
        if (entry.result !== 1 && entry.result !== 'success') continue;
        if (!entry.data || entry.data.length === 0) continue;

        const cx = centerX + (entry.dx || 0);
        const cz = centerZ + (entry.dz || 0);
        const subY = entry.dy || 0;

        const result = this._parseSubChunk(entry.data, 0);
        if (!result) continue;

        const key = `${cx},${cz}`;
        let column = this._chunks.get(key);
        if (!column) {
          column = new Array(24).fill(null);
          this._chunks.set(key, column);
        }

        // subY is signed: -4 = y[-64,-48), 0 = y[0,16), etc.
        const idx = subY + 4;
        if (idx >= 0 && idx < 24) {
          column[idx] = result.subChunk;
        }
      }
    } catch (_) {}
  }

  /**
   * Look up block at world position. Returns { name: "stone" } or null.
   */
  blockAt(pos) {
    if (!pos) return null;
    const x = Math.floor(pos.x);
    const y = Math.floor(pos.y);
    const z = Math.floor(pos.z);

    // Check overrides first (from update_block)
    const override = this._blockOverrides.get(`${x},${y},${z}`);
    if (override !== undefined) {
      return { name: override };
    }

    // Look up in chunk column
    const cx = x >> 4;
    const cz = z >> 4;
    const column = this._chunks.get(`${cx},${cz}`);
    if (!column) return null;

    // Sub-chunk index: y = -64 → idx 0, y = 0 → idx 4
    const subIdx = (y + 64) >> 4;
    if (subIdx < 0 || subIdx >= 24) return null;
    const sub = column[subIdx];
    if (!sub) return null;

    // Block index within sub-chunk (XZY order in Bedrock)
    const lx = x & 0xF;
    const ly = y & 0xF;
    const lz = z & 0xF;
    const blockIdx = (lx << 8) | (lz << 4) | ly;

    const paletteIdx = sub.blocks[blockIdx];
    if (paletteIdx === undefined) return null;

    const name = sub.palette[paletteIdx];
    return name ? { name } : null;
  }

  /**
   * Prune chunks far from player. Call periodically (e.g., on position update).
   */
  prune(playerPos) {
    if (!playerPos) return;
    const pcx = Math.floor(playerPos.x) >> 4;
    const pcz = Math.floor(playerPos.z) >> 4;

    // Only prune on significant movement (2+ chunks)
    if (this._lastPruneChunk) {
      const dx = pcx - this._lastPruneChunk.x;
      const dz = pcz - this._lastPruneChunk.z;
      if (dx * dx + dz * dz < 4) return;
    }
    this._lastPruneChunk = { x: pcx, z: pcz };

    const r = this._pruneRadius;
    for (const key of this._chunks.keys()) {
      const [cx, cz] = key.split(',').map(Number);
      if (Math.abs(cx - pcx) > r || Math.abs(cz - pcz) > r) {
        this._chunks.delete(key);
      }
    }

    for (const key of this._blockOverrides.keys()) {
      const [bx, , bz] = key.split(',').map(Number);
      if (Math.abs((bx >> 4) - pcx) > r || Math.abs((bz >> 4) - pcz) > r) {
        this._blockOverrides.delete(key);
      }
    }
  }

  get stats() {
    return {
      paletteSize: this._palette.size,
      chunksLoaded: this._chunks.size,
      blockOverrides: this._blockOverrides.size,
      ready: this._ready,
    };
  }

  // ---- Internal sub-chunk parsing ----

  _parseSubChunk(buffer, offset) {
    if (offset >= buffer.length) return null;

    const version = buffer[offset++];

    // Version 1: legacy format (4096 byte block IDs + 4096 nibble data)
    if (version === 1) {
      return this._parseLegacySubChunk(buffer, offset);
    }

    // Version 8/9: modern paletted format
    if (version !== 8 && version !== 9) return null;

    if (offset >= buffer.length) return null;
    const storageCount = buffer[offset++];
    if (storageCount === 0) return null;

    // Parse first storage layer (primary blocks)
    const result = this._parseBlockStorage(buffer, offset);
    if (!result) return null;

    // Skip remaining layers (waterlogging, etc.)
    let finalOffset = result.offset;
    for (let i = 1; i < storageCount; i++) {
      const skip = this._skipBlockStorage(buffer, finalOffset);
      if (skip === null) break;
      finalOffset = skip;
    }

    return { subChunk: result.storage, offset: finalOffset };
  }

  _parseLegacySubChunk(buffer, offset) {
    if (offset + 4096 > buffer.length) return null;

    const blocks = new Uint16Array(4096);
    const palette = [];
    const paletteMap = new Map();

    for (let i = 0; i < 4096; i++) {
      const id = buffer[offset + i];
      if (!paletteMap.has(id)) {
        paletteMap.set(id, palette.length);
        palette.push(this._palette.get(id) || 'unknown');
      }
      blocks[i] = paletteMap.get(id);
    }
    offset += 4096;
    offset += 4096; // Skip nibble data

    return { subChunk: { blocks, palette }, offset };
  }

  _parseBlockStorage(buffer, offset) {
    if (offset >= buffer.length) return null;

    const header = buffer[offset++];
    const bitsPerBlock = header >> 1;
    const paletteType = header & 1; // 0 = persistence (NBT), 1 = runtime

    // Single-block sub-chunk (all same block)
    if (bitsPerBlock === 0) {
      if (paletteType !== 1) return null; // Can't parse NBT palette here
      const vr = this._readSignedVarint(buffer, offset);
      if (!vr) return null;
      const name = this._palette.get(vr.value) || 'air';
      const blocks = new Uint16Array(4096); // all 0
      return { storage: { blocks, palette: [name] }, offset: vr.offset };
    }

    // Read padded bit array
    const blocksPerWord = Math.floor(32 / bitsPerBlock);
    const wordCount = Math.ceil(4096 / blocksPerWord);
    const byteCount = wordCount * 4;
    if (offset + byteCount > buffer.length) return null;

    const blocks = new Uint16Array(4096);
    const mask = (1 << bitsPerBlock) - 1;
    let blockIdx = 0;

    for (let w = 0; w < wordCount && blockIdx < 4096; w++) {
      const word = buffer.readUInt32LE(offset + w * 4);
      for (let b = 0; b < blocksPerWord && blockIdx < 4096; b++) {
        blocks[blockIdx++] = (word >>> (b * bitsPerBlock)) & mask;
      }
    }
    offset += byteCount;

    // Read palette
    const pr = this._readUnsignedVarint(buffer, offset);
    if (!pr) return null;
    const paletteSize = pr.value;
    offset = pr.offset;

    const palette = [];
    if (paletteType === 1) {
      // Runtime IDs — map through start_game block palette
      for (let i = 0; i < paletteSize; i++) {
        const vr = this._readSignedVarint(buffer, offset);
        if (!vr) return null;
        palette.push(this._palette.get(vr.value) || 'unknown');
        offset = vr.offset;
      }
    } else {
      // Persistence (NBT) — can't parse easily, bail
      return null;
    }

    return { storage: { blocks, palette }, offset };
  }

  _skipBlockStorage(buffer, offset) {
    if (offset >= buffer.length) return null;

    const header = buffer[offset++];
    const bitsPerBlock = header >> 1;
    const paletteType = header & 1;

    if (bitsPerBlock === 0) {
      if (paletteType !== 1) return null;
      const vr = this._readSignedVarint(buffer, offset);
      return vr ? vr.offset : null;
    }

    const blocksPerWord = Math.floor(32 / bitsPerBlock);
    const wordCount = Math.ceil(4096 / blocksPerWord);
    offset += wordCount * 4;
    if (offset >= buffer.length) return null;

    const pr = this._readUnsignedVarint(buffer, offset);
    if (!pr) return null;
    const paletteSize = pr.value;
    offset = pr.offset;

    if (paletteType === 1) {
      for (let i = 0; i < paletteSize; i++) {
        const vr = this._readSignedVarint(buffer, offset);
        if (!vr) return null;
        offset = vr.offset;
      }
    } else {
      return null; // Can't skip NBT easily
    }

    return offset;
  }

  _readUnsignedVarint(buffer, offset) {
    let value = 0;
    let shift = 0;
    let byte;
    do {
      if (offset >= buffer.length) return null;
      byte = buffer[offset++];
      value |= (byte & 0x7F) << shift;
      shift += 7;
      if (shift > 35) return null; // Overflow protection
    } while (byte & 0x80);
    return { value: value >>> 0, offset };
  }

  _readSignedVarint(buffer, offset) {
    const result = this._readUnsignedVarint(buffer, offset);
    if (!result) return null;
    // ZigZag decode
    result.value = (result.value >>> 1) ^ -(result.value & 1);
    return result;
  }
}

module.exports = { BlockCache };
