/**
 * JavaBlockCache — parses Java Edition chunk data to provide blockAt().
 *
 * Java 1.18+ uses paletted containers per 16x16x16 section.
 * Sections are packed inside the chunkData buffer of map_chunk packets.
 *
 * Block index within a section: y*256 + z*16 + x  (YZX order)
 * Compact longs: 1.16+ padded format (entries don't cross long boundaries)
 * VarInt: unsigned, not zigzag
 */

class JavaBlockCache {
  /**
   * @param {object} mcData — minecraft-data instance for the server version.
   *   Must have mcData.blocksByStateId mapping stateId → { name }.
   */
  constructor(mcData) {
    this._mcData = mcData

    // Chunk columns: "cx,cz" → Array(24) of { blocks, palette } | null
    // 24 sections cover y [-64, 320) in 1.18+ overworld
    this._chunks = new Map()

    // Individual block overrides from block_change / multi_block_change
    this._blockOverrides = new Map() // "x,y,z" → blockName

    // Prune chunks beyond this radius (in chunk coords)
    this._pruneRadius = 8
    this._lastPruneChunk = null
  }

  // ---- Public packet handlers ----

  /**
   * Handle map_chunk packet — full chunk column.
   * @param {object} data — { x, z, groundUp, bitMap, chunkData, blockEntities }
   */
  handleMapChunk(data) {
    try {
      const cx = data.x
      const cz = data.z
      const buf = data.chunkData
      if (!Buffer.isBuffer(buf) || buf.length === 0) return

      const column = this._parseSections(buf, 24)
      this._chunks.set(`${cx},${cz}`, column)
    } catch (e) {
      // Graceful fallback — chunk parsing is best-effort
    }
  }

  /**
   * Handle block_change packet — single block override.
   * @param {object} data — { location: { x, y, z }, type: stateId }
   */
  handleBlockChange(data) {
    if (!data.location) return
    const { x, y, z } = data.location
    const name = this._stateIdToName(data.type)
    this._blockOverrides.set(`${x},${y},${z}`, name)
  }

  /**
   * Handle multi_block_change packet — batch block overrides.
   * @param {object} data — { chunkCoordinates: { x, y, z }, records: [{ blockId, chunkBlockCoordinate }] }
   *   chunkCoordinates.x/z are chunk coords, y is section index.
   *   chunkBlockCoordinate encodes relative position as (x<<8 | z<<4 | y).
   */
  handleMultiBlockChange(data) {
    if (!data.chunkCoordinates || !data.records) return
    const { x: cx, y: sectionY, z: cz } = data.chunkCoordinates

    for (const record of data.records) {
      // chunkBlockCoordinate packs relative x, z, y within the section
      const packed = record.chunkBlockCoordinate
      const relX = (packed >> 8) & 0xF
      const relZ = (packed >> 4) & 0xF
      const relY = packed & 0xF

      const worldX = cx * 16 + relX
      const worldY = sectionY * 16 + relY
      const worldZ = cz * 16 + relZ

      const name = this._stateIdToName(record.blockId)
      this._blockOverrides.set(`${worldX},${worldY},${worldZ}`, name)
    }
  }

  /**
   * Look up block at world position. Returns { name: "stone" } or null.
   */
  blockAt(pos) {
    if (!pos) return null
    const x = Math.floor(pos.x)
    const y = Math.floor(pos.y)
    const z = Math.floor(pos.z)

    // Check overrides first (from block_change / multi_block_change)
    const override = this._blockOverrides.get(`${x},${y},${z}`)
    if (override !== undefined) {
      return { name: override }
    }

    // Look up in chunk column
    const cx = x >> 4
    const cz = z >> 4
    const column = this._chunks.get(`${cx},${cz}`)
    if (!column) return null

    // Section index: y = -64 → idx 0, y = 0 → idx 4
    const sectionIdx = (y + 64) >> 4
    if (sectionIdx < 0 || sectionIdx >= 24) return null
    const section = column[sectionIdx]
    if (!section) return null

    // Block index within section (YZX order)
    const blockIdx = ((y & 0xF) << 8) | ((z & 0xF) << 4) | (x & 0xF)

    const paletteIdx = section.blocks[blockIdx]
    if (paletteIdx === undefined) return null

    const name = section.palette[paletteIdx]
    return name ? { name } : null
  }

  /**
   * Prune chunks far from player. Call periodically (e.g., on position update).
   */
  prune(playerPos) {
    if (!playerPos) return
    const pcx = Math.floor(playerPos.x) >> 4
    const pcz = Math.floor(playerPos.z) >> 4

    // Only prune on significant movement (2+ chunks)
    if (this._lastPruneChunk) {
      const dx = pcx - this._lastPruneChunk.x
      const dz = pcz - this._lastPruneChunk.z
      if (dx * dx + dz * dz < 4) return
    }
    this._lastPruneChunk = { x: pcx, z: pcz }

    const r = this._pruneRadius
    for (const key of this._chunks.keys()) {
      const [cx, cz] = key.split(',').map(Number)
      if (Math.abs(cx - pcx) > r || Math.abs(cz - pcz) > r) {
        this._chunks.delete(key)
      }
    }

    for (const key of this._blockOverrides.keys()) {
      const [bx, , bz] = key.split(',').map(Number)
      if (Math.abs((bx >> 4) - pcx) > r || Math.abs((bz >> 4) - pcz) > r) {
        this._blockOverrides.delete(key)
      }
    }
  }

  get stats() {
    return {
      chunksLoaded: this._chunks.size,
      blockOverrides: this._blockOverrides.size,
    }
  }

  // ---- Internal parsing ----

  /**
   * Parse all sections from a chunk data buffer.
   * @param {Buffer} buffer — raw chunkData from map_chunk
   * @param {number} sectionCount — number of sections (24 for 1.18+)
   * @returns {Array} column — Array(sectionCount) of { blocks, palette } | null
   */
  _parseSections(buffer, sectionCount) {
    const column = new Array(sectionCount).fill(null)
    let offset = 0

    for (let i = 0; i < sectionCount && offset < buffer.length; i++) {
      try {
        const result = this._parseSection(buffer, offset)
        if (result) {
          column[i] = { blocks: result.blocks, palette: result.palette }
          offset = result.newOffset
        } else {
          break
        }
      } catch (e) {
        // One bad section shouldn't break the whole chunk — skip remaining
        break
      }
    }

    return column
  }

  /**
   * Parse a single 16x16x16 section from the buffer.
   * Returns { blocks: Uint16Array(4096), palette: string[], newOffset } or null.
   */
  _parseSection(buffer, offset) {
    if (offset + 2 >= buffer.length) return null

    // Block count (Int16BE) — informational, skip
    offset += 2

    // ---- Block paletted container ----
    if (offset >= buffer.length) return null
    const bitsPerEntry = buffer[offset++]

    let blocks, palette

    if (bitsPerEntry === 0) {
      // Single-valued section — one VarInt = global stateId for all 4096 blocks
      const vr = _readVarInt(buffer, offset)
      if (!vr) return null
      offset = vr.offset

      const name = this._stateIdToName(vr.value)
      blocks = new Uint16Array(4096) // all 0 → palette index 0
      palette = [name]

      // Data array length (should be 0)
      const dl = _readVarInt(buffer, offset)
      if (!dl) return null
      offset = dl.offset
      // Skip dl.value longs (should be 0)
      offset += dl.value * 8

    } else if (bitsPerEntry <= 8) {
      // Indirect palette
      const plr = _readVarInt(buffer, offset)
      if (!plr) return null
      const paletteLength = plr.value
      offset = plr.offset

      palette = new Array(paletteLength)
      for (let i = 0; i < paletteLength; i++) {
        const vr = _readVarInt(buffer, offset)
        if (!vr) return null
        palette[i] = this._stateIdToName(vr.value)
        offset = vr.offset
      }

      // Data array
      const dlr = _readVarInt(buffer, offset)
      if (!dlr) return null
      const dataArrayLength = dlr.value
      offset = dlr.offset

      if (offset + dataArrayLength * 8 > buffer.length) return null
      const longs = new Array(dataArrayLength)
      for (let i = 0; i < dataArrayLength; i++) {
        longs[i] = buffer.readBigInt64BE(offset)
        offset += 8
      }

      blocks = this._unpackEntries(longs, bitsPerEntry, 4096)

    } else {
      // Direct mapping (bitsPerEntry > 8) — no palette, global stateIds
      const dlr = _readVarInt(buffer, offset)
      if (!dlr) return null
      const dataArrayLength = dlr.value
      offset = dlr.offset

      if (offset + dataArrayLength * 8 > buffer.length) return null
      const longs = new Array(dataArrayLength)
      for (let i = 0; i < dataArrayLength; i++) {
        longs[i] = buffer.readBigInt64BE(offset)
        offset += 8
      }

      // Unpack raw stateIds, then build a local palette
      const rawIds = this._unpackEntries(longs, bitsPerEntry, 4096)
      const paletteMap = new Map()
      palette = []
      blocks = new Uint16Array(4096)

      for (let i = 0; i < 4096; i++) {
        const stateId = rawIds[i]
        let idx = paletteMap.get(stateId)
        if (idx === undefined) {
          idx = palette.length
          paletteMap.set(stateId, idx)
          palette.push(this._stateIdToName(stateId))
        }
        blocks[i] = idx
      }
    }

    // ---- Biome paletted container (SKIP) ----
    offset = this._skipBiomeContainer(buffer, offset)
    if (offset === null) return null

    return { blocks, palette, newOffset: offset }
  }

  /**
   * Skip the biome paletted container (64 entries, 3-bit resolution).
   * Returns new offset or null on error.
   */
  _skipBiomeContainer(buffer, offset) {
    if (offset >= buffer.length) return null
    const bitsPerEntry = buffer[offset++]

    if (bitsPerEntry === 0) {
      // Single-valued — skip 1 VarInt (biome id) + data array length
      const vr = _readVarInt(buffer, offset)
      if (!vr) return null
      offset = vr.offset
      const dl = _readVarInt(buffer, offset)
      if (!dl) return null
      offset = dl.offset
      offset += dl.value * 8
    } else if (bitsPerEntry <= 3) {
      // Indirect palette
      const plr = _readVarInt(buffer, offset)
      if (!plr) return null
      offset = plr.offset
      for (let i = 0; i < plr.value; i++) {
        const vr = _readVarInt(buffer, offset)
        if (!vr) return null
        offset = vr.offset
      }
      const dl = _readVarInt(buffer, offset)
      if (!dl) return null
      offset = dl.offset
      offset += dl.value * 8
    } else {
      // Direct mapping
      const dl = _readVarInt(buffer, offset)
      if (!dl) return null
      offset = dl.offset
      offset += dl.value * 8
    }

    return offset
  }

  /**
   * Unpack entries from a padded compact long array (1.16+ format).
   * Entries don't cross long boundaries.
   *
   * @param {BigInt[]} longs — array of 64-bit signed longs (BigInt)
   * @param {number} bitsPerEntry — bits per entry
   * @param {number} count — total entries to unpack (4096 for blocks)
   * @returns {Uint16Array} — unpacked entry values
   */
  _unpackEntries(longs, bitsPerEntry, count) {
    const entries = new Uint16Array(count)
    const entriesPerLong = Math.floor(64 / bitsPerEntry)
    const mask = (1n << BigInt(bitsPerEntry)) - 1n

    let entryIdx = 0
    for (let longIdx = 0; longIdx < longs.length && entryIdx < count; longIdx++) {
      // Convert signed to unsigned 64-bit for bit extraction
      const longVal = BigInt.asUintN(64, longs[longIdx])

      for (let j = 0; j < entriesPerLong && entryIdx < count; j++) {
        const bitOffset = BigInt(j * bitsPerEntry)
        const value = (longVal >> bitOffset) & mask
        entries[entryIdx++] = Number(value)
      }
    }

    return entries
  }

  /**
   * Map a global block stateId to a block name string.
   */
  _stateIdToName(stateId) {
    if (!this._mcData || !this._mcData.blocksByStateId) return 'unknown'
    const block = this._mcData.blocksByStateId[stateId]
    if (!block) return 'unknown'
    return (block.name || 'unknown').replace('minecraft:', '')
  }
}

// ---- Standalone helpers (not on prototype, avoids `this` overhead) ----

/**
 * Read a VarInt (unsigned, not zigzag) from buffer at offset.
 * Returns { value, offset } or null on error.
 */
function _readVarInt(buffer, offset) {
  let value = 0
  let shift = 0
  let byte
  do {
    if (offset >= buffer.length) return null
    byte = buffer[offset++]
    value |= (byte & 0x7F) << shift
    shift += 7
    if (shift > 35) return null // Overflow protection
  } while (byte & 0x80)
  return { value: value >>> 0, offset }
}

module.exports = { JavaBlockCache }
