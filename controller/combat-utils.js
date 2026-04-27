/**
 * Shared combat utilities used by both Bedrock and Java combat modules.
 */

function getEntityAABB(entity) {
  const pos = entity.position;
  if (!pos) return null;
  const width = entity.width || 0.6;
  const height = entity.height || 1.8;
  const halfWidth = width / 2;
  return {
    minX: pos.x - halfWidth, minY: pos.y, minZ: pos.z - halfWidth,
    maxX: pos.x + halfWidth, maxY: pos.y + height, maxZ: pos.z + halfWidth,
  };
}

function getEyePosition(entity) {
  const pos = entity.position;
  if (!pos) return null;
  const eyeHeight = entity.type === 'player' ? 1.62 : (entity.height || 1.8) * 0.85;
  return { x: pos.x, y: pos.y + eyeHeight, z: pos.z };
}

function getAABBDistance(adapter, entity) {
  if (!entity || !entity.position) return Infinity;
  const botPos = adapter.position;
  const eyePos = { x: botPos.x, y: botPos.y + 1.62, z: botPos.z };
  const aabb = getEntityAABB(entity);
  if (!aabb) return Infinity;
  const cx = Math.max(aabb.minX, Math.min(eyePos.x, aabb.maxX));
  const cy = Math.max(aabb.minY, Math.min(eyePos.y, aabb.maxY));
  const cz = Math.max(aabb.minZ, Math.min(eyePos.z, aabb.maxZ));
  const dx = eyePos.x - cx, dy = eyePos.y - cy, dz = eyePos.z - cz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function isShielding(target) {
  if (!target) return false;
  const handState = target._handState || 0;
  // Bit 0x01 = hand_active (using item). Best available without equipment tracking.
  return (handState & 0x01) !== 0;
}

module.exports = { getEntityAABB, getEyePosition, getAABBDistance, isShielding };
