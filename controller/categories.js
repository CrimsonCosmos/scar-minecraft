/**
 * Block categories and entity classification constants.
 */

const BLOCK_CATEGORIES = {
  air: new Set(['air', 'cave_air', 'void_air']),
  stone: new Set([
    'stone', 'cobblestone', 'granite', 'diorite', 'andesite',
    'deepslate', 'cobbled_deepslate', 'sandstone', 'red_sandstone',
    'smooth_stone', 'mossy_cobblestone', 'stone_bricks',
  ]),
  dirt: new Set([
    'dirt', 'grass_block', 'podzol', 'mycelium', 'coarse_dirt',
    'rooted_dirt', 'mud', 'clay', 'gravel', 'sand', 'red_sand',
    'farmland', 'dirt_path',
  ]),
  wood: new Set([
    'oak_log', 'spruce_log', 'birch_log', 'jungle_log', 'acacia_log',
    'dark_oak_log', 'mangrove_log', 'cherry_log',
    'oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks',
    'acacia_planks', 'dark_oak_planks', 'mangrove_planks', 'cherry_planks',
    'oak_leaves', 'spruce_leaves', 'birch_leaves', 'jungle_leaves',
    'acacia_leaves', 'dark_oak_leaves', 'mangrove_leaves', 'cherry_leaves',
    'azalea_leaves', 'flowering_azalea_leaves',
  ]),
  water: new Set(['water', 'ice', 'packed_ice', 'blue_ice', 'frosted_ice']),
  ore: new Set([
    'coal_ore', 'iron_ore', 'gold_ore', 'diamond_ore', 'emerald_ore',
    'lapis_ore', 'redstone_ore', 'copper_ore',
    'deepslate_coal_ore', 'deepslate_iron_ore', 'deepslate_gold_ore',
    'deepslate_diamond_ore', 'deepslate_emerald_ore', 'deepslate_lapis_ore',
    'deepslate_redstone_ore', 'deepslate_copper_ore', 'nether_gold_ore',
    'nether_quartz_ore', 'ancient_debris',
  ]),
  danger: new Set([
    'lava', 'fire', 'soul_fire', 'magma_block', 'cactus',
    'sweet_berry_bush', 'wither_rose', 'pointed_dripstone',
  ]),
  // "other" is the default — anything not in the above categories
};

function categorizeBlock(name) {
  for (const [cat, names] of Object.entries(BLOCK_CATEGORIES)) {
    if (names.has(name)) return cat;
  }
  return 'other';
}

const HOSTILE_MOBS = new Set([
  'zombie', 'skeleton', 'creeper', 'spider', 'cave_spider', 'enderman',
  'witch', 'slime', 'phantom', 'drowned', 'husk', 'stray', 'pillager',
  'vindicator', 'evoker', 'ravager', 'vex', 'blaze', 'ghast',
  'magma_cube', 'wither_skeleton', 'piglin_brute', 'warden',
  'zombified_piglin', 'hoglin', 'zoglin', 'guardian', 'elder_guardian',
  'shulker', 'silverfish', 'endermite',
]);

const PASSIVE_MOBS = new Set([
  'cow', 'pig', 'sheep', 'chicken', 'horse', 'donkey', 'mule',
  'rabbit', 'mooshroom', 'goat', 'frog', 'axolotl', 'turtle',
  'bee', 'cat', 'wolf', 'fox', 'panda', 'parrot', 'dolphin',
  'squid', 'glow_squid', 'cod', 'salmon', 'tropical_fish', 'pufferfish',
  'villager', 'wandering_trader', 'sniffer', 'camel', 'allay',
  'strider', 'llama', 'trader_llama', 'bat', 'ocelot', 'iron_golem',
  'snow_golem',
]);

// Map hostile mob types to index (0-3) for the 4 type bases
const HOSTILE_TYPE_MAP = { zombie: 0, skeleton: 1, spider: 2, creeper: 3 };
// Map passive mob types to index (0-3)
const PASSIVE_TYPE_MAP = { cow: 0, pig: 1, sheep: 2, chicken: 3 };

function getEntityTypeIndex(name, typeMap) {
  if (name in typeMap) return typeMap[name];
  // Hash unknown types to 0-3
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return Math.abs(h) % 4;
}

module.exports = {
  BLOCK_CATEGORIES,
  HOSTILE_MOBS,
  PASSIVE_MOBS,
  HOSTILE_TYPE_MAP,
  PASSIVE_TYPE_MAP,
  categorizeBlock,
  getEntityTypeIndex,
};
