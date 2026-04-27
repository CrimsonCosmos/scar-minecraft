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

// 3-way voxel classification for spatial grid encoding
const PASSABLE_BLOCKS = new Set([
  'air', 'cave_air', 'void_air',
  'grass', 'tall_grass', 'fern', 'large_fern', 'dead_bush',
  'dandelion', 'poppy', 'blue_orchid', 'allium', 'azure_bluet',
  'red_tulip', 'orange_tulip', 'white_tulip', 'pink_tulip',
  'oxeye_daisy', 'cornflower', 'lily_of_the_valley', 'sunflower',
  'lilac', 'rose_bush', 'peony', 'torchflower', 'pitcher_plant',
  'torch', 'wall_torch', 'soul_torch', 'soul_wall_torch',
  'redstone_torch', 'redstone_wall_torch',
  'snow', 'rail', 'powered_rail', 'detector_rail', 'activator_rail',
  'lever', 'stone_button', 'oak_button', 'spruce_button',
  'birch_button', 'jungle_button', 'acacia_button', 'dark_oak_button',
  'crimson_button', 'warped_button', 'polished_blackstone_button',
  'stone_pressure_plate', 'oak_pressure_plate', 'spruce_pressure_plate',
  'birch_pressure_plate', 'jungle_pressure_plate', 'acacia_pressure_plate',
  'dark_oak_pressure_plate', 'crimson_pressure_plate', 'warped_pressure_plate',
  'light_weighted_pressure_plate', 'heavy_weighted_pressure_plate',
  'tripwire', 'tripwire_hook', 'string',
  'sign', 'wall_sign', 'oak_sign', 'oak_wall_sign',
  'spruce_sign', 'spruce_wall_sign', 'birch_sign', 'birch_wall_sign',
  'carpet', 'white_carpet', 'orange_carpet', 'magenta_carpet',
  'light_blue_carpet', 'yellow_carpet', 'lime_carpet', 'pink_carpet',
  'gray_carpet', 'light_gray_carpet', 'cyan_carpet', 'purple_carpet',
  'blue_carpet', 'brown_carpet', 'green_carpet', 'red_carpet', 'black_carpet',
  'vine', 'ladder', 'cobweb',
]);

const DANGER_BLOCKS = new Set([
  'lava', 'fire', 'soul_fire', 'magma_block', 'cactus',
  'sweet_berry_bush', 'wither_rose', 'pointed_dripstone',
]);

function classifyVoxel(name) {
  if (!name) return 'air';
  if (PASSABLE_BLOCKS.has(name)) return 'air';
  if (DANGER_BLOCKS.has(name)) return 'danger';
  return 'solid';
}

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

// --- Item categorization for hotbar encoding ---

const ITEM_CATEGORIES = {
  EMPTY: 0,
  SWORD: 1,
  PICKAXE: 2,
  AXE: 3,
  SHOVEL: 4,
  RANGED: 5,
  FOOD: 6,
  BLOCK: 7,
  OTHER: 8,
};

const MATERIAL_TIERS = {
  wooden: 0.2, wood: 0.2,
  stone: 0.4,
  golden: 0.3, gold: 0.3,
  iron: 0.6,
  chainmail: 0.5,
  diamond: 0.8,
  netherite: 1.0,
};

// Max durability for tools/weapons by material
const MAX_DURABILITY = {
  wooden: 59, wood: 59,
  stone: 131,
  golden: 32, gold: 32,
  iron: 250,
  diamond: 1561,
  netherite: 2031,
  bow: 384,
  crossbow: 465,
  trident: 250,
};

const FOOD_NAMES = new Set([
  'apple', 'baked_potato', 'beetroot', 'beetroot_soup', 'bread',
  'carrot', 'chorus_fruit', 'cooked_beef', 'cooked_chicken',
  'cooked_cod', 'cooked_mutton', 'cooked_porkchop', 'cooked_rabbit',
  'cooked_salmon', 'cookie', 'dried_kelp', 'enchanted_golden_apple',
  'golden_apple', 'golden_carrot', 'honey_bottle', 'melon_slice',
  'mushroom_stew', 'poisonous_potato', 'potato', 'pufferfish',
  'pumpkin_pie', 'rabbit_stew', 'raw_beef', 'raw_chicken', 'raw_cod',
  'raw_mutton', 'raw_porkchop', 'raw_rabbit', 'raw_salmon', 'rotten_flesh',
  'spider_eye', 'steak', 'suspicious_stew', 'sweet_berries',
  'glow_berries', 'tropical_fish', 'beef', 'porkchop', 'mutton',
  'chicken', 'rabbit', 'cod', 'salmon',
]);

const BLOCK_ITEM_NAMES = new Set([
  'oak_log', 'spruce_log', 'birch_log', 'jungle_log', 'acacia_log',
  'dark_oak_log', 'mangrove_log', 'cherry_log',
  'oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks',
  'acacia_planks', 'dark_oak_planks', 'mangrove_planks', 'cherry_planks',
  'cobblestone', 'stone', 'dirt', 'sand', 'gravel', 'glass',
  'stick', 'torch', 'crafting_table', 'furnace', 'chest',
]);

/**
 * Categorize an item by name.
 * Returns { category, tier, maxDurability, maxStack }.
 */
function categorizeItem(name) {
  if (!name) return { category: ITEM_CATEGORIES.EMPTY, tier: 0, maxDurability: 0, maxStack: 64 };

  const n = name.toLowerCase();

  // Extract material prefix (e.g., "diamond" from "diamond_sword")
  const parts = n.split('_');
  const material = parts[0];
  const tier = MATERIAL_TIERS[material] || 0;

  if (n.endsWith('_sword') || n === 'sword') {
    return { category: ITEM_CATEGORIES.SWORD, tier, maxDurability: MAX_DURABILITY[material] || 60, maxStack: 1 };
  }
  if (n.endsWith('_pickaxe') || n === 'pickaxe') {
    return { category: ITEM_CATEGORIES.PICKAXE, tier, maxDurability: MAX_DURABILITY[material] || 60, maxStack: 1 };
  }
  if (n.endsWith('_axe') && !n.endsWith('_pickaxe') || n === 'axe') {
    return { category: ITEM_CATEGORIES.AXE, tier, maxDurability: MAX_DURABILITY[material] || 60, maxStack: 1 };
  }
  if (n.endsWith('_shovel') || n === 'shovel') {
    return { category: ITEM_CATEGORIES.SHOVEL, tier, maxDurability: MAX_DURABILITY[material] || 60, maxStack: 1 };
  }
  if (n === 'bow') {
    return { category: ITEM_CATEGORIES.RANGED, tier: 0.5, maxDurability: 384, maxStack: 1 };
  }
  if (n === 'crossbow') {
    return { category: ITEM_CATEGORIES.RANGED, tier: 0.6, maxDurability: 465, maxStack: 1 };
  }
  if (n === 'trident') {
    return { category: ITEM_CATEGORIES.RANGED, tier: 0.7, maxDurability: 250, maxStack: 1 };
  }
  if (FOOD_NAMES.has(n) || n.includes('cooked_') || n.includes('raw_')) {
    return { category: ITEM_CATEGORIES.FOOD, tier: 0, maxDurability: 0, maxStack: 64 };
  }
  if (BLOCK_ITEM_NAMES.has(n) || n.endsWith('_log') || n.endsWith('_planks') ||
      n.endsWith('_block') || n.endsWith('_slab') || n.endsWith('_stairs')) {
    return { category: ITEM_CATEGORIES.BLOCK, tier: 0, maxDurability: 0, maxStack: 64 };
  }

  return { category: ITEM_CATEGORIES.OTHER, tier: 0, maxDurability: 0, maxStack: 64 };
}

// --- Entity metadata key indices ---

// Java entity_metadata indices (1.20+ / 26.1+)
const ENTITY_META = {
  FLAGS: 0,            // byte: on_fire=0x01, crouching=0x02, sprinting=0x08
  HAND_STATE: 8,       // byte: hand_active=0x01, offhand=0x02
  HEALTH: 9,           // float: living entity health
  ZOMBIE_BABY: 14,     // boolean: is baby zombie
  CREEPER_STATE: 17,   // int: -1=idle, 0-30=fuse ticks
  CREEPER_CHARGED: 18, // boolean
};

// Bedrock set_entity_data keys
const BEDROCK_ENTITY_META = {
  FLAGS: 0,            // long bitfield: fire=0, sneak=1, sprint=3, using_item=4, baby=22
  HEALTH: 2,           // int
  FUSE_LENGTH: 56,     // int: creeper fuse ticks remaining
};

module.exports = {
  PASSABLE_BLOCKS,
  DANGER_BLOCKS,
  HOSTILE_MOBS,
  PASSIVE_MOBS,
  classifyVoxel,
  categorizeItem,
  FOOD_NAMES,
  ENTITY_META,
  BEDROCK_ENTITY_META,
};
