/**
 * Shared utility functions.
 */

function parseArgs(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : 'true';
      result[key] = val;
      if (val !== 'true') i++;
    }
  }
  return result;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function waitTicks(tickRate, ticks) {
  const msPerTick = tickRate > 0 ? 1000 / tickRate : 50;
  return sleep(Math.max(ticks * msPerTick, 20));
}

module.exports = { parseArgs, sleep, waitTicks };
