# SCAR-Minecraft

AI agent controlling a real Minecraft client (Bedrock/Java) via transparent MITM relay proxy. Python FPI agent ↔ TCP bridge ↔ Node.js relay ↔ real client ↔ server/Realm.

## Commands

```bash
source .venv/bin/activate
python -m pytest tests/ -x -q              # run tests (1 known failure in test_live_encoder.py)
node controller/main.js --realm-invite URL  # Bedrock Realm
node controller/main.js --world "MyWorld"   # Java (auto-starts server from save)
npm run gui                                 # Electron desktop app
npm run build                               # build .app bundle
```

## Gotchas

- `@nut-tree/nut-js` was removed from npm — use `@nut-tree-fork/nut-js` (already installed)
- Bedrock Realms kick at ~12-15 CPS — bot attack timing must stay under 12
- Java Edition requires keyboard fallback for movement (client-side physics, can't inject position packets)
- Signal is 428-dim (396 base + 16 vision + 16 history) — changes to encoder slices cascade through env.py, fast_train.py, neural_policy.py, and tests
- Bridge protocol (bridge.js ↔ bridge.py) and state dict (state.js ↔ env.py) must stay in sync between JS and Python
