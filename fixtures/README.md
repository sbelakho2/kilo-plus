# Fixtures

- `protocol/` — golden wire fixtures mirrored into `compat/kilo-v756/`
- `providers/` — provider behavior fixtures (capability probes, wire shapes);
  locked by `tests/integration/tests/provider_fixtures.rs`; see
  `providers/README.md` for the consuming adapter tests
- `screenshots/` — frozen client screenshots for the visual compatibility
  suite (VS Code light/dark, JetBrains light/dark; zero-pixel-diff outside
  branding masks), plus `manifest.json` locking each case's mask and
  OS font-rendering tolerance zone; verified by
  `tests/visual/tests/pixel_regression.rs`
- `repositories/` — synthetic fixture repositories for indexing/search tests
