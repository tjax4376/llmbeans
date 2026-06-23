# Review warnings + suggestions fixes

**Version:** 0.1.0  
**Author:** Cursor Agent  
**Timestamp:** 2026-06-22  
**Change rationale:** Address code review warnings/suggestions (import consistency, dedup, SSD detection, caching, stale scripts).

## Context

User supplied 10-item review list covering import paths, duplicate code, dead code, Linux SSD detection, profile caching, magic constants, stale demo scripts, MLX repo guessing, and Apple RAM type gaps.

## Discussion points

1. **register_tool** — vllm/mlx/lmstudio now import from `registry` directly (same as llamacpp/ollama).
2. **script_gen triplication** — `generate_summary` / `write_scripts` canonical in `llmbeans/cli.py`; `output/script_gen.py` re-exports; deleted `recommenders/script_gen.py`.
3. **scan_model alias** — removed; CLI uses `scan` directly.
4. **load_profiles caching** — `_profiles_json()` cached with `lru_cache`; fresh `HardwareProfileEntry` objects per call to avoid test mutation bleed.
5. **detector Linux SSD** — `findmnt` + `mountpoint -d` via subprocess; sysfs rotational check on resolved block device.
6. **estimator** — `RUNTIME_OVERHEAD_GB`, `INFERENCE_BANDWIDTH_EFFICIENCY` named constants.
7. **MLX** — warning when guessing `mlx-community/{name}`; engine merges tool `warnings`.
8. **Apple RAM** — `_apple_ram_type_fallback()` covers M1 (LPDDR4X), M2 (LPDDR5), M3/M4 (LPDDR5X).
9. **Stale root scripts** — removed duplicate root `cli.py`; updated `test_cli_imports.py`, `test_cli_integration.py`, `run_cli_demo.py`.

## Code changed

| Area | Files |
|------|-------|
| Tool imports | `recommenders/tools/vllm.py`, `mlx.py`, `lmstudio.py` |
| Engine | `recommenders/engine.py` (tool warnings merge) |
| Profiles | `hardware/profiles.py` (JSON cache, Apple RAM fallback) |
| Detector | `hardware/detector.py` (Linux block device resolution) |
| Estimator | `hardware/estimator.py` (named constants) |
| Scanner | `models/scanner.py` (removed `scan_model` alias) |
| CLI | `cli.py` (`scan` import) |
| Output | `output/script_gen.py` (re-exports + shell/batch helpers) |
| Removed | `recommenders/script_gen.py`, root `cli.py` |
| Tests | detector, profiles, tools, output/recommenders script_gen, scanner, cli patches |
| Demos | `test_cli_imports.py`, `test_cli_integration.py`, `run_cli_demo.py` |

**Test plan:** `pytest tests/ --cov=llmbeans` — 178 tests, 100% line coverage.
