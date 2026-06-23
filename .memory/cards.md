# llmbeans issue cards

## `-ngl` always 0 despite GPU offload in summary
**Issue:** Auto-detected hardware had `cuda=False` hardcoded in `from_detection()`, so llamacpp flag generator always emitted `-ngl 0`.
**Solution:** `lookup_specs_for_detection()` + profile DB match now sets `cuda=True` for NVIDIA GPUs; `-ngl` follows offload layers when `cuda` or `metal` is true.

## Tokens/sec ~0.0 on NVIDIA
**Issue:** `detect_hardware()` never populated `memory_bandwidth_gbps`; estimator returns 0 when bandwidth <= 0.
**Solution:** After detection, `lookup_specs_for_detection()` fills RAM/VRAM bandwidth from best-matching profile in `profiles.json`.

## Blank RAM type in summary (`RAM: 15 GB ()`)
**Issue:** `from_detection()` set `ram_type=""` for non-Apple platforms.
**Solution:** Profile match supplies `ram_type`; NVIDIA fallback `DDR5`, Apple M3/M4 fallback `LPDDR5X`; summary omits empty parens.

## `summary.txt` written when user declines
**Issue:** `cli.main()` always wrote `llmbeans-output/summary.txt` after the write prompt.
**Solution:** Summary only written inside `write_scripts()` when user confirms disk write.

## Custom path required full filename
**Issue:** Custom model path step was free-text only.
**Solution:** `_prompt_custom_path()` uses `_working_directory()` (`Path.cwd()`) — no hardcoded scan paths; menu shows actual CWD and relative model paths.

## 100% test coverage session
**Issue:** Test suite ~26% coverage; `fail_under=100` in pyproject.toml not met.
**Solution:** Added broad test modules (`test_estimator`, `test_detector`, `test_profiles`, `test_scanner`, `test_engine`, `test_tools`, `test_cli_helpers`, `test_coverage_gaps`, etc.) plus targeted edge-case tests in `tests/test_coverage_gaps.py`. 170 tests, 100% line coverage on `llmbeans/`.
**Note:** `IgnoreSourceDir` dict helper used to exercise CLI branches where `source_dir` must stay empty; defensive platform branches marked `# pragma: no cover` where not testable on macOS CI host.

## Review warnings batch (2026-06-22)
**Issue:** Review listed import inconsistency, triplicated script_gen, dead Linux SSD path, uncached profiles JSON, stale root demo scripts, MLX repo guess, M1/M2 RAM mis-ID.
**Solution:** Registry imports unified; canonical summary/scripts in `llmbeans/cli.py` with re-export; `scan` not alias; `_profiles_json()` lru_cache; subprocess mount resolution for Linux SSD; named estimator constants; tool warnings for MLX guess; `_apple_ram_type_fallback()`; root demo scripts updated; stale root `cli.py` removed. 178 tests, 100% coverage.
