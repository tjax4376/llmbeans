# Journal: reported llmbeans bug fixes

**Version:** 0.1.1-dev  
**Author:** Cursor Agent  
**Timestamp:** 2026-06-22  
**Change rationale:** Fix five reported CLI/hardware bugs affecting NVIDIA users and disk-write behavior.

## Context

User reported five llmbeans issues: incorrect `-ngl` in generated commands, zero tokens/sec estimates on NVIDIA, blank RAM type in summaries, unconditional `summary.txt` writes, and poor custom model path UX.

## Discussion points

1. Root cause for `-ngl 0` was `from_detection()` hardcoding `cuda=False`, bypassing llamacpp GPU offload flag logic.
2. Zero tok/s traced to missing bandwidth on auto-detected hardware; profile DB already had NVIDIA bandwidth values.
3. Summary disk write happened outside the user confirmation branch.
4. Custom path UX improved with CWD model scan menu (same pattern as folder search).

## Code changed

| File | Change |
|------|--------|
| `llmbeans/hardware/profiles.py` | Added profile matching + `lookup_specs_for_detection()`; fixed `from_detection()` cuda/ram_type/bandwidth |
| `llmbeans/hardware/detector.py` | Populate bandwidth from profile lookup; fix Apple Silicon detection flag |
| `llmbeans/cli.py` | CWD model menu in custom path; summary only on confirmed write; RAM line formatting |
| `tests/test_hardware_profiles.py` | New tests for NVIDIA cuda/bandwidth/ngl |
| `tests/test_cli.py` | Updated write flow tests incl. decline path |

## Test plan

- [x] `pytest tests/` — 7 passed
- [x] NVIDIA auto-detect sets `cuda=True`, bandwidth > 0, `-ngl` matches offload
- [x] Declining disk write skips `write_scripts()` and no summary file
