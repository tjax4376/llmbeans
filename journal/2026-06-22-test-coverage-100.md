# Journal: 100% test coverage

**Version:** 0.1.1-dev  
**Author:** Cursor Agent  
**Timestamp:** 2026-06-22  
**Change rationale:** Raise test suite from ~26% to 100% line coverage per `fail_under=100` in pyproject.toml.

## Context

User requested more tests and full coverage. Prior session had reached ~98% with 157 tests; 24 lines remained across `cli.py`, `detector.py`, `profiles.py`, and `scanner.py`.

## Discussion points

1. Coverage config: `pytest-cov` with `source = ["llmbeans"]`, `fail_under = 100`, exclude `pragma: no cover` and `if __name__ == .__main__.:`.
2. Remaining gaps closed via targeted tests in `tests/test_coverage_gaps.py` rather than production refactors.
3. CLI edge paths: empty `source_dir`, invalid menu choices, model path outside CWD (`ValueError` on `relative_to`), search-folder validation.
4. Profile scoring branches: GPU substring match (+12), RAM near-miss (+2), Apple M2 `LPDDR5` fallback.
5. Scanner edge paths: nested index-only dirs, missing local paths, GGUF default vocab + `general.file_type` quant, HF repo quant from config only.
6. `IgnoreSourceDir` test helper prevents CLI loop from overwriting `source_dir` when testing display-name fallbacks.

## Code changed

| File | Change |
|------|--------|
| `tests/test_coverage_gaps.py` | Expanded with 13+ targeted edge-case tests for final coverage gaps |
| `tests/conftest.py` | Shared fixtures (from prior session) |
| `tests/test_*.py` | Broad module tests (estimator, detector, profiles, scanner, engine, cli, tools) |
| `pyproject.toml` | `fail_under = 100`, dev deps for pytest-cov |
| `.memory/cards.md` | Coverage session card |

## Test plan

- [x] `python -m pytest tests/ --cov=llmbeans --cov-report=term-missing -q`
- [x] 170 passed, 100.00% line coverage on all `llmbeans/` modules
