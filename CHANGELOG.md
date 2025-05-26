# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-05-26

### Added
- Comprehensive test coverage for all utility modules in `mlsynth/utils/`. This includes:
    - `mlsynth/tests/test_denoiseutils.py` (enhanced)
    - `mlsynth/tests/test_helperutils.py` (enhanced)
    - `mlsynth/tests/test_inferutils.py` (enhanced)
    - `mlsynth/tests/test_resultutils.py` (initial tests)
    - `mlsynth/tests/test_selectorsutils.py` (enhanced)
    - `mlsynth/tests/test_sdidutils.py` (new test file and tests)
    - `mlsynth/tests/test_spillover.py` (tests for `mlsynth/utils/spillover.py`)
- Test coverage was also previously deepened for all 14 main estimator classes in `mlsynth/estimators/` during Phase 2.
- Note added to `dev/legacy_comparison_tests/README.md` clarifying the status of GSC comparison.

### Fixed
- **GSC Estimator Output**:
    - Resolved an issue where the `counterfactual_outcome` for the GSC estimator was incorrectly reported as `NoneType` by the legacy comparison test script (`dev/legacy_comparison_tests/comparison_test_gsc.py`).
    - Added `json_encoders` to `BaseEstimatorResults.Config` in `mlsynth/config_models.py` to ensure correct JSON serialization of NumPy arrays, including proper handling of `np.nan` values (converting them to `null`).
    - Corrected attribute access in `dev/legacy_comparison_tests/comparison_test_gsc.py` from `synthetic_outcome` to `counterfactual_outcome` to align with the Pydantic model definition.
- **`mlsynth/utils/spillover.py` & `mlsynth/tests/test_spillover.py`**:
    - Resolved mocking failures in `test_iterative_scm_smoke` and `test_iterative_scm_final_fit_fails`.
    - Modified `iterative_scm` in `mlsynth/utils/spillover.py` to use explicit SCM constructor calls (e.g., `CLUSTERSC(config)`) instead of `type(scm)(...)`.
    - Updated `isinstance` checks in `iterative_scm` to use original SCM types (e.g., `isinstance(scm, mlsynth.CLUSTERSC)`) by importing the top-level `mlsynth` package. This prevents `TypeError` when SCM classes are mocked in tests.
    - Corrected patch targets and strategy in `mlsynth/tests/test_spillover.py` to align with the changes in `iterative_scm`, ensuring mocks are applied to the SCM constructors as looked up within the `spillover` module.
    - All 19 tests in `mlsynth/tests/test_spillover.py` are now passing.
- Indentation concern in `mlsynth/utils/estutils.py` (LASSO block in `pda` function) addressed; Python interpreter parses correctly.
- Various minor fixes in utility functions and test fixtures to support expanded test coverage (details in Memory Bank `progress.md` and `activeContext.md`).

### Changed
- Refactored SCM re-instantiation logic in `mlsynth/utils/spillover.py` for improved testability and correctness when mocking.
- Updated project version to `0.2.0` in `setup.py`.

### Project Status
- Completed Phase 3 sub-task: "Utility Modules Test Coverage Review & Enhancement".
- Resolved GSC estimator output discrepancy in legacy comparison tests.
- Next major steps involve "API Standardization Review" and "Comprehensive Docstring Content Review".

## [Unreleased]
