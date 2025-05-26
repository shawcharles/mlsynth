# mlsynth Library Roadmap

This document outlines the planned future development and improvement areas for the `mlsynth` library. It is intended to provide transparency for users and contributors regarding the project's direction.

## Near-Term Goals (Targeting v0.2.0 Release Focus)

The primary focus for the v0.2.0 release was to achieve a "good" level of robustness and test coverage, building upon the significant standardization and refactoring work already completed. Key tasks accomplished and remaining include:

*   **Completed - Utility Module Test Coverage:**
    *   Comprehensive test coverage for all utility modules in `mlsynth/utils/` was reviewed and enhanced.
*   **Completed - GSC Estimator Output Fix:**
    *   Resolved an issue where the GSC estimator's counterfactual outcome was incorrectly reported as `NoneType` by legacy comparison tests. This involved fixes to Pydantic model serialization and test script attribute access.
*   **Ongoing - Complete Error Handling Implementation (Item 1.6 from `improvement_strategy.md`):**
    *   Finalize the review and refactoring of error handling for all remaining estimator modules.
    *   Ensure comprehensive unit tests are added for all new error conditions and validation logic introduced.
*   **Ongoing - Resolve Known Test Issues:**
    -   Investigate and fix skipped tests (e.g., in `mlsynth/tests/test_proximal.py`).
*   **Ongoing - Baseline Operational Tests:**
    *   Ensure basic end-to-end "smoke tests" exist for all estimators, confirming they run with simple valid configurations.
*   **Ongoing - Critical Utility Function Test Review:**
    *   Conduct a final review of tests for the most critical utility functions to ensure main success and failure paths are covered.

## Mid-Term Goals (Potentially for v0.0.3 and Beyond)

These items represent more substantial enhancements and efforts towards achieving comprehensive test coverage and functionality.

### Testing Enhancements
*   **Exhaustive Edge Case Testing:** Systematically identify and test a wider range of edge cases for all functions and parameter combinations.
*   **Full-Scale Integration Tests:** Develop more extensive integration tests that cover complex interactions between modules and diverse data configurations for each estimator.
*   **Synthetic Data Generation Framework:** Develop and integrate a framework for generating synthetic datasets with known true effects to allow for more rigorous validation of estimator correctness.
*   **Advanced Assertion Strategies:** Develop custom helper functions or strategies for more detailed comparison of complex numerical outputs or large data structures in tests.
*   **Code Coverage Targets:** Actively work towards achieving and maintaining a high percentage of code coverage (e.g., >90%) as measured by tools like `coverage.py`.
*   **Performance and Benchmarking Tests:** Introduce tests to validate and track the computational performance of estimators under various conditions.
*   **Systematic Use of `basedata`:** Further integrate the existing datasets in `basedata/` into a broader and more systematic range of test scenarios for all relevant estimators.

### Modularity & Design Refinements
*   **Utility Module Structure:** Evaluate `mlsynth/utils/estutils.py` for potential further breakdown into more specialized modules (e.g., `optutils.py`).
*   **Class Refactoring:** Review and potentially refactor single-method utility classes (e.g., `Opt`, `effects` in `estutils.py`) if a more functional approach is clearer.
*   **Circular Dependency Resolution:** Address the circular import issue related to `dataprep` in `mlsynth/utils/helperutils.py`.
*   **Plotting Logic:** Consider standardizing plotting further, potentially by ensuring all estimators have a dedicated `plot_estimates()` method or by using centralized plotting functions that consume standardized `fit()` results.

### Documentation & API
*   **Complete Documentation:**
    *   Ensure every estimator has a comprehensive `.rst` page in the Sphinx documentation.
    *   Fully document all public utility functions and modules.
    *   Address any remaining "To Do List" items in the documentation.
*   **API Reference:** Continue to refine and ensure the completeness of the auto-generated API reference.

### Python Best Practices & Other
*   **Python 3.9+ Features:** Systematically review the codebase for opportunities to leverage more modern Python features where they improve clarity or performance.
*   **Dependency Management:** Consider adopting modern dependency management tools like Poetry or PDM for future project development.
*   **`screenot` Dependency:** Periodically review if `screenot` remains the optimal choice for `adaptiveHardThresholding` or if suitable alternatives emerge.

---
*This roadmap is a living document and will be updated as the project evolves. Contributions and suggestions are welcome via GitHub Issues.*
