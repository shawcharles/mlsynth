# Legacy Comparison Tests

This directory contains scripts used for numerical comparison of estimators in the current `mlsynth` library against a legacy version (`mlsynth-legacy`).

These tests are part of the "Sense Check" phase to ensure that refactoring and updates to the main `mlsynth` library maintain numerical consistency with previous, stable versions for key estimators.

## Overall Comparison Results Summary

This section provides a high-level summary of the comparison results for each tested estimator. More details can be found under the specific script descriptions in the "Scripts" section below.

-   **TSSC (Two-Step Synthetic Control)**:
    -   Status: **CONSISTENT** (User confirmed prior to these tests).
    -   ATTs: Match for all sub-methods (SIMPLEX, MSCa, MSCb, MSCc).
    -   Counterfactuals: Match for all sub-methods.
    -   Weights: Match for SIMPLEX and MSCb. For MSCa and MSCc, weights are not populated by the current estimator due to an internal mismatch, leading to a difference.

-   **FDID (Forward Difference-in-Differences)**:
    -   Status: **CONSISTENT ATT, INCONSISTENT COUNTERFACTUALS**
    -   ATTs: Match for FDID, DID, and AUGDID sub-methods.
    -   Counterfactuals: No Match for FDID, DID, or AUGDID.
    -   Weights (main FDID method): Match.

-   **FMA (Factor Model Approach)**:
    -   Status: **CONSISTENT ATT, INCONSISTENT COUNTERFACTUALS**
    -   ATT: Match.
    -   Counterfactuals: No Match.

-   **FSCM (Forward Selected Synthetic Control Method)**:
    -   Status: **CONSISTENT ATT, INCONSISTENT COUNTERFACTUALS & WEIGHTS**
    -   ATT: Consistent.
    -   Counterfactuals: Differ.
    -   Weights: Differ.

-   **GSC (Generalized Synthetic Control)**:
    -   Status: **LEGACY VERSION FAILED, CURRENT VERSION RUNS**
    -   Legacy Version: Failed due to an internal `NameError`. This was subsequently patched (by defining `rank=1` in the legacy `GSC.fit` method) to allow the legacy script to run for comparison purposes.
    -   Current Version: Runs successfully. The initial issue where the current version's counterfactual was reported as `NoneType` by the comparison script has been resolved. This was due to the script looking for the attribute `synthetic_outcome` instead of `counterfactual_outcome` and issues with JSON serialization of NumPy arrays which are now fixed.
    -   Comparison: After fixes, the comparison script runs but reports differences in ATT and counterfactual values. These remaining numerical differences are separate from the `NoneType` issue and may reflect underlying algorithmic or default parameter changes between the legacy and current GSC implementations.

-   **NSC (Nonlinear Synthetic Control)**:
    -   Status: **CONSISTENT**
    -   ATT, Counterfactuals, and Weights match between legacy (with `k=5`) and current (default config).

-   **PDA (Panel Data Approach)**:
    -   Status: **INCONSISTENT ATT (~6.29% diff)**
    -   ATTs differ by approximately 6.29%, exceeding the 5% threshold.
    -   Counterfactuals also differ.

-   **Proximal (Proximal Inference)**:
    -   Status: **FAILED TO RUN (Both Versions)**
    -   Both versions failed with a "Singular matrix" error using `smoking_data.csv` and "Proposition 99" as a proxy.

-   **SCMO (Synthetic Control with Multiple Outcomes)**:
    -   Status: **CONSISTENT**
    -   ATT, Counterfactuals, and Weights match.

-   **SDID (Synthetic Difference-in-Differences)**:
    -   Status: **FAILED TO RUN (Both Versions)**
    -   Legacy: TypeError (int64 key). Current: MlsynthDataError (treated_indices type).

-   **SI (Synthetic Interventions)**:
    -   Status: **INCONCLUSIVE (Output Read Failure)**
    -   Legacy ATT was -0.0. Current output JSON could not be fully read.

-   **SRC (Synthetic Regressing Control)**:
    -   Status: **CONSISTENT**
    -   ATT, Counterfactuals, and Weights match.

*(This summary reflects the completion of the planned comparison tests as of May 26, 2025.)*

## Scripts

-   `comparison_test_tssc.py`: Compares the `TSSC` (Two-Step Synthetic Control) estimator.
    -   Uses `smoking_data.csv`.
    -   **Result Summary**: User confirmed consistent prior to these tests. Detailed results show ATT and counterfactuals match. Weights for SIMPLEX and MSCb match; MSCa/MSCc weights differ due to current estimator not populating them.

-   `comparison_test_fdid.py`: Compares the `FDID` (Forward Difference-in-Differences) estimator.
    -   Uses `basque_data.csv`.
    -   **Result Summary**: ATT matches for FDID, DID, and AUGDID methods. Donor weights for the main FDID method also match. However, counterfactual time series do not match for any of the three methods.

-   `comparison_test_fma.py`: Compares the `FMA` (Factor Model Approach) estimator.
    -   Uses `smoking_data.csv`.
    -   **Result Summary**: ATT matches. Counterfactual time series do not match.

-   `comparison_test_fscm.py`: Compares the `FSCM` (Forward Selected Synthetic Control Method) estimator.
    -   Uses `smoking_data.csv`.
    -   **Result Summary**: ATT is consistent. However, both the full counterfactual time series and the selected donor weights differ.

-   `comparison_test_gsc.py`: Compares the `GSC` (Generalized Synthetic Control) estimator.
    -   Uses `smoking_data.csv`.
    -   **Result Summary**: Legacy GSC initially failed due to an internal `NameError`. This was patched to allow the script to run. The current GSC version initially showed its counterfactual as `NoneType` in the script's output due to incorrect attribute access in the test script and JSON serialization issues with NumPy arrays; these have been fixed. After fixes, the comparison script runs but reports differences in ATT and counterfactual values. These remaining numerical differences are separate from the `NoneType` issue and may reflect underlying algorithmic or default parameter changes between the legacy and current GSC implementations.

-   `comparison_test_nsc.py`: Compares the `NSC` (Nonlinear Synthetic Control) estimator.
    -   Uses `smoking_data.csv`.
    -   Legacy NSC is run with `k=5`. Current NSC is run with its default configuration (hyperparameter search for `a` and `b`).
    -   **Result Summary**: ATT, Counterfactuals, and Weights are consistent between the legacy and current versions under these configurations.

-   `comparison_test_pda.py`: Compares the `PDA` (Panel Data Approach) estimator.
    -   Uses `smoking_data.csv`.
    -   **Result Summary**: ATT values differ by ~6.29% (Legacy: -15.6490, Current: -16.6340), exceeding the 5% threshold. Counterfactuals also differ.

-   `comparison_test_proximal.py`: Compares the `PROXIMAL` (Proximal Inference) estimator.
    -   Uses `smoking_data.csv`.
    -   **Result Summary**: Both legacy and current versions failed with a "Singular matrix" error when configured with "Proposition 99" as a proxy. Comparison not possible.

-   `comparison_test_scmo.py`: Compares the `SCMO` (Synthetic Control with Multiple Outcomes) estimator.
    -   Uses `smoking_data.csv`.
    -   **Result Summary**: ATT, Counterfactuals, and Weights are consistent between legacy and current versions.

-   `comparison_test_sdid.py`: Compares the `SDID` (Synthetic Difference-in-Differences) estimator.
    -   Uses `smoking_data.csv`.
    -   **Result Summary**: Legacy SDID failed (TypeError: int64 key). Current SDID failed (MlsynthDataError: treated_indices type). Comparison not possible.

-   `comparison_test_si.py`: Compares the `SI` (Synthetic Interventions) estimator.
    -   Uses `smoking_data.csv`, with "Proposition 99" as the alternative intervention.
    -   **Result Summary**: Inconclusive. Legacy SI produced an ATT of -0.0. The output JSON for the current SI version could not be fully read due to an environment issue, preventing a reliable comparison.

-   `comparison_test_src.py`: Compares the `SRC` (Synthetic Regressing Control) estimator.
    -   Uses `smoking_data.csv`.
    -   **Result Summary**: ATT, Counterfactuals, and Weights are consistent between legacy and current versions.

## Running Tests

To run a comparison test, navigate to the project root directory (`mlsynth-main`) and execute the script using Python, for example:

```bash
python -u dev/legacy_comparison_tests/comparison_test_fscm.py
```

Ensure that the `mlsynth-legacy` directory is present at the project root for the legacy part of the script to function correctly.

## Output

-   JSON files containing detailed outputs from both legacy and current estimator runs are saved in `dev/legacy_comparison_tests/comparison_outputs/`.
-   A summary comparing ATTs, counterfactuals, and weights is printed to standard output.
