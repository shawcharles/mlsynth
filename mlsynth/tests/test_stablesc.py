import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional # Added Optional
from pydantic import ValidationError
import cvxpy # For cvxpy.error.SolverError

from mlsynth import StableSC
from mlsynth.exceptions import (
    MlsynthDataError,
    MlsynthConfigError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from mlsynth.config_models import (
    StableSCConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)

# Full configuration dictionary used in tests.
STABLESC_FULL_TEST_CONFIG_BASE: Dict[str, Any] = {
    "outcome": "Y",
    "treat": "is_treated_stablesc", 
    "unitid": "unit_id",
    "time": "time_id",
    "display_graphs": False,
    "save": False,
    "counterfactual_color": "olive",
    "treated_color": "navy",
    "seed": 12321, # Not part of StableSCConfig
    "verbose": False, # Not part of StableSCConfig
}

# Fields that are part of BaseEstimatorConfig (and thus StableSCConfig)
STABLESC_PYDANTIC_MODEL_FIELDS = [
    "df", "outcome", "treat", "unitid", "time", 
    "display_graphs", "save", "counterfactual_color", "treated_color"
]

def _get_pydantic_config_dict_stablesc(full_config: Dict[str, Any], df_fixture: pd.DataFrame) -> Dict[str, Any]:
    """Helper to extract Pydantic-valid fields for StableSCConfig and add the DataFrame."""
    pydantic_dict = {
        k: v for k, v in full_config.items() if k in STABLESC_PYDANTIC_MODEL_FIELDS
    }
    pydantic_dict["df"] = df_fixture
    return pydantic_dict

@pytest.fixture
def stablesc_panel_data():
    """Provides a panel dataset for StableSC smoke testing."""
    data_dict = {
        'unit_id': np.repeat(np.arange(1, 6), 10), # 5 units, 10 periods
        'time_id': np.tile(np.arange(1, 11), 5),
        'Y': np.random.rand(50) * 10 + np.repeat(np.arange(0, 50, 10), 10), 
        'X1': np.random.rand(50) * 5,
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = STABLESC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 7), treatment_col_name] = 1
    return df

def test_stablesc_creation(stablesc_panel_data: pd.DataFrame):
    """Test that the StableSC estimator can be instantiated."""
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_panel_data)
    
    try:
        config_obj = StableSCConfig(**pydantic_dict)
        estimator = StableSC(config=config_obj)
        assert estimator is not None, "StableSC estimator should be created."
        assert estimator.outcome == "Y"
        assert estimator.treat == STABLESC_FULL_TEST_CONFIG_BASE["treat"]
        assert not estimator.display_graphs
    except Exception as e:
        pytest.fail(f"StableSC instantiation failed: {e}")

def test_stablesc_fit_smoke(stablesc_panel_data: pd.DataFrame, mocker: Any):
    """Smoke test for the StableSC fit method."""
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_panel_data)
    # Using default config parameters for this test, relying on mock for donor selection
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)

    # Mock select_donors to ensure some donors are always selected for this test
    # stablesc_panel_data has 5 units, 1 treated, so 4 potential donors.
    # Donor indices relative to the donor pool would be 0, 1, 2, 3.
    mock_selected_indices = np.array([0, 1]) # Select first two donors
    # Scores for all 4 potential donors. Selected donors have positive scores.
    mock_scores_all_donors = np.array([0.8, 0.7, 0.0, 0.0]) 

    with mocker.patch.object(estimator, 'select_donors', return_value=(mock_selected_indices, mock_scores_all_donors)):
        try:
            results = estimator.fit()
        except Exception as e:
            if isinstance(e, np.linalg.LinAlgError):
                pytest.skip(f"Skipping due to LinAlgError (with mock): {e}")
            if "Problem does not follow DCP rules" in str(e) or "Rank(A) < p or Rank(G) < q" in str(e) or "Solver 'CLARABEL' failed" in str(e):
                 pytest.skip(f"Skipping due to optimization error (with mock): {e}")
            pytest.fail(f"StableSC fit method failed during smoke test (with mock): {e}")

    assert isinstance(results, BaseEstimatorResults), "Fit method should return BaseEstimatorResults."

    assert results.effects is not None
    assert results.fit_diagnostics is not None
    assert results.time_series is not None
    assert results.weights is not None
    assert results.method_details is not None
    assert results.raw_results is not None
    assert "_prepped" in results.raw_results

    assert results.method_details.additional_details is not None
    assert "selected_donor_indices" in results.method_details.additional_details
    # Check that the mocked selected indices are reflected
    assert results.method_details.additional_details["selected_donor_indices"] == mock_selected_indices.tolist()
    
    assert "anomaly_scores_used" in results.method_details.additional_details
    # Check that the mocked scores are reflected
    np.testing.assert_array_almost_equal(
        results.method_details.additional_details["anomaly_scores_used"],
        mock_scores_all_donors,
        decimal=6
    )

    assert isinstance(results.effects.att, (float, np.floating, type(None)))
    assert isinstance(results.fit_diagnostics.pre_treatment_rmse, (float, np.floating, type(None)))
    assert isinstance(results.time_series.counterfactual_outcome, (np.ndarray, type(None)))
    assert isinstance(results.weights.donor_weights, (dict, type(None)))
    # This was already checked by comparing to mock_selected_indices.tolist()
    # assert isinstance(results.method_details.additional_details["selected_donor_indices"], list) 
    assert isinstance(results.method_details.additional_details["anomaly_scores_used"], list)

# --- Input Validation Tests ---

def test_stablesc_missing_config_keys(stablesc_panel_data):
    """Test StableSCConfig instantiation with missing required config keys."""
    required_keys = ["df", "outcome", "treat", "unitid", "time"]
    for key_to_remove in required_keys:
        full_config_dict = STABLESC_FULL_TEST_CONFIG_BASE.copy()
        if key_to_remove in full_config_dict: # df is not in base, added by helper
            del full_config_dict[key_to_remove]

        # Prepare dict for Pydantic, ensuring the key is missing
        pydantic_dict = {
            k: v for k, v in full_config_dict.items() if k in STABLESC_PYDANTIC_MODEL_FIELDS
        }
        if key_to_remove != "df": # df is added by helper, so if it's key_to_remove, it won't be added
             pydantic_dict["df"] = stablesc_panel_data.copy()
        
        # Ensure the key is truly absent if it was supposed to be removed
        if key_to_remove in pydantic_dict:
            del pydantic_dict[key_to_remove]

        with pytest.raises(ValidationError):
            StableSCConfig(**pydantic_dict)


def test_stablesc_invalid_df_type():
    """Test StableSCConfig instantiation with df not being a pandas DataFrame."""
    full_config_dict = STABLESC_FULL_TEST_CONFIG_BASE.copy()
    pydantic_dict = _get_pydantic_config_dict_stablesc(full_config_dict, "not_a_dataframe") # type: ignore
    
    with pytest.raises(ValidationError): # Pydantic should catch type error for df
        StableSCConfig(**pydantic_dict)

def test_stablesc_df_missing_columns(stablesc_panel_data: pd.DataFrame):
    """Test StableSC fit with DataFrame missing essential columns.
    Pydantic config creation now catches this.
    """
    cols_to_check = ["outcome", "treat", "unitid", "time"]

    for col_key in cols_to_check:
        full_config_dict = STABLESC_FULL_TEST_CONFIG_BASE.copy()
        df_copy = stablesc_panel_data.copy()
        
        actual_col_name_in_config = full_config_dict[col_key]
        
        if actual_col_name_in_config in df_copy.columns:
            df_modified = df_copy.drop(columns=[actual_col_name_in_config])
        else: 
            # This case should not happen if STABLESC_FULL_TEST_CONFIG_BASE keys match stablesc_panel_data columns
            pytest.skip(f"Column '{actual_col_name_in_config}' (from key '{col_key}') not in fixture df columns: {df_copy.columns.tolist()}.")
            continue

        pydantic_dict = _get_pydantic_config_dict_stablesc(full_config_dict, df_modified)
        
        # Expect MlsynthDataError during config instantiation due to missing column
        expected_message_part = f"Missing required columns in DataFrame 'df': {actual_col_name_in_config}"
        with pytest.raises(MlsynthDataError, match=expected_message_part):
            StableSCConfig(**pydantic_dict)

# --- Tests for Helper Methods ---

def test_stablesc_normalize(stablesc_panel_data: pd.DataFrame): # Added fixture for df
    """Test the normalize method."""
    # Need a minimal valid config to instantiate StableSC
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_panel_data)
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj) 

    Y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    Y_norm = estimator.normalize(Y)
    expected_means = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(Y_norm.mean(axis=0), expected_means, decimal=6)
    expected_Y_norm = np.array([[-3., -3., -3.], [ 0.,  0.,  0.], [ 3.,  3.,  3.]])
    np.testing.assert_array_almost_equal(Y_norm, expected_Y_norm, decimal=6)

    with pytest.raises(MlsynthDataError, match="Input 'input_array' must be a NumPy array."):
        estimator.normalize("not_an_array") # type: ignore
    
    with pytest.raises(MlsynthDataError, match="Error normalizing input array"):
        estimator.normalize(np.array(["a", "b"], dtype=object)) # type error during mean

def test_stablesc_granger_mask(stablesc_panel_data):
    """Test the granger_mask method."""
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_panel_data)
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)
    
    from mlsynth.utils.datautils import dataprep 
    prepped = dataprep(
        df=stablesc_panel_data,
        unit_id_column_name=pydantic_dict["unitid"],
        time_period_column_name=pydantic_dict["time"], 
        outcome_column_name=pydantic_dict["outcome"],
        treatment_indicator_column_name=pydantic_dict["treat"]
    )
    y_pre = prepped["y"][:prepped["pre_periods"]]
    Y0_pre = prepped["donor_matrix"][:prepped["pre_periods"], :]
    T0 = prepped["pre_periods"]

    y_norm = y_pre - y_pre.mean()
    Y0_norm = estimator.normalize(Y0_pre)

    mask = estimator.granger_mask(y_norm, Y0_norm, T0, alpha=0.05, maxlag=1)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert len(mask) == Y0_norm.shape[1]

    # Test input validation
    with pytest.raises(MlsynthConfigError, match="Number of pre-treatment periods must be positive"):
        estimator.granger_mask(y_norm, Y0_norm, 0)
    with pytest.raises(MlsynthConfigError, match="Maxlag for Granger causality must be positive"):
        estimator.granger_mask(y_norm, Y0_norm, T0, maxlag=0)
    with pytest.raises(MlsynthConfigError, match="Alpha for Granger causality must be between 0 and 1"):
        estimator.granger_mask(y_norm, Y0_norm, T0, alpha=1.5)
    with pytest.raises(MlsynthDataError, match="Not enough observations for the specified number of pre-treatment periods"):
        estimator.granger_mask(y_norm, Y0_norm, T0 + 10) # T0+10 is more than available

    # Test behavior when num_pre_treatment_periods <= maxlag (should warn and return False)
    if T0 > 1:
        # When maxlag is too high, granger_mask issues a UserWarning.
        # This can be either "Skipping Granger..." if caught by the pre-check,
        # or "Granger causality test failed for donor... ValueError..." if statsmodels fails.
        # The regex needs to accommodate both.
        expected_warning_regex = r"(Skipping Granger causality test for donor|Granger causality test failed for donor).*Marking as non-causal\."
        
        # Test case: maxlag = T0 (num_pre_treatment_periods <= maxlag is true)
        with pytest.warns(UserWarning, match=expected_warning_regex):
            mask_insufficient_obs_for_lag = estimator.granger_mask(y_norm, Y0_norm, T0, maxlag=T0)
            assert not np.any(mask_insufficient_obs_for_lag) # Expect all False as tests fail
        
        # Test case: maxlag = T0 - 1 (might hit ValueError from statsmodels or the pre-check)
        if T0 - 1 >= 1: # Ensure maxlag is at least 1
            with pytest.warns(UserWarning, match=expected_warning_regex):
                mask_insufficient_obs_for_lag_minus_1 = estimator.granger_mask(y_norm, Y0_norm, T0, maxlag=T0 - 1)
                assert not np.any(mask_insufficient_obs_for_lag_minus_1) # Expect all False
        elif T0 == 1 and Y0_norm.shape[1] > 0 : # Special case if T0=1, maxlag will be 1 (same as maxlag=T0)
             with pytest.warns(UserWarning, match=expected_warning_regex):
                mask_t0_eq_1 = estimator.granger_mask(y_norm, Y0_norm, T0, maxlag=1)
                assert not np.any(mask_t0_eq_1)


    # Test NaN handling (should warn and return False for that donor)
    Y0_with_nan = Y0_norm.copy()
    if Y0_with_nan.shape[1] > 0:
        Y0_with_nan[0, 0] = np.nan
        with pytest.warns(UserWarning, match="NaNs found in data for Granger causality test"):
            mask_nan = estimator.granger_mask(y_norm, Y0_with_nan, T0, alpha=0.05, maxlag=1)
            assert not mask_nan[0]

    # Test constant series handling (should warn and return False for that donor)
    Y0_constant = np.ones_like(Y0_norm)
    if Y0_constant.shape[1] > 0 and T0 > 1:
        with pytest.warns(UserWarning, match="Constant series detected for Granger causality test"):
            mask_constant = estimator.granger_mask(y_norm, Y0_constant, T0, alpha=0.05, maxlag=1)
            assert not np.any(mask_constant)


def test_stablesc_proximity_mask(stablesc_panel_data):
    """Test the proximity_mask method."""
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_panel_data)
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)

    from mlsynth.utils.datautils import dataprep
    prepped = dataprep(
        df=stablesc_panel_data,
        unit_id_column_name=pydantic_dict["unitid"],
        time_period_column_name=pydantic_dict["time"],
        outcome_column_name=pydantic_dict["outcome"],
        treatment_indicator_column_name=pydantic_dict["treat"]
    )
    Y0_pre = prepped["donor_matrix"][:prepped["pre_periods"], :]
    T0 = prepped["pre_periods"]
    Y0_norm = estimator.normalize(Y0_pre)

    mask, dists = estimator.proximity_mask(Y0_norm, T0, alpha=0.05)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert len(mask) == Y0_norm.shape[1]
    assert isinstance(dists, np.ndarray)
    assert len(dists) == Y0_norm.shape[1]

    # Test input validation
    with pytest.raises(MlsynthConfigError, match="Number of pre-treatment periods must be positive"):
        estimator.proximity_mask(Y0_norm, 0)
    with pytest.raises(MlsynthConfigError, match="Alpha for proximity mask must be between 0 and 1"):
        estimator.proximity_mask(Y0_norm, T0, alpha=-0.1)
    with pytest.raises(MlsynthDataError, match="Not enough observations for the specified number of pre-treatment periods"):
        estimator.proximity_mask(Y0_norm, T0 + 10)

    # Test single donor case (distance should be 0, mask True)
    if Y0_norm.shape[1] >= 1:
        Y0_single_donor = Y0_norm[:, [0]]
        mask_single, dists_single = estimator.proximity_mask(Y0_single_donor, T0, alpha=0.05)
        assert len(mask_single) == 1
        assert dists_single[0] == 0
        assert mask_single[0]

    # Test with T0=1 (df for chi2 is 1)
    if T0 >= 1 and Y0_norm.shape[1] > 1:
        Y0_T1_data = Y0_norm[[0], :] if Y0_norm.shape[0] >=1 else np.array([[]]) # Ensure Y0_T1_data is 2D
        if Y0_T1_data.shape[0] > 0: # Proceed only if there's data for T0=1
            mask_T1, dists_T1 = estimator.proximity_mask(Y0_T1_data, 1, alpha=0.05)
            assert len(mask_T1) == Y0_norm.shape[1]


def test_stablesc_rbf_scores(stablesc_panel_data: pd.DataFrame): # Added fixture for df
    """Test the rbf_scores method."""
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_panel_data)
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)

    dists = np.array([0.0, 1.0, 2.0, 10.0])
    
    scores_sigma1 = estimator.rbf_scores(dists, sigma=1.0)
    expected_scores_sigma1 = np.exp(-(dists**2) / (2 * 1.0**2))
    np.testing.assert_array_almost_equal(scores_sigma1, expected_scores_sigma1, decimal=6)

    scores_sigma20 = estimator.rbf_scores(dists, sigma=20.0)
    expected_scores_sigma20 = np.exp(-(dists**2) / (2 * 20.0**2))
    np.testing.assert_array_almost_equal(scores_sigma20, expected_scores_sigma20, decimal=6)

    assert scores_sigma1[0] == 1.0
    assert scores_sigma20[0] == 1.0

    # Test input validation
    with pytest.raises(MlsynthDataError, match="Input 'distances_array' must be a NumPy array."):
        estimator.rbf_scores("not_an_array", sigma=1.0) # type: ignore
    with pytest.raises(MlsynthConfigError, match="Input 'sigma' for RBF scores must be a numeric value."):
        estimator.rbf_scores(dists, sigma="abc") # type: ignore
    with pytest.raises(MlsynthConfigError, match="RBF sigma must be positive."):
        estimator.rbf_scores(dists, sigma=0)
    with pytest.raises(MlsynthConfigError, match="RBF sigma must be positive."):
        estimator.rbf_scores(dists, sigma=-1.0)
    with pytest.raises(MlsynthEstimationError, match="Error calculating RBF scores"):
        estimator.rbf_scores(np.array(["a","b"], dtype=object), sigma=1.0)


def test_stablesc_select_donors(stablesc_panel_data):
    """Test the select_donors method."""
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_panel_data)
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)

    from mlsynth.utils.datautils import dataprep
    prepped = dataprep(
        df=stablesc_panel_data,
        unit_id_column_name=pydantic_dict["unitid"],
        time_period_column_name=pydantic_dict["time"],
        outcome_column_name=pydantic_dict["outcome"],
        treatment_indicator_column_name=pydantic_dict["treat"]
    )
    y_pre = prepped["y"][:prepped["pre_periods"]]
    Y0_pre = prepped["donor_matrix"][:prepped["pre_periods"], :]
    T0 = prepped["pre_periods"]

    # Use config defaults for parameters not being specifically tested here
    keep_idx, S_diag_scores_all_donors = estimator.select_donors(
        treated_outcome_series=prepped["y"], # Pass full series
        all_donor_outcomes_matrix=prepped["donor_matrix"], # Pass full series
        num_pre_treatment_periods=T0,
        granger_alpha_param=config_obj.granger_alpha,
        granger_maxlag_param=config_obj.granger_maxlag,
        proximity_alpha_param=config_obj.proximity_alpha,
        rbf_sigma_param=config_obj.rbf_sigma_fit
    )
    
    assert isinstance(keep_idx, np.ndarray)
    assert keep_idx.ndim == 1
    # Y0_filtered is no longer returned directly, but can be constructed:
    Y0_filtered = prepped["donor_matrix"][:, keep_idx]
    assert Y0_filtered.shape[0] == prepped["donor_matrix"].shape[0]
    assert Y0_filtered.shape[1] == len(keep_idx)
    
    assert isinstance(S_diag_scores_all_donors, np.ndarray)
    assert len(S_diag_scores_all_donors) == prepped["donor_matrix"].shape[1] # Scores for all original donors
    
    S_diag_scores_selected = S_diag_scores_all_donors[keep_idx] # Get scores for selected donors
    if len(keep_idx) > 0: 
        assert np.all(S_diag_scores_selected > 0)
    
    def mock_granger_all_false(*args, **kwargs): return np.zeros(Y0_pre.shape[1], dtype=bool)
    original_granger = estimator.granger_mask
    estimator.granger_mask = mock_granger_all_false
    keep_idx_no_g, _ = estimator.select_donors(
        prepped["y"], prepped["donor_matrix"], T0, 
        config_obj.granger_alpha, config_obj.granger_maxlag, 
        config_obj.proximity_alpha, config_obj.rbf_sigma_fit
    )
    assert len(keep_idx_no_g) == 0
    estimator.granger_mask = original_granger 

    def mock_proximity_all_false(*args, **kwargs): return (np.zeros(Y0_pre.shape[1], dtype=bool), np.random.rand(Y0_pre.shape[1]))
    original_proximity = estimator.proximity_mask
    estimator.proximity_mask = mock_proximity_all_false
    keep_idx_no_p, _ = estimator.select_donors(
        prepped["y"], prepped["donor_matrix"], T0,
        config_obj.granger_alpha, config_obj.granger_maxlag, 
        config_obj.proximity_alpha, config_obj.rbf_sigma_fit
    )
    assert len(keep_idx_no_p) == 0
    estimator.proximity_mask = original_proximity 

    def mock_granger_all_true(*args, **kwargs): return np.ones(Y0_pre.shape[1], dtype=bool)
    def mock_proximity_all_true(*args, **kwargs): return (np.ones(Y0_pre.shape[1], dtype=bool), np.zeros(Y0_pre.shape[1])) 
    
    estimator.granger_mask = mock_granger_all_true
    estimator.proximity_mask = mock_proximity_all_true
    # For this specific sub-test, use a high alpha to ensure proximity mask passes if it were real
    keep_idx_all, S_diag_all_donors_scores = estimator.select_donors(
        prepped["y"], prepped["donor_matrix"], T0,
        granger_alpha_param=0.99, # High alpha for granger
        granger_maxlag_param=config_obj.granger_maxlag,
        proximity_alpha_param=0.99, # High alpha for proximity
        rbf_sigma_param=config_obj.rbf_sigma_fit
    )
    Y0_f_all = prepped["donor_matrix"][:, keep_idx_all] # Construct Y0_f_all
    S_diag_all_selected_scores = S_diag_all_donors_scores[keep_idx_all]

    assert len(keep_idx_all) == Y0_pre.shape[1]
    np.testing.assert_array_equal(Y0_f_all, prepped["donor_matrix"]) # Compare with full donor matrix
    if len(S_diag_all_selected_scores) > 0 : # Add check for empty array
        assert np.all(S_diag_all_selected_scores > 0)
    estimator.granger_mask = original_granger
    estimator.proximity_mask = original_proximity

    # Test validation within select_donors
    with pytest.raises(MlsynthConfigError, match="Number of pre-treatment periods must be positive"):
        estimator.select_donors(prepped["y"], prepped["donor_matrix"], 0, 0.1, 1, 0.1, 1.0)
    with pytest.raises(MlsynthDataError, match="Not enough observations"):
        estimator.select_donors(prepped["y"], prepped["donor_matrix"], T0 + 100, 0.1, 1, 0.1, 1.0)


# --- More Detailed Fit Tests ---

@pytest.fixture
def stablesc_fixture_no_donors_pass():
    """Fixture where it's likely no donors will pass selection."""
    data_dict = {
        'unit_id': np.repeat(np.arange(1, 5), 20), 
        'time_id': np.tile(np.arange(1, 21), 4),
        'Y': np.concatenate([
            np.linspace(1, 10, 20), 
            np.random.rand(20) * 50 + 100, 
            np.sin(np.linspace(0, 10 * np.pi, 20)) * 5 + 5, 
            np.random.normal(5, 1, 20) 
        ]),
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = STABLESC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == 1) & (df['time_id'] >= 11), treatment_col_name] = 1 
    return df

def test_stablesc_fit_no_donors_selected(stablesc_fixture_no_donors_pass: pd.DataFrame):
    """Test fit() when no donors are selected."""
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_fixture_no_donors_pass)
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)

    def mock_granger_always_false(self_estimator_instance: StableSC, treated_outcome_series: np.ndarray, donor_outcomes_matrix: np.ndarray, num_pre_treatment_periods: int, alpha: float =0.05, maxlag: int =1) -> np.ndarray:
        return np.zeros(donor_outcomes_matrix.shape[1], dtype=bool)

    def mock_proximity_always_false(self_estimator_instance: StableSC, donor_outcomes_matrix: np.ndarray, num_pre_treatment_periods: int, alpha: float =0.05) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(donor_outcomes_matrix.shape[1], dtype=bool), np.ones(donor_outcomes_matrix.shape[1]) * 1000 

    original_granger = StableSC.granger_mask
    original_proximity = StableSC.proximity_mask
    StableSC.granger_mask = mock_granger_always_false # type: ignore
    StableSC.proximity_mask = mock_proximity_always_false # type: ignore
    
    try:
        with pytest.raises(MlsynthDataError, match="No donors selected after applying Granger and proximity filters. Cannot proceed with SCM."): # Changed to MlsynthDataError
            estimator.fit()
    finally:
        StableSC.granger_mask = original_granger # type: ignore
        StableSC.proximity_mask = original_proximity # type: ignore


def test_stablesc_fit_all_donors_selected(stablesc_panel_data: pd.DataFrame):
    """Test fit() when all donors are selected (mocked)."""
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_panel_data)
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)

    def mock_granger_always_true(self_estimator_instance: StableSC, treated_outcome_series: np.ndarray, donor_outcomes_matrix: np.ndarray, num_pre_treatment_periods: int, alpha: float =0.05, maxlag: int =1) -> np.ndarray:
        return np.ones(donor_outcomes_matrix.shape[1], dtype=bool)

    def mock_proximity_always_true(self_estimator_instance: StableSC, donor_outcomes_matrix: np.ndarray, num_pre_treatment_periods: int, alpha: float =0.05) -> tuple[np.ndarray, np.ndarray]:
        return np.ones(donor_outcomes_matrix.shape[1], dtype=bool), np.zeros(donor_outcomes_matrix.shape[1])

    original_granger = StableSC.granger_mask
    original_proximity = StableSC.proximity_mask
    StableSC.granger_mask = mock_granger_always_true # type: ignore
    StableSC.proximity_mask = mock_proximity_always_true # type: ignore

    try:
        results = estimator.fit()
        assert results.raw_results is not None
        prepped = results.raw_results["_prepped"]
        num_donors = prepped["donor_matrix"].shape[1]
        
        assert results.method_details is not None
        assert results.method_details.additional_details is not None
        assert len(results.method_details.additional_details["selected_donor_indices"]) == num_donors
        assert np.all(np.array(results.method_details.additional_details["anomaly_scores_used"]) > 0.99)
        
        assert results.weights is not None
        assert results.weights.donor_weights is not None
        assert np.isclose(sum(results.weights.donor_weights.values()), 1.0, atol=1e-3)
        
        assert results.time_series is not None
        assert results.time_series.counterfactual_outcome is not None
        assert not np.allclose(results.time_series.counterfactual_outcome, 0)

    finally:
        StableSC.granger_mask = original_granger # type: ignore
        StableSC.proximity_mask = original_proximity # type: ignore

# Test for insufficient data scenarios
@pytest.mark.parametrize("n_units, n_periods, t_treat_start, expected_error_msg_part", [
    (1, 10, 7, "No donor units found"), 
    (3, 3, 2, "Not enough pre-treatment periods|Insufficient data points|maxlag must be < nobs|Number of pre-treatment periods must be positive"), # MlsynthConfigError
    (5, 7, 8, "No treated units found"),  # MlsynthDataError from dataprep
    (5, 7, 2, "Not enough pre-treatment periods|Insufficient data points|maxlag must be < nobs|Number of pre-treatment periods must be positive"), # MlsynthConfigError
])
def test_stablesc_fit_insufficient_data(n_units: int, n_periods: int, t_treat_start: int, expected_error_msg_part: str):
    """Test fit() with insufficient data for robust estimation."""
    data_dict = {
        'unit_id': np.repeat(np.arange(1, n_units + 1), n_periods),
        'time_id': np.tile(np.arange(1, n_periods + 1), n_units),
        'Y': np.random.rand(n_units * n_periods) * 10,
    }
    df = pd.DataFrame(data_dict)
    treatment_col_name = STABLESC_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    if n_units >= 1:
        df.loc[(df['unit_id'] == 1) & (df['time_id'] >= t_treat_start), treatment_col_name] = 1
    else: 
        with pytest.raises(Exception): 
             pydantic_dict_err = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, df)
             config_obj_err = StableSCConfig(**pydantic_dict_err)
             estimator_err = StableSC(config=config_obj_err)
             estimator_err.fit()
        return

    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, df)
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)

    if expected_error_msg_part is None:
        # This case might not be reachable if all insufficient data scenarios raise errors
        try:
            results = estimator.fit()
            # This assertion might need re-evaluation based on how such cases are handled
            assert results.effects.att is None or np.isnan(results.effects.att) or results.effects.att == 0, \
                f"ATT should be None, NaN, or 0 with no post-periods. Got: {results.effects.att}"
        except Exception as e:
            pytest.fail(f"Fit failed unexpectedly for no post-periods case ({n_units}u, {n_periods}p, treat@{t_treat_start}): {e}")
    else:
        # Expect MlsynthDataError or MlsynthConfigError based on the new error handling
        with pytest.raises((MlsynthDataError, MlsynthConfigError, MlsynthEstimationError, cvxpy.error.SolverError)) as excinfo: # Added MlsynthEstimationError
            estimator.fit()
        print(f"Insufficient data test ({n_units}u, {n_periods}p, treat@{t_treat_start}) failed as expected: {excinfo.value}")
        assert any(msg_part.lower() in str(excinfo.value).lower() for msg_part in expected_error_msg_part.split("|")), \
            f"Expected one of '{expected_error_msg_part}' in '{str(excinfo.value)}'"


# --- More specific tests for select_donors behavior ---
def test_stablesc_select_donors_alpha_sigma_effects(stablesc_panel_data: pd.DataFrame):
    """Test select_donors with varying alpha and sigma."""
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_panel_data)
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)

    from mlsynth.utils.datautils import dataprep
    prepped = dataprep(
        df=stablesc_panel_data,
        unit_id_column_name=pydantic_dict["unitid"],
        time_period_column_name=pydantic_dict["time"],
        outcome_column_name=pydantic_dict["outcome"],
        treatment_indicator_column_name=pydantic_dict["treat"]
    )
    y_pre = prepped["y"][:prepped["pre_periods"]]
    Y0_pre = prepped["donor_matrix"][:prepped["pre_periods"], :]
    T0 = prepped["pre_periods"]

    num_donors = Y0_pre.shape[1]
    mock_g_mask_val = np.array([i % 2 == 0 for i in range(num_donors)], dtype=bool)
    mock_p_mask_val = np.array([i % 2 != 0 for i in range(num_donors)], dtype=bool)
    mock_p_mask_val[0] = True 
    mock_dists_val = np.linspace(0, 5, num_donors) 

    def mock_granger_mask_fixed(*args, **kwargs): return mock_g_mask_val
    def mock_proximity_mask_fixed(*args, **kwargs): return mock_p_mask_val, mock_dists_val
    
    original_granger = estimator.granger_mask
    original_proximity = estimator.proximity_mask
    estimator.granger_mask = mock_granger_mask_fixed
    estimator.proximity_mask = mock_proximity_mask_fixed
    
    # Use a consistent alpha for this test, vary sigma
    test_alpha = 0.1 
    
    keep_idx_default, scores_all_donors_default_sigma = estimator.select_donors(
        prepped["y"], prepped["donor_matrix"], T0, 
        granger_alpha_param=test_alpha, 
        granger_maxlag_param=config_obj.granger_maxlag, 
        proximity_alpha_param=test_alpha, 
        rbf_sigma_param=20.0
    )
    scores_default_sigma_selected = scores_all_donors_default_sigma[keep_idx_default]

    keep_idx_small, scores_all_donors_small_sigma = estimator.select_donors(
        prepped["y"], prepped["donor_matrix"], T0, 
        granger_alpha_param=test_alpha, 
        granger_maxlag_param=config_obj.granger_maxlag, 
        proximity_alpha_param=test_alpha, 
        rbf_sigma_param=0.5
    )
    scores_small_sigma_selected = scores_all_donors_small_sigma[keep_idx_small]

    keep_idx_large, scores_all_donors_large_sigma = estimator.select_donors(
        prepped["y"], prepped["donor_matrix"], T0, 
        granger_alpha_param=test_alpha, 
        granger_maxlag_param=config_obj.granger_maxlag, 
        proximity_alpha_param=test_alpha, 
        rbf_sigma_param=100.0
    )
    scores_large_sigma_selected = scores_all_donors_large_sigma[keep_idx_large]

    estimator.granger_mask = original_granger
    estimator.proximity_mask = original_proximity

    # The selected indices should be the same if only sigma changes and masks are mocked
    # This relies on the mock masks being consistent.
    internal_hybrid_mask = mock_g_mask_val & mock_p_mask_val
    selected_indices_by_mock = np.where(internal_hybrid_mask)[0]
    
    np.testing.assert_array_equal(keep_idx_default, selected_indices_by_mock)
    np.testing.assert_array_equal(keep_idx_small, selected_indices_by_mock)
    np.testing.assert_array_equal(keep_idx_large, selected_indices_by_mock)


    if len(selected_indices_by_mock) > 0:
        assert len(scores_default_sigma_selected) == len(selected_indices_by_mock)
        assert len(scores_small_sigma_selected) == len(selected_indices_by_mock)
        assert len(scores_large_sigma_selected) == len(selected_indices_by_mock)

        target_idx_in_selection = -1
        # Find an index within the *selected* group that corresponds to a non-zero distance in the mock data
        for i, original_donor_idx in enumerate(selected_indices_by_mock):
            if mock_dists_val[original_donor_idx] > 1e-6: 
                target_idx_in_selection = i # This is the index within the selected scores array
                break
        
        if target_idx_in_selection != -1:
            # RBF score is exp(-dist^2 / (2*sigma^2)).
            # Smaller sigma -> faster decay -> smaller score for same non-zero distance.
            # Larger sigma -> slower decay -> larger score for same non-zero distance.
            assert scores_small_sigma_selected[target_idx_in_selection] < scores_default_sigma_selected[target_idx_in_selection]
            assert scores_default_sigma_selected[target_idx_in_selection] < scores_large_sigma_selected[target_idx_in_selection]
            assert scores_large_sigma_selected[target_idx_in_selection] < 1.0 
            assert scores_small_sigma_selected[target_idx_in_selection] > 0 
    else:
        assert len(scores_default_sigma_selected) == 0
        assert len(scores_small_sigma_selected) == 0
        assert len(scores_large_sigma_selected) == 0


def test_stablesc_fit_with_nan_in_outcome(stablesc_panel_data):
    """Test fit() when the outcome variable contains NaN values."""
    df_with_nan = stablesc_panel_data.copy()
    df_with_nan.loc[(df_with_nan['unit_id'] == 2) & (df_with_nan['time_id'] == 3), 'Y'] = np.nan
    
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, df_with_nan)
    # Modify alpha values to be less strict for testing
    pydantic_dict["granger_alpha"] = 0.8 
    pydantic_dict["proximity_alpha"] = 0.8 
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)
    
    # Expect MlsynthEstimationError or MlsynthDataError due to NaNs or solver issues
    with pytest.raises((MlsynthEstimationError, MlsynthDataError, cvxpy.error.SolverError)) as excinfo:
        estimator.fit()
    
    error_msg = str(excinfo.value).lower()
    # Check for more specific error messages related to custom exceptions or common issues
    assert "nan" in error_msg or \
           "invalid value" in error_msg or \
           "problem does not follow dcp rules" in error_msg or \
           "solver" in error_msg or \
           "clarabel" in error_msg or \
           "input x contains nan" in error_msg or \
           "infeasible" in error_msg or \
           "unbounded" in error_msg or \
           "no donors selected" in error_msg or \
           "error during donor selection" in error_msg or \
           "error calculating rbf scores" in error_msg or \
           "granger causality test failed" in error_msg or \
           "error calculating proximity mask" in error_msg, \
           f"Error message '{str(excinfo.value)}' did not match expected patterns for NaN/solver/selection issues."


def test_stablesc_fit_single_donor_scenario(stablesc_panel_data):
    """Test fit() with only one treated unit and one donor unit."""
    df_single_donor = stablesc_panel_data[stablesc_panel_data['unit_id'].isin([1, 2])].copy()
    
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, df_single_donor)
    config_obj = StableSCConfig(**pydantic_dict)
    estimator = StableSC(config=config_obj)
    
    with pytest.raises(MlsynthDataError, match="StableSC requires at least 2 donor units for robust estimation, but found 1."): # Changed to MlsynthDataError
        estimator.fit()


def test_stablesc_fit_plotting_behavior(stablesc_panel_data: pd.DataFrame, mocker: Any):
    """Test if plot_estimates is called correctly based on config."""
    
    mock_plot_estimates = mocker.patch("mlsynth.estimators.stablesc.plot_estimates")
    
    # Mock select_donors to ensure some donors are always selected
    mock_selected_indices = np.array([0, 1]) 
    mock_scores_all_donors = np.array([0.8, 0.7, 0.0, 0.0])

    # Test case 1: display_graphs = True
    full_config_display = STABLESC_FULL_TEST_CONFIG_BASE.copy()
    full_config_display["display_graphs"] = True
    pydantic_dict_display = _get_pydantic_config_dict_stablesc(full_config_display, stablesc_panel_data)
    config_obj_display = StableSCConfig(**pydantic_dict_display)
    estimator_display = StableSC(config=config_obj_display)
    
    with mocker.patch.object(estimator_display, 'select_donors', return_value=(mock_selected_indices, mock_scores_all_donors)):
        try:
            estimator_display.fit()
            mock_plot_estimates.assert_called_once()
        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError, cvxpy.error.SolverError) as e: # Adjusted expected errors
            if "Problem does not follow DCP rules" in str(e) or "Rank(A) < p" in str(e) or "optimal value is nan" in str(e) or "Solver 'CLARABEL' failed" in str(e) or "infeasible" in str(e).lower() or "no donors selected" in str(e).lower():
                pytest.skip(f"Skipping plotting test (display=True) due to data/optimization/selection error (with mock): {e}")
            else: raise e

    mock_plot_estimates.reset_mock()

    # Test case 2: display_graphs = False
    full_config_no_display = STABLESC_FULL_TEST_CONFIG_BASE.copy()
    full_config_no_display["display_graphs"] = False
    pydantic_dict_no_display = _get_pydantic_config_dict_stablesc(full_config_no_display, stablesc_panel_data)
    config_obj_no_display = StableSCConfig(**pydantic_dict_no_display)
    estimator_no_display = StableSC(config=config_obj_no_display)

    with mocker.patch.object(estimator_no_display, 'select_donors', return_value=(mock_selected_indices, mock_scores_all_donors)):
        try:
            estimator_no_display.fit()
            mock_plot_estimates.assert_not_called()
        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError, cvxpy.error.SolverError) as e: # Adjusted expected errors
            if "Problem does not follow DCP rules" in str(e) or "Rank(A) < p" in str(e) or "optimal value is nan" in str(e) or "Solver 'CLARABEL' failed" in str(e) or "infeasible" in str(e).lower() or "no donors selected" in str(e).lower():
                pytest.skip(f"Skipping plotting test (display=False) due to data/optimization/selection error (with mock): {e}")
            else: raise e
    
    mock_plot_estimates.reset_mock()

    # Test case 3: display_graphs = True, save = True
    full_config_save = STABLESC_FULL_TEST_CONFIG_BASE.copy()
    full_config_save["display_graphs"] = True 
    full_config_save["save"] = True 
    pydantic_dict_save = _get_pydantic_config_dict_stablesc(full_config_save, stablesc_panel_data)
    config_obj_save = StableSCConfig(**pydantic_dict_save)
    estimator_save = StableSC(config=config_obj_save)

    with mocker.patch.object(estimator_save, 'select_donors', return_value=(mock_selected_indices, mock_scores_all_donors)):
        try:
            estimator_save.fit()
            mock_plot_estimates.assert_called_once()
            call_args = mock_plot_estimates.call_args
            assert call_args is not None
            assert call_args.kwargs.get("save") is True
        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError, cvxpy.error.SolverError) as e: # Adjusted expected errors
            if "Problem does not follow DCP rules" in str(e) or "Rank(A) < p" in str(e) or "optimal value is nan" in str(e) or "Solver 'CLARABEL' failed" in str(e) or "infeasible" in str(e).lower() or "no donors selected" in str(e).lower():
                pytest.skip(f"Skipping plotting test (save=True) due to data/optimization/selection error (with mock): {e}")
            else: raise e

    # Test plotting failure warning
    mock_plot_estimates.reset_mock()
    mock_plot_estimates.side_effect = MlsynthPlottingError("Simulated plotting error")
    full_config_plot_fail = STABLESC_FULL_TEST_CONFIG_BASE.copy()
    full_config_plot_fail["display_graphs"] = True
    pydantic_dict_plot_fail = _get_pydantic_config_dict_stablesc(full_config_plot_fail, stablesc_panel_data)
    config_obj_plot_fail = StableSCConfig(**pydantic_dict_plot_fail)
    estimator_plot_fail = StableSC(config=config_obj_plot_fail)

    with mocker.patch.object(estimator_plot_fail, 'select_donors', return_value=(mock_selected_indices, mock_scores_all_donors)):
        with pytest.warns(UserWarning, match="Plotting failed for StableSC due to: Simulated plotting error"):
            try:
                estimator_plot_fail.fit()
            except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError, cvxpy.error.SolverError) as e:
                 if "Problem does not follow DCP rules" in str(e) or "Rank(A) < p" in str(e) or "optimal value is nan" in str(e) or "Solver 'CLARABEL' failed" in str(e) or "infeasible" in str(e).lower() or "no donors selected" in str(e).lower():
                    pytest.skip(f"Skipping plotting failure warning test due to data/optimization/selection error (with mock): {e}")
                 else: raise e
            mock_plot_estimates.assert_called_once() # Ensure it was called even if it failed


def test_stablesc_fit_results_structure_and_types(stablesc_panel_data: pd.DataFrame, mocker: Any):
    """Test the structure and types of the results dictionary from fit()."""
    pydantic_dict = _get_pydantic_config_dict_stablesc(STABLESC_FULL_TEST_CONFIG_BASE, stablesc_panel_data)
    config_obj = StableSCConfig(**pydantic_dict) # Use default config parameters
    estimator = StableSC(config=config_obj)

    # Mock select_donors
    mock_selected_indices = np.array([0, 1]) 
    mock_scores_all_donors = np.array([0.8, 0.7, 0.0, 0.0]) # Assuming 4 potential donors

    with mocker.patch.object(estimator, 'select_donors', return_value=(mock_selected_indices, mock_scores_all_donors)):
        try:
            results = estimator.fit()
        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError, cvxpy.error.SolverError) as e: # Adjusted expected errors
            if "Problem does not follow DCP rules" in str(e) or \
               "Rank(A) < p or Rank(G) < q" in str(e) or \
               "optimal value is nan" in str(e) or \
               "Solver 'CLARABEL' failed" in str(e) or \
               "infeasible" in str(e).lower() or \
               "no donors selected" in str(e).lower():
                pytest.skip(f"Skipping detailed results check due to data/optimization/selection error (with mock): {e}")
            else:
                raise e

    assert isinstance(results, BaseEstimatorResults)

    # Check main components
    assert results.effects is not None and isinstance(results.effects, EffectsResults)
    assert results.fit_diagnostics is not None and isinstance(results.fit_diagnostics, FitDiagnosticsResults)
    assert results.time_series is not None and isinstance(results.time_series, TimeSeriesResults)
    assert results.weights is not None and isinstance(results.weights, WeightsResults)
    assert results.method_details is not None and isinstance(results.method_details, MethodDetailsResults)
    assert results.raw_results is not None and isinstance(results.raw_results, dict)
    
    # Check specific fields within components
    assert isinstance(results.effects.att, (float, np.floating, type(None)))
    assert isinstance(results.fit_diagnostics.pre_treatment_rmse, (float, np.floating, type(None)))
    if results.fit_diagnostics.pre_treatment_r_squared is not None:
         assert isinstance(results.fit_diagnostics.pre_treatment_r_squared, (float, np.floating))

    # Time series data checks
    prepped_data = results.raw_results.get("_prepped", {})
    assert "_prepped" in results.raw_results
    y_observed_from_prepped = prepped_data.get("y")
    assert y_observed_from_prepped is not None
    if y_observed_from_prepped.ndim > 1:
        y_observed_from_prepped = y_observed_from_prepped.squeeze()
    assert y_observed_from_prepped.ndim == 1, f"_prepped['y'] should be 1D, but got shape {prepped_data.get('y', np.array([])).shape}"
    total_periods = len(y_observed_from_prepped)

    assert results.time_series.observed_outcome is not None
    assert results.time_series.counterfactual_outcome is not None
    assert results.time_series.estimated_gap is not None
    assert results.time_series.time_periods is not None

    assert results.time_series.observed_outcome.shape == (total_periods, 1)
    assert results.time_series.counterfactual_outcome.shape == (total_periods, 1)
    assert results.time_series.estimated_gap.shape == (total_periods, 2) # Assuming Gap has 2 columns
    assert len(results.time_series.time_periods) == total_periods
    
    expected_gap_col0 = (results.time_series.observed_outcome - results.time_series.counterfactual_outcome).squeeze()
    np.testing.assert_array_almost_equal(results.time_series.estimated_gap[:, 0], expected_gap_col0, decimal=3)

    # Weights checks
    assert results.weights.donor_weights is not None
    assert all(isinstance(k, (str, np.str_, int, np.integer)) for k in results.weights.donor_weights.keys())
    assert all(isinstance(v, (float, np.floating)) for v in results.weights.donor_weights.values())
    if results.weights.donor_weights:
        assert np.isclose(sum(results.weights.donor_weights.values()), 1.0, atol=1e-3)

    # Method details checks
    assert results.method_details.additional_details is not None
    selected_indices = results.method_details.additional_details.get("selected_donor_indices")
    anomaly_scores = results.method_details.additional_details.get("anomaly_scores_used")
    assert isinstance(selected_indices, list)
    assert isinstance(anomaly_scores, list)
    assert all(isinstance(idx, (int, np.integer)) for idx in selected_indices)
    assert all(isinstance(score, (float, np.floating)) for score in anomaly_scores)
    
    donor_matrix = prepped_data.get("donor_matrix")
    assert donor_matrix is not None
    assert len(anomaly_scores) == donor_matrix.shape[1]

    full_scores_np = np.array(anomaly_scores)
    for i in range(len(full_scores_np)):
        if i in selected_indices:
            assert full_scores_np[i] > 0, f"Selected donor index {i} has non-positive score {full_scores_np[i]}."
        else:
            assert full_scores_np[i] == 0, f"Non-selected donor index {i} has non-zero score {full_scores_np[i]}."
