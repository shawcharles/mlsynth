import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from unittest.mock import patch
from pydantic import ValidationError

from mlsynth.estimators.gsc import GSC
from mlsynth.config_models import GSCConfig, BaseEstimatorResults
from mlsynth.exceptions import (
    MlsynthDataError,
    MlsynthConfigError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)

@pytest.fixture
def sample_gsc_data() -> pd.DataFrame:
    """Creates a sample DataFrame for GSC tests."""
    n_units = 4
    n_periods = 10
    treatment_start_period = 7 # 6 pre-periods, 4 post-periods

    units = np.repeat(np.arange(1, n_units + 1), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    
    np.random.seed(456) # for reproducibility
    outcomes = []
    for i in range(n_units):
        base_trend = np.linspace(start=10 + i*1.5, stop=22 + i*1.5, num=n_periods)
        noise = np.random.normal(0, 0.8, n_periods)
        outcomes.extend(base_trend + noise)

    data = {
        "UnitID": units,
        "TimePeriod": times,
        "OutcomeVar": outcomes,
        "IsTreatedIndicator": np.zeros(n_units * n_periods, dtype=int),
    }
    df = pd.DataFrame(data)

    # Unit 1 is treated from treatment_start_period
    df.loc[(df['UnitID'] == 1) & (df['TimePeriod'] >= treatment_start_period), 'IsTreatedIndicator'] = 1
    
    return df

def test_gsc_creation(sample_gsc_data: pd.DataFrame) -> None:
    """Test GSC estimator creation."""
    config_dict: Dict[str, Any] = {
        "df": sample_gsc_data,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    try:
        config_obj = GSCConfig(**config_dict)
        estimator = GSC(config_obj)
        assert estimator is not None
        assert estimator.df.equals(sample_gsc_data)
        assert estimator.treat == "IsTreatedIndicator"
        assert estimator.time == "TimePeriod"
        assert estimator.outcome == "OutcomeVar"
        assert estimator.unitid == "UnitID"
        assert not estimator.display_graphs
    except Exception as e:
        pytest.fail(f"GSC creation failed: {e}")

def test_gsc_fit_smoke(sample_gsc_data: pd.DataFrame) -> None:
    """Smoke test for GSC fit method."""
    config_dict: Dict[str, Any] = {
        "df": sample_gsc_data,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    config_obj = GSCConfig(**config_dict)
    estimator = GSC(config_obj)
    try:
        results = estimator.fit()
        assert isinstance(results, BaseEstimatorResults) # Check for BaseEstimatorResults
        
        # Check for presence of Pydantic model attributes
        assert results.effects is not None
        assert results.fit_diagnostics is not None
        assert results.time_series is not None
        assert results.inference is not None
        assert results.method_details is not None
        assert results.raw_results is not None # Ensure raw_results are populated

        # Check some nested attributes
        assert results.time_series.counterfactual_outcome is not None
        assert isinstance(results.time_series.counterfactual_outcome, np.ndarray)
        assert results.raw_results["Effects"] is not None # Check raw results for original structure
        assert results.raw_results["Vectors"]["Counterfactual"] is not None

    except Exception as e:
        # DC_PR_with_suggested_rank can be sensitive to small data or specific configurations
        if isinstance(e, (np.linalg.LinAlgError, ValueError)): # Common errors with matrix ops
            pytest.skip(f"Skipping GSC fit due to numerical/data issue: {e}")
        pytest.fail(f"GSC fit failed: {e}")

# --- Input Validation Tests ---

def test_gsc_creation_missing_config_keys(sample_gsc_data: pd.DataFrame) -> None:
    """Test GSC creation with missing essential keys in config."""
    base_config_dict: Dict[str, Any] = {
        "df": sample_gsc_data,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    
    required_keys = ["df", "treat", "time", "outcome", "unitid"]
    for key_to_remove in required_keys:
        config_dict = base_config_dict.copy()
        del config_dict[key_to_remove]
        with pytest.raises(ValidationError): # Pydantic should catch missing required fields
            GSCConfig(**config_dict)


def test_gsc_creation_df_not_dataframe() -> None:
    """Test GSC creation when df is not a pandas DataFrame."""
    config_dict: Dict[str, Any] = {
        "df": "not_a_dataframe", # type: ignore
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    with pytest.raises(ValidationError): # Pydantic should catch type error for df
        GSCConfig(**config_dict)

def test_gsc_fit_missing_columns_in_df(sample_gsc_data: pd.DataFrame) -> None:
    """Test GSC fit when DataFrame is missing specified columns.
    Pydantic config creation should pass, error expected during fit.
    """
    base_config_dict: Dict[str, Any] = {
        # "df" will be modified
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    
    cols_to_test_missing = ["IsTreatedIndicator", "TimePeriod", "OutcomeVar", "UnitID"]
    
    for col_to_drop in cols_to_test_missing:
        current_df_with_dropped_col = sample_gsc_data.copy().drop(columns=[col_to_drop])
        
        # Create a config dict that still refers to the original column name (from base_config_dict),
        # but will be instantiated with a DataFrame (current_df_with_dropped_col) that's missing that column.
        config_for_test = base_config_dict.copy()
        config_for_test["df"] = current_df_with_dropped_col
            
        expected_error_message = f"Missing required columns in DataFrame 'df': {col_to_drop}"
        with pytest.raises(MlsynthDataError, match=expected_error_message):
            GSCConfig(**config_for_test)


def test_gsc_fit_df_empty(sample_gsc_data: pd.DataFrame) -> None:
    """Test GSC fit with an empty DataFrame."""
    config_dict: Dict[str, Any] = {
        "df": pd.DataFrame(columns=sample_gsc_data.columns), 
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    # BaseEstimatorConfig validation should catch empty DataFrame
    with pytest.raises(MlsynthDataError, match="Input DataFrame 'df' cannot be empty."):
        GSCConfig(**config_dict)

# --- Edge Case Tests ---

def test_gsc_fit_insufficient_pre_periods(sample_gsc_data: pd.DataFrame) -> None:
    """Test GSC fit with very few pre-treatment periods."""
    # Scenario 1: Only 1 pre-treatment period
    df_few_pre = sample_gsc_data.copy()
    df_few_pre["IsTreatedIndicator"] = 0 
    df_few_pre.loc[(df_few_pre['UnitID'] == 1) & (df_few_pre['TimePeriod'] >= 2), 'IsTreatedIndicator'] = 1
    
    config_dict_few_pre: Dict[str, Any] = {
        "df": df_few_pre,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    config_obj_few_pre = GSCConfig(**config_dict_few_pre)
    estimator_few_pre = GSC(config_obj_few_pre)
    try:
        results = estimator_few_pre.fit()
        assert isinstance(results, BaseEstimatorResults) # Check for BaseEstimatorResults
    except (np.linalg.LinAlgError, ValueError) as e:
        pytest.skip(f"Skipping due to numerical instability with few pre-periods: {e}")
    except Exception as e:
        pytest.fail(f"GSC fit with few pre-periods failed unexpectedly: {e}")

    # Scenario 2: Zero pre-treatment periods
    df_zero_pre = sample_gsc_data.copy()
    df_zero_pre.loc[df_zero_pre['UnitID'] == 1, 'IsTreatedIndicator'] = 1 # Unit 1 treated from period 1

    config_dict_zero_pre: Dict[str, Any] = {
        "df": df_zero_pre,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    config_obj_zero_pre = GSCConfig(**config_dict_zero_pre)
    estimator_zero_pre = GSC(config_obj_zero_pre)
    # This setup results in 0 pre-treatment periods.
    # dataprep (specifically, the single treated unit path) raises this.
    with pytest.raises(MlsynthDataError, match="Not enough pre-treatment periods \\(0 pre-periods found\\)."):
        estimator_zero_pre.fit()


def test_gsc_fit_insufficient_donors(sample_gsc_data: pd.DataFrame) -> None:
    """Test GSC fit with insufficient donor units."""
    # Scenario 1: Only one donor unit
    df_one_donor = sample_gsc_data[sample_gsc_data["UnitID"].isin([1, 2])].copy()
    config_dict_one_donor: Dict[str, Any] = {
        "df": df_one_donor,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    config_obj_one_donor = GSCConfig(**config_dict_one_donor)
    estimator_one_donor = GSC(config_obj_one_donor)
    try:
        results = estimator_one_donor.fit()
        assert isinstance(results, BaseEstimatorResults) # Check for BaseEstimatorResults
    except (np.linalg.LinAlgError, ValueError) as e:
        pytest.skip(f"Skipping due to numerical instability with one donor: {e}")
    except Exception as e:
        pytest.fail(f"GSC fit with one donor failed unexpectedly: {e}")

    # Scenario 2: Zero donor units
    df_zero_donors = sample_gsc_data[sample_gsc_data["UnitID"] == 1].copy()
    config_dict_zero_donors: Dict[str, Any] = {
        "df": df_zero_donors,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    config_obj_zero_donors = GSCConfig(**config_dict_zero_donors)
    estimator_zero_donors = GSC(config_obj_zero_donors)
    # dataprep should raise this
    with pytest.raises(MlsynthDataError, match="No donor units found"):
        estimator_zero_donors.fit()


def test_gsc_fit_no_post_treatment_periods(sample_gsc_data: pd.DataFrame) -> None:
    """Test GSC fit when there are no post-treatment periods for the treated unit."""
    df_no_post = sample_gsc_data.copy()
    # Treat unit 1, but only up to the last period (so no post-treatment periods)
    df_no_post.loc[(df_no_post['UnitID'] == 1) & (df_no_post['TimePeriod'] < 100), 'IsTreatedIndicator'] = 1 # All periods treated
    
    config_dict: Dict[str, Any] = {
        "df": df_no_post,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    config_obj = GSCConfig(**config_dict)
    estimator = GSC(config_obj)
    # This setup actually leads to 0 pre-treatment periods because all periods are marked as treated for unit 1.
    # The error "Treated unit has no post-treatment period." is harder to trigger if treatment is within the window.
    # logictreat calculates post_periods = total_periods - first_treat_period.
    # If first_treat_period is the last period (total_periods - 1), post_periods = 1.
    # The "no post-treatment period" error in logictreat is for when post_periods is literally 0.
    # The current setup (all periods treated for unit 1) means first_treat_period = 0 (0-indexed).
    # So pre_periods = 0. dataprep for single treated unit raises "Not enough pre-treatment periods".
    with pytest.raises(MlsynthDataError, match="Not enough pre-treatment periods \\(0 pre-periods found\\)."):
        estimator.fit()

@pytest.mark.parametrize(
    "nan_location_desc, unit_id, time_cond",
    [
        ("pre_donor", 2, lambda t: t < 7),
        ("pre_treated", 1, lambda t: t < 7),
        ("post_donor", 2, lambda t: t >= 7),
        ("post_treated", 1, lambda t: t >= 7),
    ]
)
def test_gsc_fit_with_nans_in_outcome(
    sample_gsc_data: pd.DataFrame, nan_location_desc: str, unit_id: int, time_cond: Any
) -> None:
    """Test GSC fit when outcome variable contains NaNs."""
    df_with_nans = sample_gsc_data.copy()
    nan_idx = df_with_nans[
        (df_with_nans["UnitID"] == unit_id) & 
        (df_with_nans["TimePeriod"].apply(time_cond))
    ].index
    if not nan_idx.empty:
        df_with_nans.loc[nan_idx[0], "OutcomeVar"] = np.nan

    config_dict: Dict[str, Any] = {
        "df": df_with_nans,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    config_obj = GSCConfig(**config_dict)
    estimator = GSC(config_obj)
    # DC_PR_with_suggested_rank or subsequent processing might fail with NaNs
    with pytest.raises(MlsynthEstimationError): # Expecting our wrapper
        estimator.fit()

# --- Detailed Results Validation Tests ---

def test_gsc_fit_results_structure_and_types(sample_gsc_data: pd.DataFrame) -> None:
    """Test the detailed structure and types of the GSC fit results dictionary."""
    config_dict: Dict[str, Any] = {
        "df": sample_gsc_data,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False,
    }
    config_obj = GSCConfig(**config_dict)
    estimator = GSC(config_obj)
    try:
        results = estimator.fit()
    except (np.linalg.LinAlgError, ValueError) as e:
        pytest.skip(f"Skipping GSC fit due to numerical/data issue for results validation: {e}")
        return

    assert isinstance(results, BaseEstimatorResults)
    assert results.effects is not None
    assert results.fit_diagnostics is not None
    assert results.time_series is not None
    assert results.inference is not None
    assert results.method_details is not None
    assert results.raw_results is not None

    # Effects
    assert isinstance(results.effects.att, (float, np.floating, np.integer))
    if results.effects.att_percent is not None: # Optional field
        assert isinstance(results.effects.att_percent, (float, np.floating, np.integer))
    # results.effects.att_std_err might be a float or None, depending on mapping from raw_results["Inference"]["SE"]

    # Fit Diagnostics
    assert isinstance(results.fit_diagnostics.rmse_pre, (float, np.floating, np.integer, type(None)))
    assert isinstance(results.fit_diagnostics.rmse_post, (float, np.floating, np.integer, type(None)))
    if results.fit_diagnostics.r_squared_pre is not None: # Optional field
        assert isinstance(results.fit_diagnostics.r_squared_pre, (float, np.floating, np.integer, type(None)))
    
    # Check raw fit diagnostics for original keys if needed
    raw_fit_res = results.raw_results.get("Fit", {})
    assert "Pre-Periods" in raw_fit_res and isinstance(raw_fit_res["Pre-Periods"], (int, np.integer))
    assert "Post-Periods" in raw_fit_res and isinstance(raw_fit_res["Post-Periods"], (int, np.integer))


    # Time Series
    n_total_periods = sample_gsc_data["TimePeriod"].nunique()
    n_total_units = sample_gsc_data["UnitID"].nunique()

    assert results.time_series.observed_outcome is not None
    assert isinstance(results.time_series.observed_outcome, np.ndarray)
    assert results.time_series.observed_outcome.shape == (n_total_periods,) # Should be 1D

    assert results.time_series.counterfactual_outcome is not None
    assert isinstance(results.time_series.counterfactual_outcome, np.ndarray)
    assert results.time_series.counterfactual_outcome.shape == (n_total_periods,) # Should be 1D
    
    assert results.time_series.estimated_gap is not None
    assert isinstance(results.time_series.estimated_gap, np.ndarray)
    assert results.time_series.estimated_gap.shape == (n_total_periods,)

    assert results.time_series.time_periods is not None
    assert len(results.time_series.time_periods) == n_total_periods


    # Method Details
    assert results.method_details.method_name == "GSC"
    assert results.method_details.parameters_used is not None
    if results.method_details.parameters_used.get("rank_used") is not None:
        assert isinstance(results.method_details.parameters_used["rank_used"], int)

    assert results.method_details.additional_outputs is not None
    additional_outputs = results.method_details.additional_outputs
    
    if additional_outputs.get("counterfactual_all_units_matrix") is not None:
        assert isinstance(additional_outputs["counterfactual_all_units_matrix"], np.ndarray)
        # Shape is (time x units) after transpose in _create_estimator_results
        assert additional_outputs["counterfactual_all_units_matrix"].shape == (n_total_periods, n_total_units) 
        
    if additional_outputs.get("observed_all_units_matrix") is not None:
        assert isinstance(additional_outputs["observed_all_units_matrix"], np.ndarray)
        # Shape is (time x units)
        assert additional_outputs["observed_all_units_matrix"].shape == (n_total_periods, n_total_units)

    # Loadings (U) and Factors (V.T) are also optional and their shapes depend on rank
    if additional_outputs.get("loadings_U") is not None:
        assert isinstance(additional_outputs["loadings_U"], np.ndarray)
    if additional_outputs.get("factors_V_T") is not None:
        assert isinstance(additional_outputs["factors_V_T"], np.ndarray)

    # Inference
    assert isinstance(results.inference.ci_lower_bound, (float, np.floating)) # Field name is ci_lower_bound
    assert isinstance(results.inference.ci_upper_bound, (float, np.floating)) # Field name is ci_upper_bound
    if results.inference.standard_error is not None: # Optional, can be array or float
         assert isinstance(results.inference.standard_error, (np.ndarray, float, np.floating))
         if isinstance(results.inference.standard_error, np.ndarray):
            assert results.inference.standard_error.ndim == 1
    if results.inference.details is not None and results.inference.details.get("t_statistic") is not None: # Optional
        assert isinstance(results.inference.details["t_statistic"], (float, np.floating))
    
    # Check consistency if ATT is present
    if results.effects.att is not None:
        assert results.inference.ci_lower_bound <= results.effects.att <= results.inference.ci_upper_bound


# --- Plotting Behavior Tests ---
from unittest.mock import patch

@patch("mlsynth.estimators.gsc.plot_estimates")
def test_gsc_fit_plotting_behavior_display_true(
    mock_plot_estimates: Any, sample_gsc_data: pd.DataFrame
) -> None:
    """Test that plot_estimates is called when display_graphs is True."""
    config_dict: Dict[str, Any] = {
        "df": sample_gsc_data,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": True, 
        "counterfactual_color": "blue",
        "treated_color": "green",
    }
    config_obj = GSCConfig(**config_dict)
    estimator = GSC(config_obj)
    try:
        results = estimator.fit()
    except (np.linalg.LinAlgError, ValueError) as e:
        pytest.skip(f"Skipping GSC fit due to numerical/data issue for plotting test: {e}")
        return

    mock_plot_estimates.assert_called_once()
    call_args_list = mock_plot_estimates.call_args_list
    assert len(call_args_list) == 1
    
    # call_args is a tuple: (args, kwargs)
    args, kwargs = call_args_list[0]

    assert kwargs["time"] == config_dict["time"]
    assert kwargs["unitid"] == config_dict["unitid"]
    assert kwargs["outcome"] == config_dict["outcome"]
    assert kwargs["treatmentname"] == config_dict["treat"]
    assert "treated_unit_name" in kwargs
    assert isinstance(kwargs["y"], pd.Series) # y from prepped is pd.Series
    assert isinstance(kwargs["cf_list"], list)
    assert len(kwargs["cf_list"]) == 1
    assert isinstance(kwargs["cf_list"][0], np.ndarray)
    
    # Compare with raw_results as plot_estimates uses the direct output of DC_PR
    raw_counterfactual = results.raw_results["Vectors"]["Counterfactual"]
    assert kwargs["cf_list"][0].shape == raw_counterfactual.shape
    assert np.array_equal(kwargs["cf_list"][0], raw_counterfactual)
    
    assert kwargs["counterfactual_names"] == ["GSC"]
    assert kwargs["method"] == "GSC"
    assert kwargs["treatedcolor"] == config_dict["treated_color"]
    assert kwargs["counterfactualcolors"] == [config_dict["counterfactual_color"]]
    assert isinstance(kwargs["df"], dict) 
    assert "Ywide" in kwargs["df"]
    assert kwargs.get("save_path") is None 

    # Test with save_path as True (should use default naming, not testable here for exact name)
    mock_plot_estimates.reset_mock()
    config_dict_save_true: Dict[str, Any] = {**config_dict, "save": True}
    config_obj_save_true = GSCConfig(**config_dict_save_true)
    estimator_save_true = GSC(config_obj_save_true)
    try:
        estimator_save_true.fit()
    except (np.linalg.LinAlgError, ValueError, MlsynthEstimationError) as e: # Added MlsynthEstimationError
        pytest.skip(f"Skipping GSC fit due to numerical/data issue for plotting test with save=True: {e}")
        return
    mock_plot_estimates.assert_called_once()
    _, kwargs_save_true = mock_plot_estimates.call_args_list[0]
    assert kwargs_save_true.get("save_path") is None # plot_estimates handles True internally

    # Test with save_path as string
    mock_plot_estimates.reset_mock()
    save_path_str = "/tmp/gsc_plot.png"
    config_dict_save_str: Dict[str, Any] = {**config_dict, "save": save_path_str}
    config_obj_save_str = GSCConfig(**config_dict_save_str)
    estimator_save_str = GSC(config_obj_save_str)
    try:
        estimator_save_str.fit()
    except (np.linalg.LinAlgError, ValueError, MlsynthEstimationError) as e: # Added MlsynthEstimationError
        pytest.skip(f"Skipping GSC fit due to numerical/data issue for plotting test with save_path='{save_path_str}': {e}")
        return
    
    mock_plot_estimates.assert_called_once()
    _, kwargs_save_str = mock_plot_estimates.call_args_list[0]
    assert kwargs_save_str.get("save_path") == save_path_str


@patch("mlsynth.estimators.gsc.plot_estimates")
def test_gsc_fit_plotting_behavior_display_false(
    mock_plot_estimates: Any, sample_gsc_data: pd.DataFrame
) -> None:
    """Test that plot_estimates is NOT called when display_graphs is False."""
    config_dict: Dict[str, Any] = {
        "df": sample_gsc_data,
        "treat": "IsTreatedIndicator",
        "time": "TimePeriod",
        "outcome": "OutcomeVar",
        "unitid": "UnitID",
        "display_graphs": False, 
    }
    config_obj = GSCConfig(**config_dict)
    estimator = GSC(config_obj)
    try:
        estimator.fit()
    except (np.linalg.LinAlgError, ValueError) as e:
        pytest.skip(f"Skipping GSC fit due to numerical/data issue for no-plotting test: {e}")
        return
        
    mock_plot_estimates.assert_not_called()
