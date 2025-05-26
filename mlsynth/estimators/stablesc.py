import pandas as pd
import numpy as np
import warnings
import pydantic
from scipy.stats import chi2
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, Any, List, Tuple, Union, Optional

from ..utils.datautils import balance, dataprep
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.estutils import Opt
from ..utils.resultutils import effects, plot_estimates
from ..config_models import (
    StableSCConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)

# Constants for plotting
_PLOT_METHOD_NAME = "StableSC"
_PLOT_COUNTERFACTUAL_NAME = "Hybrid StableSC"

class StableSC:
    """Stable Synthetic Control (StableSC) estimator.

    This estimator implements a Stable Synthetic Control method, which enhances
    traditional synthetic control by employing a hybrid anomaly-based donor
    selection process. It identifies suitable donor units by combining insights
    from Granger causality tests and proximity measures (based on chi-squared
    distances of time series). This approach aims to construct a more robust
    and stable synthetic counterfactual for the treated unit.

    The core idea is to filter potential donors first by their Granger causality
    with the treated unit's outcome and their proximity to the average of other
    donors. Then, Radial Basis Function (RBF) scores are computed based on
    proximity distances to weigh the selected donors. Finally, an optimization
    is performed to find the synthetic control weights.

    Attributes
    ----------
    config : StableSCConfig
        The configuration object passed during instantiation, holding all
        parameters for the estimator.
        (Inherited from `BaseEstimatorConfig` via `StableSCConfig`)
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `StableSCConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `StableSCConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `StableSCConfig`)
    unitid : str
        Name of the unit identifier (ID) column in `df`.
        (Inherited from `BaseEstimatorConfig` via `StableSCConfig`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `StableSCConfig`)
    counterfactual_color : Union[str, List[str]]
        Color(s) to use for the counterfactual line(s) in plots.
        (Inherited from `BaseEstimatorConfig` via `StableSCConfig`)
    treated_color : str
        Color to use for the treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `StableSCConfig`)
    display_graphs : bool
        If True, graphs of the results will be displayed.
        (Inherited from `BaseEstimatorConfig` via `StableSCConfig`)
    save : Union[bool, str, Dict[str, str]]
        Configuration for saving plots.

        - If `False` (default), plots are not saved.
        - If `True`, plots are saved with default names in the current directory.
        - If a `str`, it's used as the base filename for saved plots.
        - If a `Dict[str, str]`, it maps specific plot keys (e.g., "estimates_plot")
          to full file paths.
        (Inherited from `BaseEstimatorConfig` via `StableSCConfig`)

    Methods
    -------
    fit()
        Fits the StableSC model and returns the estimated treatment effects
        and other diagnostics.
    normalize(Y)
        Normalizes an array by subtracting its column means.
    granger_mask(y, Y0, T0, alpha=0.05, maxlag=1)
        Computes a Granger causality mask for donor selection.
    proximity_mask(Y0, T0, alpha=0.05)
        Computes a proximity mask based on chi-squared distances.
    rbf_scores(dists, sigma=1.0)
        Calculates Radial Basis Function (RBF) scores from distances.
    select_donors(y, Y0, T0, alpha=0.1, sigma=20.0)
        Selects donor units based on hybrid Granger and proximity criteria.
    """

    def __init__(self, config: StableSCConfig) -> None:
        """Initialize the StableSC estimator.

        Parameters
        ----------

        config : StableSCConfig
            Configuration object for the estimator. Its attributes, primarily
            inherited from `BaseEstimatorConfig`, define the data and
            behavior of the estimator. Key attributes include:

            - df (pd.DataFrame): The input DataFrame containing panel data.
            - outcome (str): Name of the outcome variable column in `df`.
            - treat (str): Name of the treatment indicator column in `df`.
            - unitid (str): Name of the unit identifier (ID) column in `df`.
            - time (str): Name of the time variable column in `df`.
            - display_graphs (bool, optional): Whether to display graphs. Defaults to True.
            - save (Union[bool, str, Dict[str, str]], optional): Configuration for saving plots.
              If `False` (default), plots are not saved. If `True`, plots are saved with
              default names. If a `str`, it's used as the base filename. If a `Dict[str, str]`,
              it maps plot keys to full file paths. Defaults to False.
            - counterfactual_color (Union[str, List[str]], optional): Color(s) for counterfactual
              line(s). Defaults to "red".
            - treated_color (str, optional): Color for treated unit line. Defaults to "black".

            For authoritative definitions, defaults, and potential validation rules,
            refer to the Pydantic models `mlsynth.config_models.StableSCConfig`
            and `mlsynth.config_models.BaseEstimatorConfig`.
        """
        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, Dict[str, str]] = config.save
        # Store specific config values for StableSC from the Pydantic model
        self.granger_alpha: float = config.granger_alpha # Significance level for Granger causality tests.
        self.granger_maxlag: int = config.granger_maxlag # Max lag for Granger causality tests.
        self.proximity_alpha: float = config.proximity_alpha # Significance level for proximity (chi-squared) tests.
        self.rbf_sigma_fit: float = config.rbf_sigma_fit # Sigma for RBF kernel when scoring donors.
        self.sc_model_type: str = config.sc_model_type # SCM optimization model type (e.g., "OLS", "SIMPLEX").

    def normalize(self, input_array: np.ndarray) -> np.ndarray:
        """Normalize an array by subtracting its column means.

        This is a common preprocessing step to center the data before
        further calculations like distance or causality measures.

        Parameters
        ----------

        input_array : np.ndarray
            Input array, typically of shape (n_timeperiods, n_features) or
            (n_timeperiods, n_donors).

        Returns
        -------

        np.ndarray
            Normalized array of the same shape as `input_array`, where each column
            has been mean-centered.
        """
        try:
            if not isinstance(input_array, np.ndarray):
                raise MlsynthDataError("Input 'input_array' must be a NumPy array.")
            return input_array - input_array.mean(axis=0, keepdims=True)
        except (TypeError, AttributeError, ValueError) as e:
            raise MlsynthDataError(
                f"Error normalizing input array: {e}. Ensure input is a valid NumPy array."
            ) from e

    def granger_mask(
        self,
        treated_outcome_series: np.ndarray,
        donor_outcomes_matrix: np.ndarray,
        num_pre_treatment_periods: int,
        alpha: float = 0.05,
        maxlag: int = 1,
    ) -> np.ndarray:
        """Compute a Granger causality mask for donor selection.

        This method tests whether each potential donor unit's time series
        Granger-causes the treated unit's time series during the pre-treatment
        period. Donors that significantly Granger-cause the outcome are
        considered more relevant.

        Parameters
        ----------

        treated_outcome_series : np.ndarray
            Time series of the outcome variable for the treated unit,
            typically normalized. Shape (n_timeperiods,).
        donor_outcomes_matrix : np.ndarray
            Time series data for all potential donor units, typically normalized.
            Shape (n_timeperiods, n_donors).
        num_pre_treatment_periods : int
            Number of pre-treatment time periods to use for the test.
            The first `num_pre_treatment_periods` observations of `treated_outcome_series`
            and `donor_outcomes_matrix` are used.
        alpha : float, default 0.05
            Significance level for the Granger causality test. If the p-value
            is less than `alpha`, the null hypothesis (no Granger causality)
            is rejected.
        maxlag : int, default 1
            The maximum number of lags to include in the Granger causality test.

        Returns
        -------

        np.ndarray
            A boolean array of shape (n_donors,). An element is True if the
            corresponding donor unit's time series Granger-causes the treated
            unit's time series at the specified significance level `alpha`.
            False otherwise, or if the test fails for a donor.
        """
        if not isinstance(treated_outcome_series, np.ndarray) or not isinstance(donor_outcomes_matrix, np.ndarray):
            raise MlsynthDataError("Inputs 'treated_outcome_series' and 'donor_outcomes_matrix' must be NumPy arrays.")
        if treated_outcome_series.ndim != 1:
            raise MlsynthDataError("'treated_outcome_series' must be a 1D NumPy array.")
        if donor_outcomes_matrix.ndim != 2:
            raise MlsynthDataError("'donor_outcomes_matrix' must be a 2D NumPy array.")
        if num_pre_treatment_periods <= 0:
            raise MlsynthConfigError("Number of pre-treatment periods must be positive for Granger causality tests.")
        if maxlag <=0:
            raise MlsynthConfigError("Maxlag for Granger causality must be positive.")
        if not (0 < alpha < 1):
            raise MlsynthConfigError("Alpha for Granger causality must be between 0 and 1.")

        if donor_outcomes_matrix.shape[0] < num_pre_treatment_periods or treated_outcome_series.shape[0] < num_pre_treatment_periods:
            raise MlsynthDataError(
                "Not enough observations for the specified number of pre-treatment periods in Granger causality test."
            )

        causality_mask: List[bool] = [] # List to store boolean results for each donor.
        for donor_idx in range(donor_outcomes_matrix.shape[1]): # Iterate through each potential donor.
            try:
                # --- Granger Causality Test Logic ---
                # The test requires a sufficient number of observations relative to the number of lags.
                # A common rule of thumb is T > k*p^2 + 1, where T is observations, k is variables (2 here), p is maxlag.
                # A simpler, conservative check: num_pre_treatment_periods should be adequately larger than maxlag.
                if num_pre_treatment_periods <= maxlag: # If not enough data points for the specified lags.
                     warnings.warn(
                        f"Skipping Granger causality test for donor {donor_idx}: "
                        f"num_pre_treatment_periods ({num_pre_treatment_periods}) "
                        f"is not sufficiently larger than maxlag ({maxlag}). Marking as non-causal.",
                        UserWarning,
                    )
                     causality_mask.append(False) # Mark as not Granger-causing.
                     continue

                # Prepare data for the test: a DataFrame with the treated unit's outcome and one donor's outcome.
                test_data_df = pd.DataFrame({
                    "y": treated_outcome_series[:num_pre_treatment_periods], # Treated unit's pre-treatment outcome.
                    "x": donor_outcomes_matrix[:num_pre_treatment_periods, donor_idx],
                })
                if test_data_df.isnull().values.any():
                    warnings.warn(
                        f"NaNs found in data for Granger causality test for donor {donor_idx}. Marking as non-causal.",
                        UserWarning,
                    )
                    causality_mask.append(False)
                    continue
                
                # Check for constant series, which can cause issues with the test
                if test_data_df["y"].nunique() <= 1 or test_data_df["x"].nunique() <= 1:
                    warnings.warn(
                        f"Constant series detected for Granger causality test for donor {donor_idx}. Marking as non-causal.",
                        UserWarning,
                    )
                    causality_mask.append(False)
                    continue

                granger_test_result = grangercausalitytests(
                    test_data_df[["y", "x"]], maxlag=maxlag, verbose=False
                )
                p_value: float = granger_test_result[maxlag][0]["ssr_ftest"][1]
                causality_mask.append(p_value < alpha)
            except ValueError as ve: # Catch specific errors from statsmodels if possible
                warnings.warn(
                    f"Granger causality test failed for donor {donor_idx} with ValueError: {ve}. Marking as non-causal.",
                    UserWarning,
                )
                causality_mask.append(False)
            except Exception as e:
                warnings.warn(
                    f"Granger causality test failed unexpectedly for donor {donor_idx}: {e}. Marking as non-causal.",
                    UserWarning,
                )
                causality_mask.append(False)
        return np.array(causality_mask)

    def proximity_mask(
        self,
        donor_outcomes_matrix: np.ndarray,
        num_pre_treatment_periods: int,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a proximity mask and distances for donor selection.

        This method assesses each potential donor unit's "proximity" to the
        average behavior of all *other* donor units during the pre-treatment
        period. It calculates the sum of squared differences between a donor's
        time series and the average time series of the remaining donors.
        A chi-squared test is then used to determine if this distance is
        statistically significant, identifying donors that are outliers
        compared to the general donor pool.

        Parameters
        ----------

        donor_outcomes_matrix : np.ndarray
            Time series data for all potential donor units, typically normalized.
            Shape (n_timeperiods, n_donors).
        num_pre_treatment_periods : int
            Number of pre-treatment time periods to use for the calculation.
            The first `num_pre_treatment_periods` observations of `donor_outcomes_matrix` are used.
        alpha : float, default 0.05
            Significance level for the chi-squared test. If a donor's
            distance (normalized by `num_pre_treatment_periods`) results in a value less than the
            chi-squared critical value (for `1-alpha` probability and `num_pre_treatment_periods`
            degrees of freedom), it is considered "proximal" (not an outlier).

        Returns
        -------

        Tuple[np.ndarray, np.ndarray]
            is_proximal_mask : np.ndarray
                A boolean array of shape (n_donors,). An element is True if the
                corresponding donor unit is considered proximal (not an outlier)
                based on the chi-squared test. False otherwise.
            proximity_distances : np.ndarray
                An array of shape (n_donors,) containing the calculated sum of
                squared differences (divided by `num_pre_treatment_periods`) for each donor unit.
        """
        if not isinstance(donor_outcomes_matrix, np.ndarray):
            raise MlsynthDataError("Input 'donor_outcomes_matrix' must be a NumPy array.")
        if donor_outcomes_matrix.ndim != 2:
            raise MlsynthDataError("'donor_outcomes_matrix' must be a 2D NumPy array.")
        if num_pre_treatment_periods <= 0:
            raise MlsynthConfigError("Number of pre-treatment periods must be positive for proximity mask.")
        if not (0 < alpha < 1):
            raise MlsynthConfigError("Alpha for proximity mask must be between 0 and 1.")
        if donor_outcomes_matrix.shape[0] < num_pre_treatment_periods:
             raise MlsynthDataError(
                "Not enough observations for the specified number of pre-treatment periods in proximity mask."
            )

        try:
            num_donors: int = donor_outcomes_matrix.shape[1]
            if num_donors == 0: # Should be caught by fit, but defensive
                return np.array([]), np.array([])

            proximity_distances: np.ndarray = np.zeros(num_donors)
            for donor_idx in range(num_donors):
                # Ensure there are other donors to compare against
                if num_donors <= 1: # If only one donor, its distance to "others" is undefined or zero.
                    proximity_distances[donor_idx] = 0 # Or handle as np.nan or specific flag
                    continue

                other_donors_outcomes: np.ndarray = np.delete(
                    donor_outcomes_matrix[:num_pre_treatment_periods], donor_idx, axis=1
                )
                # This check should be redundant if num_donors > 1, but good for safety
                if other_donors_outcomes.shape[1] == 0:
                    proximity_distances[donor_idx] = 0
                    continue
                
                average_other_donors_outcome: np.ndarray = other_donors_outcomes.mean(axis=1)
                diff_sq = (
                    donor_outcomes_matrix[:num_pre_treatment_periods, donor_idx]
                    - average_other_donors_outcome
                ) ** 2
                proximity_distances[donor_idx] = np.sum(diff_sq) / num_pre_treatment_periods
            
                # Degrees of freedom for the chi-squared test is taken as num_pre_treatment_periods.
                # This implicitly assumes that each squared difference term contributes independently to the sum.
                if num_pre_treatment_periods == 0: # Should have been caught by input validation earlier.
                 return np.array([False]*num_donors, dtype=bool), proximity_distances # Return empty/false if no pre-periods.

            # Calculate the critical value from the chi-squared distribution.
            # Donors whose (normalized) distance is less than this threshold are considered "proximal".
            chi2_threshold: float = chi2.ppf(1 - alpha, df=num_pre_treatment_periods)
            is_proximal_mask = proximity_distances < chi2_threshold # Boolean mask indicating proximal donors.
            return is_proximal_mask, proximity_distances
        except (ValueError, IndexError, TypeError, FloatingPointError) as e: # Catch potential numerical or data issues.
            raise MlsynthEstimationError(f"Error calculating proximity mask: {e}") from e

    def rbf_scores(
        self, distances_array: np.ndarray, sigma: float = 1.0
    ) -> np.ndarray:
        """Calculate Radial Basis Function (RBF) scores from distances.

        This method transforms distances (e.g., from `proximity_mask`) into
        similarity scores using an RBF (Gaussian) kernel. Smaller distances
        result in higher scores, indicating greater similarity.

        Parameters
        ----------
        distances_array : np.ndarray
            An array of distances, typically non-negative. Shape (n_donors,).
        sigma : float, default 1.0
            The width (standard deviation) parameter of the RBF kernel.
            It controls the rate at which similarity scores decrease with
            increasing distance.

        Returns
        -------
        np.ndarray
            An array of RBF scores, of the same shape as `distances_array`. Scores range
            from 0 (for very large distances) to 1 (for zero distance).
        """
        if not isinstance(distances_array, np.ndarray):
            raise MlsynthDataError("Input 'distances_array' must be a NumPy array.")
        if not isinstance(sigma, (int, float)):
            raise MlsynthConfigError("Input 'sigma' for RBF scores must be a numeric value.")
        if sigma <= 0:
            raise MlsynthConfigError("RBF sigma must be positive.")
        try:
            return np.exp(-(distances_array**2) / (2 * sigma**2))
        except (TypeError, ValueError, FloatingPointError) as e: # Added FloatingPointError
            raise MlsynthEstimationError(f"Error calculating RBF scores: {e}") from e

    def select_donors(
        self,
        treated_outcome_series: np.ndarray, # Full series for treated unit
        all_donor_outcomes_matrix: np.ndarray, # Full series for all donors
        num_pre_treatment_periods: int,
        granger_alpha_param: float,
        granger_maxlag_param: int,
        proximity_alpha_param: float,
        rbf_sigma_param: float,
    ) -> Tuple[np.ndarray, np.ndarray]: # Returns indices and scores for ALL donors
        """Select and score donor units based on hybrid criteria.

        This method implements the core donor selection logic for StableSC.
        It first normalizes the pre-treatment part of the treated unit's outcome
        and donor outcomes. Then, it applies both Granger causality (`granger_mask`)
        and proximity (`proximity_mask`) tests to identify a set of candidate donors.
        RBF scores are calculated based on the proximity distances.
        The method returns the indices of selected donors and the final scores
        for ALL original donors (scores are zero for unselected donors).

        Parameters
        ----------
        treated_outcome_series : np.ndarray
            Time series of the outcome variable for the treated unit (full period).
            Shape (n_timeperiods,).
        all_donor_outcomes_matrix : np.ndarray
            Time series data for all potential donor units (full period).
            Shape (n_timeperiods, n_donors).
        num_pre_treatment_periods : int
            Number of pre-treatment time periods to use for selection.
        granger_alpha_param : float
            Significance level for Granger causality tests.
        granger_maxlag_param : int
            Maximum lag for Granger causality tests.
        proximity_alpha_param : float
            Significance level for proximity mask chi-squared tests.
        rbf_sigma_param : float
            The width (sigma) parameter for the RBF kernel.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            selected_donor_indices : np.ndarray
                Array of integer indices (original column indices in `all_donor_outcomes_matrix`)
                of the selected donor units. Shape (n_selected_donors,).
            final_donor_selection_scores_all_donors : np.ndarray
                The RBF scores for all original donor units, where scores are zero
                for unselected donors. Shape (n_donors,).
        """
        try:
            # Validate inputs
            if not isinstance(treated_outcome_series, np.ndarray) or not isinstance(all_donor_outcomes_matrix, np.ndarray):
                raise MlsynthDataError("Inputs 'treated_outcome_series' and 'all_donor_outcomes_matrix' must be NumPy arrays.")
            if num_pre_treatment_periods <= 0:
                raise MlsynthConfigError("Number of pre-treatment periods must be positive for donor selection.")
            if all_donor_outcomes_matrix.shape[0] < num_pre_treatment_periods or \
               treated_outcome_series.shape[0] < num_pre_treatment_periods:
                raise MlsynthDataError("Not enough observations for the specified number of pre-treatment periods in select_donors.")

            # Normalize using only pre-treatment data
            normalized_treated_outcome_pre: np.ndarray = self.normalize(
                treated_outcome_series[:num_pre_treatment_periods]
            )
            normalized_all_donor_outcomes_pre: np.ndarray = self.normalize(
                all_donor_outcomes_matrix[:num_pre_treatment_periods]
            )

            granger_causality_mask: np.ndarray = self.granger_mask(
                normalized_treated_outcome_pre,
                normalized_all_donor_outcomes_pre,
                num_pre_treatment_periods,
                alpha=granger_alpha_param,
                maxlag=granger_maxlag_param,
            )
            is_proximal_mask, proximity_distances = self.proximity_mask(
                normalized_all_donor_outcomes_pre,
                num_pre_treatment_periods,
                alpha=proximity_alpha_param,
            )
            # Ensure masks are boolean arrays of the same shape for bitwise AND
            if not (isinstance(granger_causality_mask, np.ndarray) and granger_causality_mask.dtype == bool and
                    isinstance(is_proximal_mask, np.ndarray) and is_proximal_mask.dtype == bool and
                    granger_causality_mask.shape == is_proximal_mask.shape): # Validate mask compatibility.
                raise MlsynthEstimationError("Granger and proximity masks are not compatible for combination.")

            # Combine masks: a donor is a candidate if it passes BOTH Granger and proximity criteria.
            combined_selection_mask: np.ndarray = granger_causality_mask & is_proximal_mask
            
            # Calculate RBF scores based on the proximity distances for all donors.
            rbf_proximity_scores: np.ndarray = self.rbf_scores(
                proximity_distances, sigma=rbf_sigma_param
            )
            
            # Final scores for donor selection: RBF scores are applied only to donors in the combined_selection_mask.
            # Donors not meeting both criteria effectively get a score of 0.
            final_donor_selection_scores: np.ndarray = (
                combined_selection_mask * rbf_proximity_scores
            )
            
            # Identify indices of donors with a positive final score (i.e., selected donors).
            selected_donor_indices: np.ndarray = np.where(final_donor_selection_scores > 0)[0]
            
            # Return indices of selected donors and the RBF scores for ALL original donors (many will be zero).
            return selected_donor_indices, final_donor_selection_scores
        except (MlsynthDataError, MlsynthConfigError): # Re-raise specific custom errors from helper methods.
            raise
        except (ValueError, TypeError, IndexError, AttributeError) as e: # Catch common Python errors during selection.
            raise MlsynthEstimationError(f"Error during donor selection process: {e}") from e
        except Exception as e:
            raise MlsynthEstimationError(f"An unexpected error occurred in select_donors: {e}") from e

    def _create_estimator_results(
        self, raw_fit_output_dict: Dict[str, Any]
    ) -> BaseEstimatorResults:
        """Create a standardized results object from the fit output.

        This internal helper method takes the raw dictionary output from the
        `fit` method's core logic and populates a `BaseEstimatorResults`
        Pydantic model. This ensures a consistent structure for results
        returned by the estimator.

        Parameters
        ----------
        raw_fit_output_dict : Dict[str, Any]
            A dictionary containing the raw results from the fitting process.
            Expected keys include "Effects", "Fit", "Vectors", "Weights",
            "_prepped", "selected_donor_indices", and "anomaly_scores_used".

        Returns
        -------
        BaseEstimatorResults
            A Pydantic model instance containing the structured results. Key fields include:
            - effects (EffectsResults): Contains treatment effect estimates like ATT
              and percentage ATT.
            - fit_diagnostics (FitDiagnosticsResults): Includes goodness-of-fit metrics
              such as pre-treatment RMSE and R-squared.
            - time_series (TimeSeriesResults): Provides time-series data including the
              observed outcome for the treated unit, its estimated counterfactual outcome,
              the estimated treatment effect (gap) over time, and the corresponding
              time periods.
            - weights (WeightsResults): Contains a dictionary of weights assigned to
              the selected donor units.
            - inference (InferenceResults): Basic information indicating the estimation
              method ("Point estimate from weighted donors"), as StableSC's core logic
              doesn't inherently produce detailed statistical inference like p-values or CIs.
            - method_details (MethodDetailsResults): Details about the estimation method,
              including its name ("StableSC"), the configuration parameters used, and
              `additional_details` containing the indices of selected donors and the
              anomaly scores used in their selection.
            - raw_results (Optional[Dict[str, Any]]): The raw dictionary output from
              the fitting process.
        """
        try:
            prepared_panel_data = raw_fit_output_dict.get("_prepped", {})
            if not prepared_panel_data: # Ensure _prepped exists and is not empty
                 raise MlsynthEstimationError("'_prepped' data is missing or empty in raw_fit_output_dict.")


            raw_effects_data = raw_fit_output_dict.get("Effects", {})
            structured_effects_results = EffectsResults(
                att=raw_effects_data.get("ATT"),
                att_percent=raw_effects_data.get("Percent ATT"),
                additional_effects={
                    k: v
                    for k, v in raw_effects_data.items()
                    if k not in ["ATT", "Percent ATT"]
                },
            )

            raw_fit_diagnostics_data = raw_fit_output_dict.get("Fit", {})
            structured_fit_diagnostics_results = FitDiagnosticsResults(
                pre_treatment_rmse=raw_fit_diagnostics_data.get("T0 RMSE"),
                pre_treatment_r_squared=raw_fit_diagnostics_data.get("R-Squared"),
                additional_metrics={
                    k: v
                    for k, v in raw_fit_diagnostics_data.items()
                    if k not in ["T0 RMSE", "R-Squared"]
                },
            )

            sorted_time_periods_array = None
            # Check if 'Ywide' exists and is a DataFrame with an Index
            ywide_data = prepared_panel_data.get("Ywide")
            if isinstance(ywide_data, pd.DataFrame) and isinstance(ywide_data.index, pd.Index):
                time_period_values_from_index = ywide_data.index.to_numpy()
                if time_period_values_from_index.size > 0:
                    sorted_time_periods_array = time_period_values_from_index
            
            raw_time_series_vectors_data = raw_fit_output_dict.get("Vectors", {})
            structured_time_series_results = TimeSeriesResults(
                observed_outcome=raw_time_series_vectors_data.get("Observed Unit"),
                counterfactual_outcome=raw_time_series_vectors_data.get("Counterfactual"),
                estimated_gap=raw_time_series_vectors_data.get("Gap"),
                time_periods=sorted_time_periods_array,
            )

            structured_weights_results = WeightsResults(
                donor_weights=raw_fit_output_dict.get("Weights")
            )

            structured_inference_results = InferenceResults(
                method="Point estimate from weighted donors"
            )

            structured_method_details_results = MethodDetailsResults(
                name="StableSC",
                parameters_used=self.config.model_dump(exclude={"df"}),
                additional_details={
                    "selected_donor_indices": raw_fit_output_dict.get(
                        "selected_donor_indices"
                    ),
                    "anomaly_scores_used": raw_fit_output_dict.get("anomaly_scores_used"),
                },
            )

            return BaseEstimatorResults(
                effects=structured_effects_results,
                fit_diagnostics=structured_fit_diagnostics_results,
                time_series=structured_time_series_results,
                weights=structured_weights_results,
                inference=structured_inference_results,
                method_details=structured_method_details_results,
                raw_results=raw_fit_output_dict,
            )
        except pydantic.ValidationError as e:
            raise MlsynthEstimationError(f"Error validating results model for StableSC: {e}") from e
        except (KeyError, TypeError, AttributeError) as e:
            raise MlsynthEstimationError(
                f"Data inconsistency error during StableSC results creation: {e}. "
                "Check structure of raw_fit_output_dict."
            ) from e
        except Exception as e:
            raise MlsynthEstimationError(f"An unexpected error occurred while creating StableSC results: {e}") from e


    def fit(self) -> BaseEstimatorResults:
        """Fit the Stable Synthetic Control model.

        This method executes the full StableSC estimation pipeline:
        1. Balances the input panel data.
        2. Prepares data using `dataprep` (separates treated unit, donors, etc.).
        3. Performs donor selection using `select_donors` (which internally uses
           `normalize`, `granger_mask`, `proximity_mask`, and `rbf_scores`).
           Note: The current implementation in `fit` calls `granger_mask` and
           `proximity_mask` directly, effectively using their default `alpha` (0.05)
           and `maxlag` (1 for Granger) values. The `rbf_scores` are called with a
           hardcoded `rbf_kernel_width=20.0`. The `select_donors` method, which encapsulates
           these steps, has its own defaults (alpha=0.1, sigma=20.0) but is not
           directly used in `fit` in a way that leverages these defaults for the
           individual mask/score calculations.
        4. Computes synthetic control weights using optimization (`Opt.SCopt`)
           on the selected and scored donors.
        5. Generates the counterfactual outcome series.
        6. Optionally displays plots of the observed vs. counterfactual outcomes.
        7. Calculates treatment effects and fit diagnostics.
        8. Structures all results into a `BaseEstimatorResults` object via
           `_create_estimator_results`.

        Raises
        ------
        ValueError
            If there are insufficient pre-treatment periods (less than 2) for
            Granger causality tests, or if no donor units are available after
            data preparation, or if fewer than 2 donor units are available
            (as StableSC requires at least 2 for robust estimation).

        Returns
        -------
        BaseEstimatorResults
            A Pydantic model instance containing the comprehensive results of
            the estimation. Key fields include:

            - effects (EffectsResults)
                Treatment effect estimates (ATT, percentage ATT).
            - fit_diagnostics (FitDiagnosticsResults)
                Goodness-of-fit metrics (pre-treatment RMSE, R-squared).
            - time_series (TimeSeriesResults)
                Time series data (observed outcome, counterfactual outcome,
                estimated gap, time periods).
            - weights (WeightsResults)
                Dictionary of weights assigned to selected donor units.
            - inference (InferenceResults)
                Basic information about the estimation method.
            - method_details (MethodDetailsResults)
                Method name ("StableSC"), configuration parameters, and
                additional details like selected donor indices and anomaly scores.
            - raw_results (Optional[Dict[str, Any]])
                Raw dictionary output from the fitting process.

        Examples
        --------
        # doctest: +SKIP
        >>> import pandas as pd
        >>> from mlsynth.config_models import StableSCConfig
        >>> from mlsynth.estimators.stablesc import StableSC
        >>> # Assume `data` is a pandas DataFrame with columns:
        >>> # 'unit_id', 'time_period', 'outcome_var', 'treatment_status'
        >>> data = pd.DataFrame({
        ...     'unit_id': [1]*10 + [2]*10 + [3]*10 + [4]*10,
        ...     'time_period': list(range(1,11))*4,
        ...     'outcome_var': ([1,2,3,4,5,6,7,8,9,10] +  # Unit 1 (Treated)
        ...                     [1,2,3,4,4,3,2,1,1,1] +    # Unit 2 (Donor)
        ...                     [2,3,4,5,6,5,4,3,2,2] +    # Unit 3 (Donor)
        ...                     [0,1,2,3,4,5,6,7,8,8]),   # Unit 4 (Donor)
        ...     'treatment_status': ([0]*5 + [1]*5) + [0]*30 # Unit 1 treated at period 6
        ... })
        >>> config = StableSCConfig(
        ...     df=data,
        ...     unitid='unit_id',
        ...     time='time_period',
        ...     outcome='outcome_var',
        ...     treat='treatment_status',
        ...     display_graphs=False # Keep example concise
        ... )
        >>> estimator = StableSC(config=config)
        >>> results = estimator.fit()
        >>> print(f"Estimated ATT: {results.effects.att}")
        >>> # Access other results like:
        >>> # results.fit_diagnostics.pre_treatment_rmse
        >>> # results.weights.donor_weights
        >>> # results.time_series.counterfactual_outcome
        """
        try:
            # Initial data balancing and preparation
            # Initial data balancing and preparation.
            # `balance` ensures all units have observations for all time periods.
            # `dataprep` structures the data for estimation (e.g., separates treated/donor outcomes).
            # These functions can raise MlsynthDataError or MlsynthConfigError on failure.
            balance(self.df, self.unitid, self.time)
            prepared_panel_data: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )

            # Validate that critical information (period counts, outcome data) is available from dataprep.
            try:
                num_pre_treatment_periods = prepared_panel_data["pre_periods"]
                num_post_periods = prepared_panel_data["post_periods"]
                treated_outcome_all_periods: np.ndarray = prepared_panel_data["y"]
                all_donor_outcomes_all_periods: np.ndarray = prepared_panel_data["donor_matrix"]
            except KeyError as e:
                raise MlsynthDataError(
                    f"Critical key {e} missing from dataprep output. "
                    "This indicates an issue in dataprep logic or inconsistent data."
                ) from e

            # Validate data quality and suitability for StableSC
            if num_pre_treatment_periods < 2:
                raise MlsynthConfigError( # Or MlsynthDataError if it's about data quality
                    f"Not enough pre-treatment periods ({num_pre_treatment_periods}) for reliable donor selection "
                    "via Granger causality (requires at least 2)."
                )
            if not isinstance(all_donor_outcomes_all_periods, np.ndarray) or all_donor_outcomes_all_periods.ndim != 2:
                 raise MlsynthDataError("Donor matrix from dataprep is not a valid 2D NumPy array.")
            if all_donor_outcomes_all_periods.shape[1] == 0:
                raise MlsynthDataError("No donor units available from dataprep.")
            if all_donor_outcomes_all_periods.shape[1] < 2:
                raise MlsynthDataError(
                    f"StableSC requires at least 2 donor units for robust estimation, but found {all_donor_outcomes_all_periods.shape[1]}."
                )

            # Donor selection
            (
                selected_donor_indices_for_fit,
                final_donor_selection_scores_all_donors,
            ) = self.select_donors(
                treated_outcome_series=treated_outcome_all_periods,
                all_donor_outcomes_matrix=all_donor_outcomes_all_periods,
                num_pre_treatment_periods=num_pre_treatment_periods,
                granger_alpha_param=self.granger_alpha,
                granger_maxlag_param=self.granger_maxlag,
                proximity_alpha_param=self.proximity_alpha,
                rbf_sigma_param=self.rbf_sigma_fit,
            )

            if selected_donor_indices_for_fit.size == 0:
                raise MlsynthDataError(
                    "No donors selected after applying Granger and proximity filters. Cannot proceed with SCM."
                )

            # --- SCM Optimization using selected and scored donors ---
            # Prepare data for SCM optimization: treated unit's pre-treatment outcomes
            # and donor outcomes weighted by the RBF scores (derived from proximity).
            treated_outcome_pre_treatment: np.ndarray = treated_outcome_all_periods[
                :num_pre_treatment_periods # Outcomes of the treated unit before treatment.
            ]
            num_all_donors: int = all_donor_outcomes_all_periods.shape[1] # Total number of original potential donors.
            
            # Weight the pre-treatment outcomes of ALL donors by their final selection scores.
            # Donors not selected by the hybrid criteria will have a score of 0, effectively excluding them
            # from contributing to the SCM optimization beyond their score's influence.
            score_weighted_donor_outcomes_pre_treatment: np.ndarray = (
                all_donor_outcomes_all_periods[:num_pre_treatment_periods] # Pre-treatment outcomes of all donors.
                * final_donor_selection_scores_all_donors # Apply RBF scores (many might be zero).
            )

            # Perform Synthetic Control Method (SCM) optimization using the Opt.SCopt utility.
            # This finds weights for the (scored) donor units to best match the treated unit's pre-treatment outcomes.
            # Opt.SCopt handles CVXPY errors internally or raises errors caught by the broader try-except.
            optimization_problem_result = Opt.SCopt(
                num_all_donors, # Pass the total number of original donors.
                treated_outcome_pre_treatment, # Target series for matching.
                num_pre_treatment_periods, # Length of the pre-treatment period.
                score_weighted_donor_outcomes_pre_treatment, # Scored donor outcomes for matching.
                scm_model_type=self.sc_model_type, # Type of SCM optimization (e.g., "OLS", "SIMPLEX").
            )
            
            # Verify that the optimization was successful.
            if optimization_problem_result.status not in ["optimal", "optimal_inaccurate"]:
                 raise MlsynthEstimationError(
                    f"SCM optimization failed or did not find an optimal solution. Status: {optimization_problem_result.status}"
                )

            # Extract the estimated donor weights from the optimization result.
            # These weights correspond to the (potentially RBF-scored) donor units.
            optimization_variable_key = list(
                optimization_problem_result.solution.primal_vars.keys() # Get the key for the weight variable.
            )[0]
            estimated_donor_weights_array: np.ndarray = (
                optimization_problem_result.solution.primal_vars[optimization_variable_key] # Array of weights.
            )

            # --- Counterfactual Calculation ---
            # Construct the counterfactual outcome for all periods by applying the estimated SCM weights
            # to the RBF-score-weighted outcomes of all donor units across all time periods.
            score_weighted_all_donor_outcomes_all_periods: np.ndarray = (
                all_donor_outcomes_all_periods * final_donor_selection_scores_all_donors # Apply RBF scores to all periods.
            )
            estimated_counterfactual_outcome: np.ndarray = ( # Weighted sum to get synthetic counterfactual.
                score_weighted_all_donor_outcomes_all_periods
                @ estimated_donor_weights_array
            )

            # --- Effects Calculation ---
            # Calculate treatment effects (ATT, etc.) and fit diagnostics using the observed and counterfactual outcomes.
            (
                raw_effects_dict, # Contains ATT, Percent ATT, etc.
                raw_fit_diagnostics_dict,
                raw_time_series_vectors_dict,
            ) = effects.calculate(
                prepared_panel_data["y"],
                estimated_counterfactual_outcome,
                num_pre_treatment_periods,
                num_post_periods,
            )
            
            # Ensure donor_names exists before formatting weights
            donor_names_list = prepared_panel_data.get("donor_names")
            # pd.Index (returned by dataprep for donor_names) supports len(), so isinstance check for list is not needed here.
            # The crucial part is that donor_names_list is not None and its length matches the weights array.
            if donor_names_list is None or len(donor_names_list) != len(estimated_donor_weights_array):
                raise MlsynthDataError("Donor names are missing or inconsistent with estimated weights.")

            formatted_donor_weights_dict: Dict[str, float] = {
                str(donor_names_list[i]): round(
                    estimated_donor_weights_array[i], 3
                )
                for i in range(len(estimated_donor_weights_array))
            }

            aggregated_raw_fit_output = {
                "Effects": raw_effects_dict,
                "Fit": raw_fit_diagnostics_dict,
                "Vectors": raw_time_series_vectors_dict,
                "Weights": formatted_donor_weights_dict,
                "_prepped": prepared_panel_data,
                "selected_donor_indices": selected_donor_indices_for_fit.tolist(),
                "anomaly_scores_used": final_donor_selection_scores_all_donors.tolist(),
            }
            
            # Plotting (wrapped separately)
            if self.display_graphs:
                try:
                    plot_estimates(
                        df=prepared_panel_data,
                        time=self.time,
                        unitid=self.unitid,
                        outcome=self.outcome,
                        treatmentname=self.treat,
                        treated_unit_name=prepared_panel_data["treated_unit_name"],
                        y=prepared_panel_data["y"],
                        cf_list=[estimated_counterfactual_outcome],
                        counterfactual_names=[_PLOT_COUNTERFACTUAL_NAME],
                        method=_PLOT_METHOD_NAME,
                        treatedcolor=self.treated_color,
                        counterfactualcolors=(
                            [self.counterfactual_color]
                            if isinstance(self.counterfactual_color, str)
                            else self.counterfactual_color
                        ),
                        save=self.save,
                    )
                except (MlsynthPlottingError, MlsynthDataError) as plot_err: # Catch specific plotting/data errors
                    warnings.warn(f"Plotting failed for StableSC due to: {plot_err}", UserWarning)
                except Exception as plot_err: # Catch any other unexpected plotting error
                    warnings.warn(f"An unexpected error occurred during StableSC plotting: {plot_err}", UserWarning)

            return self._create_estimator_results(aggregated_raw_fit_output)

        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError):
            # Re-raise custom errors from this module or underlying utilities
            raise
        except pydantic.ValidationError as e: # Should be caught by _create_estimator_results, but defensive
            raise MlsynthEstimationError(f"Results model validation error in StableSC fit: {e}") from e
        except (KeyError, ValueError, TypeError, IndexError, AttributeError, np.linalg.LinAlgError) as e:
            # Catch common Python errors and wrap them
            raise MlsynthEstimationError(f"A processing error occurred during StableSC fitting: {e}") from e
        except Exception as e:
            # Catch any other unexpected error
            raise MlsynthEstimationError(f"An unexpected error occurred during StableSC fitting: {e}") from e
