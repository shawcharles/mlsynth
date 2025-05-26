import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from pydantic import ValidationError # For catching Pydantic errors if models are created internally

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import plot_estimates
from ..utils.denoiseutils import DC_PR_with_suggested_rank
from ..exceptions import (
    MlsynthDataError,
    MlsynthConfigError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..config_models import (
    GSCConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    InferenceResults,
    MethodDetailsResults,
)


class GSC:
    """
    Generalized Synthetic Control (GSC) method.

    Implements the Generalized Synthetic Control method, often associated with
    denoising matrix completion techniques, for estimating treatment effects in
    panel data settings. This version is guided by the approach described in
    Costa et al. (2023), which utilizes a denoising algorithm
    (`DC_PR_with_suggested_rank`) to estimate the counterfactual outcomes.

    The estimator takes panel data and a configuration object. The `fit` method
    prepares the data, applies the denoising procedure to the wide-format outcome
    matrix, and then calculates treatment effects by comparing observed outcomes
    to the denoised (counterfactual) estimates for the treated unit.

    Attributes
    ----------
    config : GSCConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `GSCConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `GSCConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `GSCConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `GSCConfig`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `GSCConfig`)
    display_graphs : bool, default True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `GSCConfig`)
    save : Union[bool, str], default False
        If False, plots are not saved. If True, plots are saved with default names.
        If a string, it's used as the directory path to save plots.
    counterfactual_color : str, default "red"
        Color for the counterfactual line in plots.
    treated_color : str, default "black"
        Color for the treated unit line in plots.
    denoising_method : str, default "non-convex"
        Method for the denoising algorithm: 'auto', 'convex', or 'non-convex'.
    target_rank : Optional[int], default None
        Optional user-specified rank for the denoising algorithm. If None,
        rank is estimated internally.

    References
    ----------
    Costa, L., Farias, V. F., Foncea, P., Gan, J. (D.), Garg, A., Montenegro, I. R.,
    Pathak, K., Peng, T., & Popovic, D. (2023). "Generalized Synthetic Control
    for TestOps at ABI: Models, Algorithms, and Infrastructure."
    *INFORMS Journal on Applied Analytics* 53(5):336-349.

    Examples
    --------
    >>> from mlsynth import GSC
    >>> from mlsynth.config_models import GSCConfig
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample data for demonstration
    >>> data = pd.DataFrame({
    ...     'unit': np.repeat(np.arange(1, 4), 10), # 3 units
    ...     'time': np.tile(np.arange(1, 11), 3),   # 10 time periods
    ...     'outcome': np.random.rand(30) + np.repeat(np.arange(0,3),10)*0.5,
    ...     'treated_unit_1': ((np.repeat(np.arange(1, 4), 10) == 1) & \
    ...                        (np.tile(np.arange(1, 11), 3) >= 6)).astype(int)
    ... })
    >>> gsc_config = GSCConfig(
    ...     df=data,
    ...     outcome='outcome',
    ...     treat='treated_unit_1',
    ...     unitid='unit',
    ...     time='time',
    ...     display_graphs=False # Typically True, False for non-interactive examples
    ... )
    >>> estimator = GSC(config=gsc_config)
    >>> # Results can be obtained by calling estimator.fit()
    >>> # results = estimator.fit() # doctest: +SKIP
    """

    def __init__(self, config: GSCConfig) -> None: # Changed to GSCConfig
        """
        Initializes the GSC estimator with a configuration object.

        Parameters
        ----------
        config : GSCConfig
            A Pydantic model instance containing all configuration parameters
            for the GSC estimator. `GSCConfig` inherits from `BaseEstimatorConfig`.
            The fields include:

                df : pd.DataFrame
                    The input panel data. Must contain columns for outcome, treatment
                    indicator, unit identifier, and time identifier.
                outcome : str
                    Name of the outcome variable column in `df`.
                treat : str
                    Name of the binary treatment indicator column in `df`.
                unitid : str
                    Name of the unit identifier (e.g., country, individual ID) column in `df`.
                time : str
                    Name of the time period column in `df`.
                display_graphs : bool, default=True
                    Whether to display plots of the results after fitting.
                save : Union[bool, str], default=False
                    If False, plots are not saved. If True, plots are saved with default names.
                    If a string, it's used as a prefix for saved plot filenames.
                counterfactual_color : str, default="red"
                    Color for the counterfactual line(s) in plots.
                treated_color : str, default="black"
                    Color for the treated unit line in plots.
                denoising_method : str, default="non-convex"
                    Method for the denoising algorithm ('auto', 'convex', 'non-convex').
                target_rank : Optional[int], default=None
                    Optional user-specified rank for denoising. If None, estimated internally.
        """
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save # Updated type to match BaseEstimatorConfig

    def _create_estimator_results( # Helper method to package GSC results into the standard Pydantic model
        self, raw_results: Dict[str, Any], prepared_data: Dict[str, Any]
    ) -> BaseEstimatorResults:
        """
        Constructs a BaseEstimatorResults object from raw GSC outputs.

        Parameters
        ----------
        raw_results : Dict[str, Any]
            Dictionary containing the raw results from the `DC_PR_with_suggested_rank`
            function, typically including 'Effects', 'Fit', 'Vectors', and 'Inference' keys.
        prepared_data : Dict[str, Any]
            Dictionary of preprocessed data from `dataprep`, containing elements
            like 'y' (treated unit outcomes), 'time_labels', and 'Ywide'.

        Returns
        -------
        BaseEstimatorResults
            A Pydantic model instance containing the standardized estimation results.
        """
        # Extract raw data components from the main GSC output dictionary
        effects_raw = raw_results.get("Effects", {}) # Raw ATT, Percent ATT
        fit_raw = raw_results.get("Fit", {})         # Raw RMSE, R-squared
        vectors_raw = raw_results.get("Vectors", {}) # Raw time series (Counterfactual, Loadings, Factors)
        inference_raw = raw_results.get("Inference", {}) # Raw p-value, CI, SE, t-stat

        # Prepare observed outcome: ensure it's a 1D NumPy array
        observed_outcome_series = prepared_data.get("y") # Treated unit's outcome series from dataprep
        if observed_outcome_series is not None and isinstance(observed_outcome_series, pd.Series):
            observed_outcome_arr = observed_outcome_series.to_numpy().flatten()
        elif isinstance(observed_outcome_series, np.ndarray):
            observed_outcome_arr = observed_outcome_series.flatten()
        else:
            observed_outcome_arr = None # Handle case where 'y' might be missing or not in expected format
        
        # Prepare counterfactual outcome: ensure it's a 1D NumPy array
        # The `DC_PR_with_suggested_rank` function returns 'Counterfactual' which is (T,1) for the treated unit.
        counterfactual_outcome_vector = vectors_raw.get("Counterfactual") 
        if counterfactual_outcome_vector is not None:
            counterfactual_outcome_arr = counterfactual_outcome_vector.flatten() # Ensure 1D
        else:
            counterfactual_outcome_arr = None

        # Calculate estimated gap (treatment effect over time)
        gap_arr: Optional[np.ndarray] = None
        if observed_outcome_arr is not None and counterfactual_outcome_arr is not None:
            if len(observed_outcome_arr) == len(counterfactual_outcome_arr): # Check for consistent lengths
                gap_arr = observed_outcome_arr - counterfactual_outcome_arr
            else:
                # If lengths mismatch, gap cannot be reliably calculated.
                # This might indicate an issue upstream; for now, gap remains None.
                pass # Consider logging a warning here if such a mismatch occurs.

        # Create EffectsResults Pydantic model
        effects = EffectsResults(
            att=effects_raw.get("ATT"),
            att_percent=effects_raw.get("Percent ATT"),
            # Store standard error of ATT in additional_effects if available from inference_raw
            additional_effects={"att_std_err": inference_raw.get("SE")} if inference_raw.get("SE") is not None else None
        )

        # Create FitDiagnosticsResults Pydantic model
        fit_diagnostics = FitDiagnosticsResults(
            pre_treatment_rmse=fit_raw.get("T0 RMSE"), # RMSE for pre-treatment period
            pre_treatment_r_squared=fit_raw.get("R-Squared"), # R-squared for pre-treatment period
            post_treatment_rmse=fit_raw.get("T1 RMSE") # RMSE for post-treatment period (often called T1 RMSE)
        )
        
        # Prepare time periods array
        time_periods_series = prepared_data.get("time_labels") # Time labels from dataprep
        time_periods_arr: Optional[np.ndarray] = None
        if time_periods_series is not None and isinstance(time_periods_series, (pd.Series, np.ndarray, pd.Index)):
            time_periods_arr = np.array(time_periods_series) # Ensure it's a NumPy array

        # Create TimeSeriesResults Pydantic model
        time_series = TimeSeriesResults(
            observed_outcome=observed_outcome_arr,
            counterfactual_outcome=counterfactual_outcome_arr,
            estimated_gap=gap_arr,
            time_periods=time_periods_arr,
        )

        # Create InferenceResults Pydantic model
        inference = InferenceResults(
            p_value=inference_raw.get("p_value"),
            ci_lower_bound=inference_raw.get("Lower Bound"), # Lower bound of confidence interval
            ci_upper_bound=inference_raw.get("Upper Bound"), # Upper bound of confidence interval
            standard_error=inference_raw.get("SE"), # Standard error of the ATT
            details={"t_statistic": inference_raw.get("tstat")} if inference_raw.get("tstat") is not None else None,
            # `confidence_level` and `method` could be added if known (e.g., 0.95 if CI is 95%, method="asymptotic")
        )
        
        # Prepare additional outputs for MethodDetailsResults
        # Observed matrix: full Y_wide from dataprep (time x units)
        observed_matrix_df: Optional[pd.DataFrame] = prepared_data.get("Ywide") 
        observed_matrix_np: Optional[np.ndarray] = None
        if observed_matrix_df is not None:
            observed_matrix_np = observed_matrix_df.to_numpy() # Convert to NumPy array (time x units)

        # Counterfactual matrix: full denoised matrix from GSC (units x time), needs transpose
        counterfactual_full_matrix_raw = vectors_raw.get("Counterfactual_Full_Matrix") # (units x time)
        counterfactual_full_matrix_transposed: Optional[np.ndarray] = None
        if counterfactual_full_matrix_raw is not None:
            if not isinstance(counterfactual_full_matrix_raw, np.ndarray) or counterfactual_full_matrix_raw.ndim != 2:
                # This check ensures the raw matrix is as expected before transposing.
                raise MlsynthEstimationError("Raw 'Counterfactual_Full_Matrix' is not a 2D NumPy array.")
            counterfactual_full_matrix_transposed = counterfactual_full_matrix_raw.T # Transpose to (time x units)

        # Create MethodDetailsResults Pydantic model
        method_details = MethodDetailsResults(
            name="GSC", # Method name
            parameters_used={ # Store key parameters used in the GSC estimation
                "rank_used": raw_results.get("rank_used", vectors_raw.get("rank_used")), # Rank used by the denoiser
                "denoising_method_used": self.config.denoising_method, # Denoising method from config
            },
            additional_outputs={ # Store other relevant outputs from the GSC process
                "loadings_U": vectors_raw.get("Loadings"), # Unit loadings (U matrix)
                "factors_V_T": vectors_raw.get("Factors"),  # Time factors (V.T matrix)
                "counterfactual_all_units_matrix": counterfactual_full_matrix_transposed, # Full denoised matrix (time x units)
                "observed_all_units_matrix": observed_matrix_np, # Full observed outcome matrix (time x units)
            }
        )

        # Assemble the final BaseEstimatorResults object
        return BaseEstimatorResults(
            effects=effects,
            fit_diagnostics=fit_diagnostics,
            time_series=time_series,
            inference=inference,
            method_details=method_details,
            raw_results=raw_results,
            # No explicit weights like SCM, so weights field is None by default
        )

    def fit(self) -> BaseEstimatorResults: # Main method to fit the GSC estimator
        """
        Fits the Generalized Synthetic Control (GSC) model to the provided data.

        This method first balances the panel data and prepares it into a wide
        format suitable for matrix operations. It then applies the
        `DC_PR_with_suggested_rank` denoising algorithm to the matrix of
        outcomes. This algorithm estimates a low-rank representation of the data,
        which is used to construct counterfactual outcomes for all units, including
        the treated unit. Treatment effects are then derived by comparing the
        observed outcomes of the treated unit to its estimated counterfactual.
        The results, including effects, diagnostics, time series, and inference
        details, are packaged into a `BaseEstimatorResults` object.

        Returns
        -------
        BaseEstimatorResults
            An object containing the standardized estimation results. Key fields include:

            - `effects` (EffectsResults)
                Contains `att` (Average Treatment Effect on the Treated),
                `att_percent` (Percentage ATT), and `additional_effects`
                (e.g., for `att_std_err`).
            - `fit_diagnostics` (FitDiagnosticsResults)
                Contains `pre_treatment_rmse`, `pre_treatment_r_squared`,
                and `additional_metrics` (e.g., for `post_treatment_rmse`).
            - `time_series` (TimeSeriesResults)
                Contains `observed_outcome` (for the treated unit),
                `counterfactual_outcome`, `estimated_gap` (effect over time),
                and `time_periods` (actual time values or event time indices).
            - `inference` (InferenceResults)
                May contain `p_value`, `ci_lower_bound`, `ci_upper_bound`,
                `standard_error` of the ATT, and `details` (e.g., t-statistic),
                depending on the output of the underlying denoising algorithm.
            - `method_details` (MethodDetailsResults)
                Contains the method `name` ("GSC"), `parameters_used` (e.g.,
                `rank_used`), and `additional_outputs` (e.g., factor loadings U,
                time factors V.T, full counterfactual matrix, full observed
                outcome matrix).
            - `weights` (WeightsResults)
                Typically `None` for GSC as it does not produce explicit donor
                weights in the same way as SCM.
            - `raw_results` (Dict[str, Any])
                The raw dictionary of results from the
                `DC_PR_with_suggested_rank` function.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from mlsynth.estimators.gsc import GSC
        >>> from mlsynth.config_models import GSCConfig
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     'id': np.repeat(np.arange(1, 6), 10), # 5 units, 10 time periods each
        ...     'year': np.tile(np.arange(2000, 2010), 5),
        ...     'value': np.random.rand(50) + np.repeat(np.arange(1,6),10)*0.5 + \
        ...              np.tile(np.arange(2000,2010),5)*0.01,
        ...     'is_treated': ((np.repeat(np.arange(1, 6), 10) == 1) & \
        ...                    (np.tile(np.arange(2000, 2010), 5) >= 2005)).astype(int)
        ... }) # Unit 1 treated from 2005 onwards
        >>> gsc_config = GSCConfig(
        ...     df=data,
        ...     outcome="value",
        ...     treat="is_treated",
        ...     unitid="id",
        ...     time="year",
        ...     display_graphs=False # Disable plots for example
        ... )
        >>> gsc_estimator = GSC(config=gsc_config)
        >>> results = gsc_estimator.fit() # doctest: +SKIP
        >>> # Example: Accessing results (actual values will vary due to random data)
        >>> print(f"Estimated ATT: {results.effects.att}") # doctest: +SKIP
        >>> if results.method_details and results.method_details.parameters_used: # doctest: +SKIP
        ...     print(f"Rank used: {results.method_details.parameters_used.get('rank_used')}") # doctest: +SKIP
        """
        # Initialize variables that might be defined in the try block and used later for plotting or return.
        pydantic_results_obj: Optional[BaseEstimatorResults] = None # Will hold the final Pydantic results object.
        prepared_data_for_plotting: Optional[Dict[str, Any]] = None # Stores output from dataprep for plotting.
        denoiser_results_for_plotting: Optional[Dict[str, Any]] = None # Stores output from the denoiser for plotting.

        try:
            # Step 1: Validate data balance (ensures each unit has the same time periods).
            balance(self.df, self.unitid, self.time) # Can raise MlsynthDataError.

            # Step 2: Prepare data using the dataprep utility.
            # This separates data into treated/control, pre/post periods, and creates wide-format matrices.
            prepared_data = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            ) # Can raise MlsynthDataError or MlsynthConfigError.
            prepared_data_for_plotting = prepared_data # Save for potential use in plotting later.

            # Step 3: Perform essential checks on the output of dataprep.
            # Ensure required keys are present and that 'Ywide' (wide outcome matrix) and 'post_periods' are valid.
            required_keys_dataprep = ["Ywide", "treated_unit_name", "post_periods", "pre_periods", "donor_matrix", "y", "time_labels"]
            for key in required_keys_dataprep:
                if key not in prepared_data or prepared_data[key] is None:
                    raise MlsynthEstimationError(f"Essential key '{key}' missing or None in dataprep output.")
            if not isinstance(prepared_data["Ywide"], pd.DataFrame): # Ywide should be a DataFrame.
                raise MlsynthEstimationError("'Ywide' from dataprep is not a DataFrame.")
            if not isinstance(prepared_data["post_periods"], int) or prepared_data["post_periods"] < 0: # post_periods should be a non-negative integer.
                 raise MlsynthEstimationError(f"Invalid 'post_periods' ({prepared_data['post_periods']}) from dataprep.")

            # Transpose Ywide to (units x time) format expected by the denoiser.
            wide_outcome_df: pd.DataFrame = prepared_data["Ywide"].T 
            wide_outcome_np: np.ndarray = wide_outcome_df.to_numpy() # Convert to NumPy array.

            # Get the index of the treated unit in the wide-format DataFrame.
            if prepared_data["treated_unit_name"] not in wide_outcome_df.index:
                raise MlsynthDataError(f"Treated unit '{prepared_data['treated_unit_name']}' not found in Ywide index after dataprep.")
            treated_unit_idx_in_wide_df: int = wide_outcome_df.index.get_loc(prepared_data["treated_unit_name"])

            # Step 4: Create the treatment indicator matrix (Omega_t in some literature).
            # This matrix has the same shape as `wide_outcome_np` and is 1 for treated unit in post-periods, 0 otherwise.
            treatment_indicator_matrix: np.ndarray = np.zeros_like(wide_outcome_np)
            if prepared_data["post_periods"] > 0 : # Only set treatment indicators if there are post-treatment periods.
                treatment_indicator_matrix[treated_unit_idx_in_wide_df, -prepared_data["post_periods"]:] = 1

            # Step 5: Determine the rank to use for the denoising algorithm.
            # If `target_rank` is specified in config, use it. Otherwise, estimate a rank.
            rank_to_use: int
            if self.config.target_rank is not None:
                rank_to_use = self.config.target_rank
            else:
                # Estimate rank based on pre-treatment donor data if available.
                # This is a heuristic; more sophisticated rank selection might be needed for robustness.
                donor_matrix_for_rank = prepared_data.get("donor_matrix")
                pre_periods_for_rank = prepared_data.get("pre_periods")
                if isinstance(donor_matrix_for_rank, np.ndarray) and isinstance(pre_periods_for_rank, int) and pre_periods_for_rank > 0:
                    pre_treatment_donor_matrix = donor_matrix_for_rank[:pre_periods_for_rank]
                    if pre_treatment_donor_matrix.ndim == 2 and pre_treatment_donor_matrix.shape[0] > 0 and pre_treatment_donor_matrix.shape[1] > 0:
                        # A simple heuristic for rank: min(N0, T0)/2 - 1, ensuring it's at least 1.
                        num_donors_pre_treatment, num_periods_pre_treatment_donors = pre_treatment_donor_matrix.shape
                        rank_calculation_k: int = (min(num_donors_pre_treatment, num_periods_pre_treatment_donors) - 1) // 2
                        rank_to_use = rank_calculation_k if rank_calculation_k > 0 else 1
                    else: # Fallback if pre-treatment donor matrix is degenerate.
                        rank_to_use = 1
                else: # Fallback if donor matrix or pre_periods are not suitable for rank estimation.
                    rank_to_use = 1

            # Step 6: Apply the denoising algorithm (`DC_PR_with_suggested_rank`).
            # This function estimates the counterfactual outcomes based on the specified rank and method.
            denoiser_results = DC_PR_with_suggested_rank(
                wide_outcome_np,            # Observed outcome matrix (units x time)
                treatment_indicator_matrix, # Treatment indicator matrix (units x time)
                target_rank=rank_to_use,    # Suggested rank for denoising
                method=self.config.denoising_method # Denoising method ('auto', 'convex', 'non-convex')
            )
            denoiser_results_for_plotting = denoiser_results # Save for potential use in plotting.
        
            # Ensure 'rank_used' is in the results, falling back to `rank_to_use` if not provided by denoiser.
            if "rank_used" not in denoiser_results.get("Vectors", {}) and "rank_used" not in denoiser_results:
                 denoiser_results["rank_used"] = rank_to_use 

            # Step 7: Package the results into the standardized Pydantic model.
            pydantic_results_obj = self._create_estimator_results(denoiser_results, prepared_data)
        
        # Step 8: Handle specific and general exceptions during the fitting process.
        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError) as e: # Propagate custom Mlsynth errors.
            raise e
        except KeyError as e: # Handle errors due to missing keys in data structures.
            raise MlsynthEstimationError(f"Missing expected key in data structures during GSC fit: {e}") from e
        except ValueError as e: # Catch other ValueErrors (e.g., from matrix operations if shapes mismatch).
            raise MlsynthEstimationError(f"ValueError during GSC estimation: {e}") from e
        except Exception as e: # Catch-all for any other unexpected errors.
            raise MlsynthEstimationError(f"An unexpected error occurred during GSC fitting: {e}") from e

        # Ensure a results object was created. This should not be reached if errors above are properly raised.
        if pydantic_results_obj is None: 
            raise MlsynthEstimationError("GSC estimation failed to produce results object.")

        # Step 9: Display graphs if requested by the user configuration.
        if self.display_graphs:
            # Check if necessary data for plotting is available.
            if prepared_data_for_plotting is not None and denoiser_results_for_plotting is not None:
                try:
                    # Determine colors for plotting.
                    plot_counterfactual_colors: List[str] = (
                        [self.counterfactual_color]
                        if isinstance(self.counterfactual_color, str)
                        else self.counterfactual_color
                    )
                    
                    # Prepare observed outcome series for plotting.
                    y_for_plot_raw = prepared_data_for_plotting.get("y")
                    y_for_plot: Optional[pd.Series] = None
                    if isinstance(y_for_plot_raw, pd.Series):
                        y_for_plot = y_for_plot_raw
                    elif isinstance(y_for_plot_raw, np.ndarray): # If 'y' is NumPy array, try to create Series with time labels.
                        time_labels_for_plot = prepared_data_for_plotting.get("time_labels")
                        if time_labels_for_plot is not None and len(time_labels_for_plot) == len(y_for_plot_raw):
                            y_for_plot = pd.Series(y_for_plot_raw.flatten(), index=time_labels_for_plot, name=self.outcome)
                        else:
                            print("Warning: Could not create Series for y_for_plot due to missing/mismatched time_labels for plotting.")
                    else:
                        print(f"Warning: Outcome data 'y' for plotting is not a Series or NumPy array: {type(y_for_plot_raw)}.")

                    # Get counterfactual outcome series from denoiser results.
                    counterfactual_for_plot_raw = denoiser_results_for_plotting.get("Vectors", {}).get("Counterfactual")
                    
                    # Proceed with plotting if observed and counterfactual data are valid.
                    if y_for_plot is not None and counterfactual_for_plot_raw is not None and isinstance(counterfactual_for_plot_raw, np.ndarray):
                        plot_estimates(
                            df=prepared_data_for_plotting, # Data from dataprep
                            time=self.time, # Time column name
                            unitid=self.unitid, # Unit ID column name
                            outcome=self.outcome, # Outcome column name
                            treatmentname=self.treat, # Treatment indicator column name
                            treated_unit_name=prepared_data_for_plotting["treated_unit_name"], # Name of the treated unit
                            y=y_for_plot, # Observed outcome series for the treated unit
                            cf_list=[counterfactual_for_plot_raw], # List containing the counterfactual series
                            counterfactual_names=["GSC"], # Name for the counterfactual series in the plot
                            method="GSC", # Method name for plot title
                            treatedcolor=self.treated_color, # Color for the treated unit's line
                            counterfactualcolors=plot_counterfactual_colors, # Color(s) for counterfactual line(s)
                            save_path=self.save if isinstance(self.save, str) else None, # Path to save plots, if specified
                        )
                    else:
                        print("Warning: Skipping plotting due to missing or invalid y_for_plot or counterfactual_for_plot_raw.")

                except MlsynthPlottingError as e: # Handle specific plotting errors.
                    print(f"Warning: Plotting failed with MlsynthPlottingError: {e}")
                except MlsynthDataError as e: # Handle data-related errors during plotting.
                    print(f"Warning: Plotting failed due to data issues: {e}")
                except Exception as e: # Catch-all for other unexpected plotting errors.
                    print(f"Warning: An unexpected error occurred during plotting: {e}")
            else: # If essential data for plotting is missing.
                 print("Warning: Skipping plotting because essential data from estimation (prepared_data or denoiser_results) is missing.")
            
        return pydantic_results_obj
