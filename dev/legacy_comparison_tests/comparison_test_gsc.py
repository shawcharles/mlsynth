import pandas as pd
import numpy as np
import sys
import json
import os

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
LEGACY_PROJECT_ROOT = os.path.join(PROJECT_ROOT, "mlsynth-legacy")

# Store original sys.path to restore it later
original_sys_path = list(sys.path)

# Import CURRENT project modules
current_gsc_estimator_class = None
current_gsc_config_class = None
try:
    sys.path.insert(0, PROJECT_ROOT)
    from mlsynth.estimators.gsc import GSC
    from mlsynth.config_models import GSCConfig
    current_gsc_estimator_class = GSC
    current_gsc_config_class = GSCConfig
finally:
    sys.path = list(original_sys_path)

if not current_gsc_estimator_class or not current_gsc_config_class:
    print("Failed to import current GSC modules. Exiting.")
    sys.exit(1)

# Import LEGACY project modules
legacy_gsc_estimator_class = None
cached_modules = {}
modules_to_clear = [m for m in sys.modules if m == 'mlsynth' or m.startswith('mlsynth.')]

for mod_name in modules_to_clear:
    if mod_name in sys.modules:
        cached_modules[mod_name] = sys.modules[mod_name]
        del sys.modules[mod_name]

try:
    sys.path.insert(0, LEGACY_PROJECT_ROOT)
    # Assuming legacy GSC is also in mlsynth.mlsynth
    from mlsynth.mlsynth import GSC as LegacyGSC
    legacy_gsc_estimator_class = LegacyGSC
except ImportError as e:
    print(f"Error importing legacy GSC: {e}")
    sys.exit(1)
finally:
    sys.path = list(original_sys_path)
    for mod_name, mod_obj in cached_modules.items():
        sys.modules[mod_name] = mod_obj

if not legacy_gsc_estimator_class:
    print("Failed to import legacy GSC class. Exiting.")
    sys.exit(1)

def save_results(data, filename, output_dir="comparison_outputs"):
    abs_output_dir = os.path.join(SCRIPT_DIR, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    filepath = os.path.join(abs_output_dir, filename)

    data_to_process = {}
    if hasattr(data, 'model_dump'): # For Pydantic models
        data_to_process = data.model_dump()
    elif hasattr(data, 'dict'): # For older Pydantic or similar
        data_to_process = data.dict()
    elif isinstance(data, dict):
        data_to_process = data
    else:
        print(f"Warning: Unexpected data type for save_results: {type(data)}.")
        data_to_process = data # Process as is, hope for the best

    # Handle _prepped key specifically if it contains DataFrames or other large non-serializable objects
    if isinstance(data_to_process, dict) and "_prepped" in data_to_process:
        data_to_process["_prepped"] = "Content of '_prepped' (potentially including DataFrames) removed for JSON serialization."

    def convert_types_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_types_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types_for_json(elem) for elem in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj): # Handle pandas NaT or other specific NaNs if necessary
            return None
        return obj

    serializable_data = convert_types_for_json(data_to_process)
    with open(filepath, "w") as f:
        json.dump(serializable_data, f, indent=4, allow_nan=True)
    print(f"Results saved to {filepath}")

def compare_gsc_outputs(legacy_res_dict, current_res_obj):
    summary = {"GSC": {}}
    method_name = "GSC"

    # ATT Comparison
    # Assuming legacy GSC output structure for ATT
    legacy_att = legacy_res_dict.get("Effects", {}).get("ATT")
    # Current GSC (BaseEstimatorResults) stores ATT in: results.effects.att
    current_effects_attr = getattr(current_res_obj, 'effects', None)
    current_att = getattr(current_effects_attr, 'att', None) if current_effects_attr else None

    if legacy_att is not None and current_att is not None:
        summary[method_name]["ATT_match"] = np.isclose(legacy_att, current_att)
    else:
        summary[method_name]["ATT_match"] = (legacy_att is None and current_att is None)
    summary[method_name]["ATT_legacy"] = legacy_att
    summary[method_name]["ATT_current"] = current_att

    # Counterfactual Comparison
    # Assuming legacy GSC output structure for Counterfactual
    legacy_cf_vector = legacy_res_dict.get("Vectors", {}).get("Counterfactual")
    # Current GSC (BaseEstimatorResults) stores counterfactual in: results.time_series.counterfactual_outcome
    current_time_series_attr = getattr(current_res_obj, 'time_series', None)
    current_cf_vector = getattr(current_time_series_attr, 'counterfactual_outcome', None) if current_time_series_attr else None
    
    # Fallback logic below is likely not needed if the above is correct, but kept for safety.
    # It should ideally check a structured part of the Pydantic model if the primary path fails,
    # e.g., method_details.additional_outputs if that's where such data might alternatively reside.
    # The original fallback to 'vectors' on the top-level object is incorrect for BaseEstimatorResults.
    if current_cf_vector is None:
        # Example of a more plausible fallback (though ideally not needed):
        # current_method_details = getattr(current_res_obj, 'method_details', None)
        # current_additional_outputs = getattr(current_method_details, 'additional_outputs', None) if current_method_details else None
        # current_cf_vector = current_additional_outputs.get("Counterfactual_TreatedUnit_Series") if isinstance(current_additional_outputs, dict) else None
        # For now, if the primary access fails, it will remain None, which is the correct behavior if data isn't there.
        pass


    if legacy_cf_vector is not None and current_cf_vector is not None:
        legacy_cf_flat = np.array(legacy_cf_vector).flatten()
        current_cf_flat = np.array(current_cf_vector).flatten()
        if legacy_cf_flat.shape == current_cf_flat.shape:
            summary[method_name]["Counterfactual_match"] = np.allclose(legacy_cf_flat, current_cf_flat, equal_nan=True)
        else:
            summary[method_name]["Counterfactual_match"] = False
            print(f"Warning: Counterfactual shapes differ for GSC. Legacy: {legacy_cf_flat.shape}, Current: {current_cf_flat.shape}")
    elif legacy_cf_vector is None and current_cf_vector is None:
        summary[method_name]["Counterfactual_match"] = True # Both are None
    else:
        summary[method_name]["Counterfactual_match"] = False # One is None and the other isn't
        print(f"Warning: One of the counterfactuals is None. Legacy: {type(legacy_cf_vector)}, Current: {type(current_cf_vector)}")

    # GSC does not typically produce simple donor weights like SCM methods.
    # If there are other specific GSC outputs to compare (e.g., feature weights), add them here.
    # For now, no weight comparison.

    return summary

def print_comparison_summary(summary):
    print("\n--- Comparison Summary ---")
    numerical_differs = False
    for method, metrics in summary.items():
        print(f"\nComparing method: {method}")
        if "ATT_match" in metrics:
            att_legacy = metrics.get('ATT_legacy', 'N/A')
            att_current = metrics.get('ATT_current', 'N/A')
            att_legacy_str = f"{att_legacy:.4f}" if isinstance(att_legacy, (float, np.floating)) else str(att_legacy)
            att_current_str = f"{att_current:.4f}" if isinstance(att_current, (float, np.floating)) else str(att_current)
            print(f"  ATT match: {metrics['ATT_match']} (Legacy: {att_legacy_str}, Current: {att_current_str})")
            if not metrics['ATT_match'] and (att_legacy is not None or att_current is not None) : numerical_differs = True
        if "Counterfactual_match" in metrics:
            print(f"  Counterfactual match: {metrics['Counterfactual_match']}")
            if not metrics['Counterfactual_match']: numerical_differs = True
        # No "Weights_match" for GSC

    if numerical_differs:
        print("\nWARNING: Some numerical outputs differ. Check saved JSON files for details.")
    else:
        print("\nAll key numerical outputs (ATT, Counterfactual) appear consistent.")

# --- Configuration for GSC Test ---
# Using smoking_data.csv as suggested for consistency
DATA_FILE = os.path.join(PROJECT_ROOT, "basedata", "smoking_data.csv")
OUTCOME_VARIABLE = "cigsale"
UNIT_ID_COLUMN_NAME = "state"
TIME_ID_COLUMN_NAME = "year"
TREATED_UNIT_ID = "California" # Example treated unit
TREATMENT_START_YEAR = 1989    # Example treatment start year
TREATMENT_INDICATOR_COLUMN = "treated_dummy" # This will be created

# --- Load and Prepare Data ---
df = pd.read_csv(DATA_FILE)
# Ensure correct data types for IDs and time
df[UNIT_ID_COLUMN_NAME] = df[UNIT_ID_COLUMN_NAME].astype(str)
df[TIME_ID_COLUMN_NAME] = df[TIME_ID_COLUMN_NAME].astype(int)
df[OUTCOME_VARIABLE] = pd.to_numeric(df[OUTCOME_VARIABLE], errors='coerce')

# Create treatment indicator column
df[TREATMENT_INDICATOR_COLUMN] = (
    (df[UNIT_ID_COLUMN_NAME] == str(TREATED_UNIT_ID)) &
    (df[TIME_ID_COLUMN_NAME] >= TREATMENT_START_YEAR)
).astype(int)

# GSC can run without explicit features, using lagged outcomes.
# We will not specify external predictors for this basic comparison.
df = df.dropna(subset=[OUTCOME_VARIABLE])


# --- Run Legacy GSC ---
print("--- Running Legacy GSC ---")
legacy_config_dict = {
    "df": df.copy(),
    "unitid": UNIT_ID_COLUMN_NAME,
    "time": TIME_ID_COLUMN_NAME,
    "outcome": OUTCOME_VARIABLE,
    "treat": TREATMENT_INDICATOR_COLUMN,
    # "features": PREDICTORS, # Legacy GSC might take 'features' - not providing for now
    "display_graphs": False,
    # Add any other GSC specific legacy parameters if known, e.g.
    # "degree": 1, "alpha": 0.05 etc.
    # For now, assuming basic config.
}
legacy_gsc_estimator = legacy_gsc_estimator_class(config=legacy_config_dict)
legacy_gsc_results = None
try:
    legacy_gsc_results = legacy_gsc_estimator.fit() # Returns a dict
    save_results(legacy_gsc_results, "legacy_gsc_outputs.json")
except NameError as e:
    print(f"!!! Legacy GSC failed to run due to an internal NameError: {e}")
    print("!!! This is likely a bug in the legacy GSC code (mlsynth-legacy/mlsynth/mlsynth.py).")
    legacy_gsc_results = {"error": f"Legacy GSC failed: {e}"}
    # Save a placeholder error JSON for legacy results
    save_results(legacy_gsc_results, "legacy_gsc_outputs.json")
except Exception as e:
    print(f"!!! Legacy GSC failed to run due to an unexpected error: {e}")
    legacy_gsc_results = {"error": f"Legacy GSC failed with unexpected error: {e}"}
    save_results(legacy_gsc_results, "legacy_gsc_outputs.json")


# --- Run Current GSC ---
print("\n--- Running Current GSC ---")
current_gsc_config = current_gsc_config_class(
    df=df.copy(),
    unitid=UNIT_ID_COLUMN_NAME,
    time=TIME_ID_COLUMN_NAME,
    outcome=OUTCOME_VARIABLE,
    treat=TREATMENT_INDICATOR_COLUMN,
    # features=PREDICTORS, # Current GSCConfig also takes 'features' - not providing for now
    display_graphs=False,
    # Add any other GSC specific current parameters if needed
    # e.g. degree=1, alpha=0.05 from GSCConfig defaults or explicit setting
)
current_gsc_estimator = current_gsc_estimator_class(config=current_gsc_config)
current_gsc_results = current_gsc_estimator.fit() # Returns a BaseEstimatorResults object
save_results(current_gsc_results, "current_gsc_outputs.json")

# --- Compare Results ---
if legacy_gsc_results and "error" not in legacy_gsc_results:
    comparison_summary = compare_gsc_outputs(legacy_gsc_results, current_gsc_results)
    print_comparison_summary(comparison_summary)
else:
    print("\n--- Comparison Summary ---")
    print("Skipping comparison due to legacy GSC failure.")
    print("Current GSC execution results saved. Legacy GSC failed to produce results.")
    # Optionally, print some info about current GSC results if needed
    if current_gsc_results:
        current_effects_attr = getattr(current_gsc_results, 'effects', None)
        current_att = getattr(current_effects_attr, 'att', None) if current_effects_attr else None
        print(f"  Current GSC ATT: {current_att if current_att is not None else 'N/A'}")


print("\nGSC Comparison Test Completed.")
