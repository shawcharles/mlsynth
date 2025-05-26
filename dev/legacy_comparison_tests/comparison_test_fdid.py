import pandas as pd
import numpy as np
import sys
import json
import os

# Ensure the main project directory and legacy directory are in the Python path
# Adjust the path as necessary if the script is moved
# Assuming the script is in dev/legacy_comparison_tests/
# and the project root is two levels up.
# The legacy project is assumed to be a subdirectory 'mlsynth-legacy' within the main project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
LEGACY_PROJECT_ROOT = os.path.join(PROJECT_ROOT, "mlsynth-legacy") # Path to the 'mlsynth-legacy' dir inside 'mlsynth-main'

# Store original sys.path to restore it later
original_sys_path = list(sys.path)

# Import CURRENT project modules
try:
    sys.path.insert(0, PROJECT_ROOT)
    from mlsynth.estimators.fdid import FDID
    from mlsynth.config_models import FDIDConfig
finally:
    # Restore original sys.path to prevent interference
    sys.path = list(original_sys_path) # Make a copy to avoid modifying original_sys_path if it's used again

# Import LEGACY project modules
# Cache and remove current mlsynth modules from sys.modules to ensure clean legacy import
cached_modules = {}
# Aggressively find all modules related to the current 'mlsynth' package that might be in the cache
modules_to_clear = [m for m in sys.modules if m == 'mlsynth' or m.startswith('mlsynth.')]

print(f"DEBUG: Modules to clear from sys.modules: {modules_to_clear}")
for mod_name in modules_to_clear:
    if mod_name in sys.modules:
        cached_modules[mod_name] = sys.modules[mod_name]
        del sys.modules[mod_name]

try:
    sys.path.insert(0, LEGACY_PROJECT_ROOT) # Add legacy path
    print(f"DEBUG: sys.path for legacy import (top entry): {sys.path[0]}")
    legacy_mlsynth_pkg_path = os.path.join(LEGACY_PROJECT_ROOT, 'mlsynth')
    legacy_mlsynth_module_file_path = os.path.join(legacy_mlsynth_pkg_path, 'mlsynth.py')
    legacy_mlsynth_init_file_path = os.path.join(legacy_mlsynth_pkg_path, '__init__.py')
    print(f"DEBUG: LEGACY_PROJECT_ROOT: {LEGACY_PROJECT_ROOT}")
    print(f"DEBUG: Expected legacy 'mlsynth' package path: {legacy_mlsynth_pkg_path}")
    print(f"DEBUG: Checking existence of legacy 'mlsynth' package dir: {os.path.isdir(legacy_mlsynth_pkg_path)}")
    print(f"DEBUG: Checking existence of legacy 'mlsynth.py' module file: {os.path.exists(legacy_mlsynth_module_file_path)}")
    print(f"DEBUG: Checking existence of legacy '__init__.py' in package: {os.path.exists(legacy_mlsynth_init_file_path)}")
    
    # Attempt to import a submodule from legacy to test package discovery
    try:
        import mlsynth.utils.datautils as legacy_datautils
        print("DEBUG: Successfully imported legacy mlsynth.utils.datautils as legacy_datautils")
    except ImportError as e_utils:
        print(f"DEBUG: Failed to import legacy mlsynth.utils.datautils: {e_utils}")

    # This assumes 'mlsynth.mlsynth' is the module structure in the legacy path
    # Standard import should work now that sys.modules is cleaned for 'mlsynth'
    from mlsynth.mlsynth import FDID as LegacyFDID
    print("DEBUG: Successfully imported LegacyFDID from mlsynth.mlsynth")
    
except ImportError as e:
    print(f"Error importing legacy FDID: {e}")
    print("Ensure LEGACY_PROJECT_ROOT is correctly set and the legacy mlsynth module is accessible.")
    # No need to restore sys.path here if exiting, but good practice if continuing
    sys.exit(1)
finally:
    # Restore original sys.path
    sys.path = list(original_sys_path)
    # Restore cached current modules
    for mod_name, mod_obj in cached_modules.items():
        sys.modules[mod_name] = mod_obj

def save_results(data, filename, output_dir="comparison_outputs"):
    """Saves dictionary data to a JSON file."""
    # Ensure output directory exists (relative to the script's location)
    abs_output_dir = os.path.join(SCRIPT_DIR, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    filepath = os.path.join(abs_output_dir, filename)

    data_to_process = {}
    if hasattr(data, 'model_dump'):  # Pydantic v2
        data_to_process = data.model_dump()
    elif hasattr(data, 'dict'):  # Pydantic v1
        data_to_process = data.dict()
    elif isinstance(data, dict):
        data_to_process = data # It's already a dict (legacy case)
    else:
        print(f"Warning: Unexpected data type for save_results: {type(data)}. Attempting to process as is, but might fail.")
        data_to_process = data # Fallback

    # Recursive function to convert numpy types and other non-serializable types
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
        # Add other type conversions if needed, e.g., for datetime objects
        return obj

    serializable_data = convert_types_for_json(data_to_process)

    with open(filepath, "w") as f:
        json.dump(serializable_data, f, indent=4, allow_nan=True)
    print(f"Results saved to {filepath}")

def compare_outputs(legacy_results_list, current_results_list):
    """Compares key metrics from legacy (list of dicts) and current (list of Pydantic models) FDID outputs."""
    summary = {}
    
    # Assuming legacy_results_list and current_results_list are ordered the same way:
    # e.g., [FDID_result, DID_result, AUGDID_result]
    
    for i, legacy_method_wrapper_dict in enumerate(legacy_results_list):
        if i >= len(current_results_list):
            print(f"Warning: Current results list shorter than legacy results list. Stopping comparison at index {i-1}.")
            break

        method_name = list(legacy_method_wrapper_dict.keys())[0]
        legacy_method_data = legacy_method_wrapper_dict[method_name] # This is the dict with "Effects", "Vectors" etc.
        
        current_method_pydantic_obj = current_results_list[i] # This is a BaseEstimatorResults object

        summary[method_name] = {}

        # --- ATT Comparison ---
        legacy_att = legacy_method_data.get("Effects", {}).get("ATT")
        
        current_effects_attr = getattr(current_method_pydantic_obj, 'effects', None)
        current_att = getattr(current_effects_attr, 'att', None) if current_effects_attr else None

        if legacy_att is not None and current_att is not None:
            summary[method_name]["ATT_match"] = np.isclose(legacy_att, current_att)
        else:
            summary[method_name]["ATT_match"] = (legacy_att is None and current_att is None)
        summary[method_name]["ATT_legacy"] = legacy_att
        summary[method_name]["ATT_current"] = current_att

        # --- Counterfactual Comparison ---
        legacy_cf_vector = legacy_method_data.get("Vectors", {}).get("Counterfactual")
        
        current_vectors_attr = getattr(current_method_pydantic_obj, 'vectors', None)
        current_cf_vector = getattr(current_vectors_attr, 'counterfactual', None) if current_vectors_attr else None

        if legacy_cf_vector is not None and current_cf_vector is not None:
            legacy_cf_flat = np.array(legacy_cf_vector).flatten()
            current_cf_flat = np.array(current_cf_vector).flatten()
            if legacy_cf_flat.shape == current_cf_flat.shape:
                summary[method_name]["Counterfactual_match"] = np.allclose(legacy_cf_flat, current_cf_flat, equal_nan=True)
            else:
                summary[method_name]["Counterfactual_match"] = False
                print(f"Warning: Counterfactual shapes differ for method {method_name}.")
        else:
            summary[method_name]["Counterfactual_match"] = (legacy_cf_vector is None and current_cf_vector is None)
            
        # --- Weights Comparison (Only for the main "FDID" method normally) ---
        if method_name == "FDID": # Legacy FDID stores weights directly under the "FDID" key
            legacy_weights_dict = legacy_method_data.get("Weights")
            
            current_weights_attr = getattr(current_method_pydantic_obj, 'weights', None)
            current_weights_dict = getattr(current_weights_attr, 'donor_weights', None) if current_weights_attr else None

            if legacy_weights_dict is not None and current_weights_dict is not None:
                legacy_w_sorted = sorted(legacy_weights_dict.items())
                current_w_sorted = sorted(current_weights_dict.items())
                
                if len(legacy_w_sorted) == len(current_w_sorted):
                    weights_match_val = all(
                        d_leg == d_curr and np.isclose(w_leg, w_curr)
                        for (d_leg, w_leg), (d_curr, w_curr) in zip(legacy_w_sorted, current_w_sorted)
                    )
                    summary[method_name]["Weights_match"] = weights_match_val
                else:
                    summary[method_name]["Weights_match"] = False
            elif legacy_weights_dict is None and current_weights_dict is None:
                 summary[method_name]["Weights_match"] = True
            else:
                summary[method_name]["Weights_match"] = False
    return summary

def print_comparison_summary(summary):
    print("\n--- Comparison Summary ---")
    numerical_differs = False
    for method, metrics in summary.items():
        print(f"\nComparing method: {method}")
        if "ATT_match" in metrics:
            print(f"  ATT match: {metrics['ATT_match']} (Legacy: {metrics.get('ATT_legacy', 'N/A'):.4f}, Current: {metrics.get('ATT_current', 'N/A'):.4f})")
            if not metrics['ATT_match']: numerical_differs = True
        if "Counterfactual_match" in metrics:
            print(f"  Counterfactual match: {metrics['Counterfactual_match']}")
            if not metrics['Counterfactual_match']: numerical_differs = True
        if "Weights_match" in metrics:
            print(f"  Weights match: {metrics['Weights_match']}")
            if not metrics['Weights_match']: numerical_differs = True
    
    if numerical_differs:
        print("\nWARNING: Some numerical outputs differ. Check saved JSON files for details.")
    else:
        print("\nAll key numerical outputs appear consistent.")

# --- Configuration ---
DATA_FILE = os.path.join(PROJECT_ROOT, "basedata", "basque_data.csv")
OUTCOME_VARIABLE = "gdpcap"
UNIT_ID_COLUMN_NAME = "regionname"
TIME_ID_COLUMN_NAME = "year"
TREATED_UNIT_ID = "Basque" # As suggested by the plot
TREATMENT_START_YEAR = 1975 # As suggested by the plot
# This will be the name of the column passed as 'treat' to the estimators
TREATMENT_INDICATOR_COLUMN = "treated_status_dummy" 

# --- Load and Prepare Data ---
df = pd.read_csv(DATA_FILE)

# Create the treatment dummy variable
df[TREATMENT_INDICATOR_COLUMN] = (
    (df[UNIT_ID_COLUMN_NAME] == TREATED_UNIT_ID) & 
    (df[TIME_ID_COLUMN_NAME] >= TREATMENT_START_YEAR)
).astype(int)


# --- Run Legacy FDID ---
print("--- Running Legacy FDID ---")
legacy_config_dict = {
    "df": df.copy(), # Use a copy to avoid modifications affecting the current run
    "unitid": UNIT_ID_COLUMN_NAME,
    "time": TIME_ID_COLUMN_NAME,
    "outcome": OUTCOME_VARIABLE,
    "treat": TREATMENT_INDICATOR_COLUMN, # Name of the 0/1 treatment status column
    "display_graphs": False # Don't show plots during automated test
}
legacy_fdid_estimator = LegacyFDID(config=legacy_config_dict)
legacy_fdid_results = legacy_fdid_estimator.fit() # This returns a list of dicts
save_results(legacy_fdid_results[0], "legacy_fdid_main_outputs.json") # FDID is the first
if len(legacy_fdid_results) > 1:
    save_results(legacy_fdid_results[1], "legacy_fdid_did_outputs.json")
if len(legacy_fdid_results) > 2:
    save_results(legacy_fdid_results[2], "legacy_fdid_augdid_outputs.json")


# --- Run Current FDID ---
print("\n--- Running Current FDID ---")
current_fdid_config = FDIDConfig(
    df=df.copy(), # Use a copy
    unitid=UNIT_ID_COLUMN_NAME,
    time=TIME_ID_COLUMN_NAME,
    outcome=OUTCOME_VARIABLE,
    treat=TREATMENT_INDICATOR_COLUMN, # Name of the 0/1 treatment status column
    display_graphs=False
)
current_fdid_estimator = FDID(config=current_fdid_config)
current_fdid_results = current_fdid_estimator.fit() # This returns a list of dicts
save_results(current_fdid_results[0], "current_fdid_main_outputs.json") # FDID is the first
if len(current_fdid_results) > 1:
    save_results(current_fdid_results[1], "current_fdid_did_outputs.json")
if len(current_fdid_results) > 2:
    save_results(current_fdid_results[2], "current_fdid_augdid_outputs.json")


# --- Compare Results ---
# The fit method for FDID returns a list: [FDID_results, DID_results, AUGDID_results]
# We need to compare them element-wise.
comparison_summary = compare_outputs(legacy_fdid_results, current_fdid_results)
print_comparison_summary(comparison_summary)

print("\nFDID Comparison Test Completed.")
