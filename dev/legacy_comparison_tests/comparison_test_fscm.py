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
current_fscm_estimator_class = None
current_fscm_config_class = None
try:
    sys.path.insert(0, PROJECT_ROOT)
    from mlsynth.estimators.fscm import FSCM
    from mlsynth.config_models import FSCMConfig
    current_fscm_estimator_class = FSCM
    current_fscm_config_class = FSCMConfig
finally:
    sys.path = list(original_sys_path)

if not current_fscm_estimator_class or not current_fscm_config_class:
    print("Failed to import current FSCM modules. Exiting.")
    sys.exit(1)

# Import LEGACY project modules
legacy_fscm_estimator_class = None
cached_modules = {}
modules_to_clear = [m for m in sys.modules if m == 'mlsynth' or m.startswith('mlsynth.')]

for mod_name in modules_to_clear:
    if mod_name in sys.modules:
        cached_modules[mod_name] = sys.modules[mod_name]
        del sys.modules[mod_name]

try:
    sys.path.insert(0, LEGACY_PROJECT_ROOT)
    from mlsynth.mlsynth import FSCM as LegacyFSCM
    legacy_fscm_estimator_class = LegacyFSCM
except ImportError as e:
    print(f"Error importing legacy FSCM: {e}")
    sys.exit(1)
finally:
    sys.path = list(original_sys_path)
    for mod_name, mod_obj in cached_modules.items():
        sys.modules[mod_name] = mod_obj

if not legacy_fscm_estimator_class:
    print("Failed to import legacy FSCM class. Exiting.")
    sys.exit(1)

def save_results(data, filename, output_dir="comparison_outputs"):
    abs_output_dir = os.path.join(SCRIPT_DIR, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    filepath = os.path.join(abs_output_dir, filename)

    data_to_process = {}
    if hasattr(data, 'model_dump'):
        data_to_process = data.model_dump()
    elif hasattr(data, 'dict'):
        data_to_process = data.dict()
    elif isinstance(data, dict):
        data_to_process = data
    else:
        print(f"Warning: Unexpected data type for save_results: {type(data)}.")
        data_to_process = data 

    # Handle _prepped key specifically for legacy FSCM results if it contains DataFrames
    if isinstance(data_to_process, dict) and "_prepped" in data_to_process:
        # Replace with a placeholder or a summary if actual data isn't needed in JSON
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
        return obj

    serializable_data = convert_types_for_json(data_to_process)
    with open(filepath, "w") as f:
        json.dump(serializable_data, f, indent=4, allow_nan=True)
    print(f"Results saved to {filepath}")

def compare_fscm_outputs(legacy_res_dict, current_res_obj):
    summary = {"FSCM": {}}
    method_name = "FSCM"

    # ATT Comparison
    legacy_att = legacy_res_dict.get("Effects", {}).get("ATT")
    current_effects_attr = getattr(current_res_obj, 'effects', None)
    current_att = getattr(current_effects_attr, 'att', None) if current_effects_attr else None

    if legacy_att is not None and current_att is not None:
        summary[method_name]["ATT_match"] = np.isclose(legacy_att, current_att)
    else:
        summary[method_name]["ATT_match"] = (legacy_att is None and current_att is None)
    summary[method_name]["ATT_legacy"] = legacy_att
    summary[method_name]["ATT_current"] = current_att

    # Counterfactual Comparison
    legacy_cf_vector = legacy_res_dict.get("Vectors", {}).get("Counterfactual")
    current_vectors_attr = getattr(current_res_obj, 'vectors', None)
    current_cf_vector = getattr(current_vectors_attr, 'counterfactual', None) if current_vectors_attr else None

    if legacy_cf_vector is not None and current_cf_vector is not None:
        legacy_cf_flat = np.array(legacy_cf_vector).flatten()
        current_cf_flat = np.array(current_cf_vector).flatten()
        if legacy_cf_flat.shape == current_cf_flat.shape:
            summary[method_name]["Counterfactual_match"] = np.allclose(legacy_cf_flat, current_cf_flat, equal_nan=True)
        else:
            summary[method_name]["Counterfactual_match"] = False
            print(f"Warning: Counterfactual shapes differ for FSCM.")
    else:
        summary[method_name]["Counterfactual_match"] = (legacy_cf_vector is None and current_cf_vector is None)
    
    # Weights Comparison
    # Legacy FSCM stores weights as: "Weights": [donor_weights_dict, {"Cardinality...": ...}]
    # Current FSCM (BaseEstimatorResults) stores weights in: results.weights.donor_weights (which is a dict)
    legacy_weights_list = legacy_res_dict.get("Weights")
    legacy_weights_dict = None
    if isinstance(legacy_weights_list, list) and len(legacy_weights_list) > 0 and isinstance(legacy_weights_list[0], dict):
        legacy_weights_dict = legacy_weights_list[0]

    current_weights_attr = getattr(current_res_obj, 'weights', None)
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
            print(f"Warning: Number of donors in weights differ for FSCM.")
    elif legacy_weights_dict is None and current_weights_dict is None:
            summary[method_name]["Weights_match"] = True # Both are None
    else:
        summary[method_name]["Weights_match"] = False # One is None and the other isn't

    return summary

def print_comparison_summary(summary):
    print("\n--- Comparison Summary ---")
    numerical_differs = False
    for method, metrics in summary.items(): 
        print(f"\nComparing method: {method}")
        if "ATT_match" in metrics:
            att_legacy = metrics.get('ATT_legacy', 'N/A')
            att_current = metrics.get('ATT_current', 'N/A')
            att_legacy_str = f"{att_legacy:.4f}" if isinstance(att_legacy, float) else str(att_legacy)
            att_current_str = f"{att_current:.4f}" if isinstance(att_current, float) else str(att_current)
            print(f"  ATT match: {metrics['ATT_match']} (Legacy: {att_legacy_str}, Current: {att_current_str})")
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
        print("\nAll key numerical outputs (ATT, Counterfactual, Weights) appear consistent.")

# --- Configuration for FSCM Test ---
DATA_FILE = os.path.join(PROJECT_ROOT, "basedata", "smoking_data.csv")
OUTCOME_VARIABLE = "cigsale"
UNIT_ID_COLUMN_NAME = "state"
TIME_ID_COLUMN_NAME = "year"
TREATED_UNIT_ID = "California"
TREATMENT_START_YEAR = 1989
TREATMENT_INDICATOR_COLUMN = "treated_dummy"

# --- Load and Prepare Data ---
df = pd.read_csv(DATA_FILE)
df[TREATMENT_INDICATOR_COLUMN] = (
    (df[UNIT_ID_COLUMN_NAME] == TREATED_UNIT_ID) & 
    (df[TIME_ID_COLUMN_NAME] >= TREATMENT_START_YEAR)
).astype(int)

# --- Run Legacy FSCM ---
print("--- Running Legacy FSCM ---")
legacy_config_dict = {
    "df": df.copy(),
    "unitid": UNIT_ID_COLUMN_NAME,
    "time": TIME_ID_COLUMN_NAME,
    "outcome": OUTCOME_VARIABLE,
    "treat": TREATMENT_INDICATOR_COLUMN,
    "display_graphs": False
}
legacy_fscm_estimator = legacy_fscm_estimator_class(config=legacy_config_dict)
legacy_fscm_results = legacy_fscm_estimator.fit() # Returns a dict
save_results(legacy_fscm_results, "legacy_fscm_outputs.json")

# --- Run Current FSCM ---
print("\n--- Running Current FSCM ---")
current_fscm_config = current_fscm_config_class(
    df=df.copy(),
    unitid=UNIT_ID_COLUMN_NAME,
    time=TIME_ID_COLUMN_NAME,
    outcome=OUTCOME_VARIABLE,
    treat=TREATMENT_INDICATOR_COLUMN,
    display_graphs=False
)
current_fscm_estimator = current_fscm_estimator_class(config=current_fscm_config)
current_fscm_results = current_fscm_estimator.fit() # Returns a BaseEstimatorResults object
save_results(current_fscm_results, "current_fscm_outputs.json")

# --- Compare Results ---
comparison_summary = compare_fscm_outputs(legacy_fscm_results, current_fscm_results)
print_comparison_summary(comparison_summary)

print("\nFSCM Comparison Test Completed.")
