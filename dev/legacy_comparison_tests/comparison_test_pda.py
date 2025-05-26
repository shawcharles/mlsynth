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
current_pda_estimator_class = None
current_pda_config_class = None
try:
    sys.path.insert(0, PROJECT_ROOT)
    from mlsynth.estimators.pda import PDA
    from mlsynth.config_models import PDAConfig
    current_pda_estimator_class = PDA
    current_pda_config_class = PDAConfig
finally:
    sys.path = list(original_sys_path)

if not current_pda_estimator_class or not current_pda_config_class:
    print("Failed to import current PDA modules. Exiting.")
    sys.exit(1)

# Import LEGACY project modules
legacy_pda_estimator_class = None
cached_modules = {}
modules_to_clear = [m for m in sys.modules if m == 'mlsynth' or m.startswith('mlsynth.')]

for mod_name in modules_to_clear:
    if mod_name in sys.modules:
        cached_modules[mod_name] = sys.modules[mod_name]
        del sys.modules[mod_name]

try:
    sys.path.insert(0, LEGACY_PROJECT_ROOT)
    from mlsynth.mlsynth import PDA as LegacyPDA
    legacy_pda_estimator_class = LegacyPDA
except ImportError as e:
    print(f"Error importing legacy PDA: {e}")
    sys.exit(1)
finally:
    sys.path = list(original_sys_path)
    for mod_name, mod_obj in cached_modules.items():
        sys.modules[mod_name] = mod_obj

if not legacy_pda_estimator_class:
    print("Failed to import legacy PDA class. Exiting.")
    sys.exit(1)

def save_results(data, filename, output_dir="comparison_outputs"):
    abs_output_dir = os.path.join(SCRIPT_DIR, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    filepath = os.path.join(abs_output_dir, filename)

    data_to_process = {}
    if hasattr(data, 'model_dump'):  # For Pydantic models
        data_to_process = data.model_dump()
    elif hasattr(data, 'dict'):  # For older Pydantic or similar
        data_to_process = data.dict()
    elif isinstance(data, dict):
        data_to_process = data
    else:
        print(f"Warning: Unexpected data type for save_results: {type(data)}.")
        data_to_process = data 

    if isinstance(data_to_process, dict) and "_prepped" in data_to_process:
        data_to_process["_prepped"] = "Content of '_prepped' removed for JSON serialization."

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
        elif pd.isna(obj):
            return None
        return obj

    serializable_data = convert_types_for_json(data_to_process)
    with open(filepath, "w") as f:
        json.dump(serializable_data, f, indent=4, allow_nan=True)
    print(f"Results saved to {filepath}")

def compare_pda_outputs(legacy_res_dict, current_res_obj):
    summary = {"PDA": {}}
    method_name = "PDA"

    # ATT Comparison
    legacy_att = legacy_res_dict.get("Effects", {}).get("ATT") # Corrected key for legacy
    current_effects_attr = getattr(current_res_obj, 'effects', None)
    current_att = None
    if current_effects_attr:
        current_att = getattr(current_effects_attr, 'att', None)
        if current_att is None: # Try nested location
            other_effects_attr = getattr(current_effects_attr, 'other_effects', None)
            if other_effects_attr and isinstance(other_effects_attr, dict):
                current_att = other_effects_attr.get("ATT")

    if legacy_att is not None and current_att is not None:
        summary[method_name]["ATT_match"] = np.isclose(legacy_att, current_att)
    else:
        summary[method_name]["ATT_match"] = (legacy_att is None and current_att is None)
    summary[method_name]["ATT_legacy"] = legacy_att
    summary[method_name]["ATT_current"] = current_att

    # Counterfactual Comparison
    legacy_cf_vector = legacy_res_dict.get("Vectors", {}).get("Counterfactual")
    current_time_series_attr = getattr(current_res_obj, 'time_series', None)
    current_cf_vector = getattr(current_time_series_attr, 'synthetic_outcome', None) if current_time_series_attr else None

    if legacy_cf_vector is not None and current_cf_vector is not None:
        legacy_cf_flat = np.array(legacy_cf_vector).flatten()
        current_cf_flat = np.array(current_cf_vector).flatten()
        if legacy_cf_flat.shape == current_cf_flat.shape:
            summary[method_name]["Counterfactual_match"] = np.allclose(legacy_cf_flat, current_cf_flat, equal_nan=True)
        else:
            summary[method_name]["Counterfactual_match"] = False
            print(f"Warning: Counterfactual shapes differ for PDA. Legacy: {legacy_cf_flat.shape}, Current: {current_cf_flat.shape}")
    elif legacy_cf_vector is None and current_cf_vector is None:
        summary[method_name]["Counterfactual_match"] = True
    else:
        summary[method_name]["Counterfactual_match"] = False
        print(f"Warning: One of the counterfactuals is None for PDA. Legacy: {type(legacy_cf_vector)}, Current: {type(current_cf_vector)}")

    # Weights Comparison
    legacy_weights_data = legacy_res_dict.get("Weights") # This is actually the "Betas" from legacy output
    legacy_weights_dict = None
    if isinstance(legacy_weights_data, dict): # If "Weights" key exists and is a dict
        legacy_weights_dict = legacy_weights_data
    else: # Fallback to "Betas" for legacy PDA if "Weights" is not what we expect
        legacy_betas_data = legacy_res_dict.get("Betas")
        if isinstance(legacy_betas_data, dict):
            legacy_weights_dict = legacy_betas_data


    current_weights_attr = getattr(current_res_obj, 'weights', None)
    current_weights_dict = None
    if current_weights_attr:
        current_weights_dict = getattr(current_weights_attr, 'donor_weights', None)
        # The current JSON shows 'donor_weights' as null.
        # The actual weights for PDA 'fs' method are coefficients/betas,
        # which might be in method_details or additional_outputs in the Pydantic model.
        # For now, we stick to 'donor_weights'. If it's consistently null,
        # and legacy has betas, they won't match, which is correct based on current structure.
        # If the current Pydantic model *does* populate donor_weights for 'fs' method,
        # then this comparison is fine. The JSON output is just a snapshot.

    if legacy_weights_dict is not None and current_weights_dict is not None:
        # Ensure keys are strings for consistent sorting and comparison
        legacy_w_str_keys = {str(k): v for k, v in legacy_weights_dict.items()}
        current_w_str_keys = {str(k): v for k, v in current_weights_dict.items()}

        legacy_w_sorted = sorted(legacy_w_str_keys.items())
        current_w_sorted = sorted(current_w_str_keys.items())
        
        if len(legacy_w_sorted) == len(current_w_sorted):
            weights_match_val = all(
                d_leg == d_curr and np.isclose(w_leg, w_curr, equal_nan=True) # Added equal_nan
                for (d_leg, w_leg), (d_curr, w_curr) in zip(legacy_w_sorted, current_w_sorted)
            )
            summary[method_name]["Weights_match"] = weights_match_val
        else:
            summary[method_name]["Weights_match"] = False
            print(f"Warning: Number of donors/coefficients in weights differ for PDA. Legacy: {len(legacy_w_sorted)}, Current: {len(current_w_sorted)}")
    elif legacy_weights_dict is None and current_weights_dict is None:
        summary[method_name]["Weights_match"] = True 
    else:
        summary[method_name]["Weights_match"] = False
        print(f"Warning: One of the weights dictionaries is None or empty for PDA. Legacy type: {type(legacy_weights_dict)}, Current type: {type(current_weights_dict)}")
        
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
            if not metrics['ATT_match'] and (att_legacy is not None or att_current is not None): numerical_differs = True
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

# --- Configuration for PDA Test ---
DATA_FILE = os.path.join(PROJECT_ROOT, "basedata", "smoking_data.csv")
OUTCOME_VARIABLE = "cigsale"
UNIT_ID_COLUMN_NAME = "state"
TIME_ID_COLUMN_NAME = "year"
TREATED_UNIT_ID = "California" 
TREATMENT_START_YEAR = 1989    
TREATMENT_INDICATOR_COLUMN = "treated_dummy"
METHOD = "fs"  # Default PDA method, can be 'LASSO', 'l2', or 'fs'

# --- Load and Prepare Data ---
df = pd.read_csv(DATA_FILE)
df[UNIT_ID_COLUMN_NAME] = df[UNIT_ID_COLUMN_NAME].astype(str)
df[TIME_ID_COLUMN_NAME] = df[TIME_ID_COLUMN_NAME].astype(int)
df[OUTCOME_VARIABLE] = pd.to_numeric(df[OUTCOME_VARIABLE], errors='coerce')

df[TREATMENT_INDICATOR_COLUMN] = (
    (df[UNIT_ID_COLUMN_NAME] == str(TREATED_UNIT_ID)) & 
    (df[TIME_ID_COLUMN_NAME] >= TREATMENT_START_YEAR)
).astype(int)

df = df.dropna(subset=[OUTCOME_VARIABLE])

# --- Run Legacy PDA ---
print("--- Running Legacy PDA ---")
legacy_config_dict = {
    "df": df.copy(),
    "unitid": UNIT_ID_COLUMN_NAME,
    "time": TIME_ID_COLUMN_NAME,
    "outcome": OUTCOME_VARIABLE,
    "treat": TREATMENT_INDICATOR_COLUMN,
    "method": METHOD,
    "display_graphs": False,
}
legacy_pda_estimator = legacy_pda_estimator_class(config=legacy_config_dict)
legacy_pda_results = None
try:
    legacy_pda_results = legacy_pda_estimator.fit() 
    save_results(legacy_pda_results, "legacy_pda_outputs.json")
except Exception as e:  # Catching generic Exception for now
    print(f"!!! Legacy PDA failed to run due to an error: {e}")
    legacy_pda_results = {"error": f"Legacy PDA failed: {e}"}
    save_results(legacy_pda_results, "legacy_pda_outputs.json")

# --- Run Current PDA ---
print("\n--- Running Current PDA ---")
current_pda_config = current_pda_config_class(
    df=df.copy(),
    unitid=UNIT_ID_COLUMN_NAME,
    time=TIME_ID_COLUMN_NAME,
    outcome=OUTCOME_VARIABLE,
    treat=TREATMENT_INDICATOR_COLUMN,
    method=METHOD,
    display_graphs=False,
)
current_pda_estimator = current_pda_estimator_class(config=current_pda_config)
current_pda_results = current_pda_estimator.fit() 
save_results(current_pda_results, "current_pda_outputs.json")

# --- Compare Results ---
if legacy_pda_results and "error" not in legacy_pda_results:
    comparison_summary = compare_pda_outputs(legacy_pda_results, current_pda_results)
    print_comparison_summary(comparison_summary)
else:
    print("\n--- Comparison Summary ---")
    print("Skipping comparison due to legacy PDA failure or no results.")
    print("Current PDA execution results saved.")
    if current_pda_results:
        current_effects_attr = getattr(current_pda_results, 'effects', None)
        current_att = getattr(current_effects_attr, 'att', None) if current_effects_attr else None
        print(f"  Current PDA ATT: {current_att if current_att is not None else 'N/A'}")

print("\nPDA Comparison Test Completed.")
