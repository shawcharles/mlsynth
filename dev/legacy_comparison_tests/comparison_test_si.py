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
current_si_estimator_class = None
current_si_config_class = None
try:
    sys.path.insert(0, PROJECT_ROOT)
    from mlsynth.estimators.si import SI
    from mlsynth.config_models import SIConfig # Removed InterventionConfig
    current_si_estimator_class = SI
    current_si_config_class = SIConfig
finally:
    sys.path = list(original_sys_path)

if not current_si_estimator_class or not current_si_config_class:
    print("Failed to import current SI modules. Exiting.")
    sys.exit(1)

# Import LEGACY project modules
legacy_si_estimator_class = None
cached_modules = {}
modules_to_clear = [m for m in sys.modules if m == 'mlsynth' or m.startswith('mlsynth.')]

for mod_name in modules_to_clear:
    if mod_name in sys.modules:
        cached_modules[mod_name] = sys.modules[mod_name]
        del sys.modules[mod_name]

try:
    sys.path.insert(0, LEGACY_PROJECT_ROOT)
    from mlsynth.mlsynth import SI as LegacySI
    legacy_si_estimator_class = LegacySI
except ImportError as e:
    print(f"Error importing legacy SI: {e}")
    sys.exit(1)
finally:
    sys.path = list(original_sys_path)
    for mod_name, mod_obj in cached_modules.items():
        sys.modules[mod_name] = mod_obj

if not legacy_si_estimator_class:
    print("Failed to import legacy SI class. Exiting.")
    sys.exit(1)

def save_results(data, filename, output_dir="comparison_outputs"):
    abs_output_dir = os.path.join(SCRIPT_DIR, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    filepath = os.path.join(abs_output_dir, filename)

    data_to_process = {}
    # Handle if data itself is a Pydantic model
    if hasattr(data, 'model_dump'):
        data_to_process = data.model_dump()
    # Handle if data is a dictionary possibly containing Pydantic models as values (like SI output)
    elif isinstance(data, dict):
        data_to_process = {}
        for key, value in data.items():
            if hasattr(value, 'model_dump'):
                data_to_process[key] = value.model_dump()
            elif hasattr(value, 'dict'): # For older Pydantic or similar dict-like objects
                data_to_process[key] = value.dict()
            else:
                data_to_process[key] = value # Assume already serializable or will be handled by convert_types_for_json
    # Handle if data is a list possibly containing Pydantic models
    elif isinstance(data, list):
        data_to_process = []
        for item in data:
            if hasattr(item, 'model_dump'):
                data_to_process.append(item.model_dump())
            elif hasattr(item, 'dict'):
                data_to_process.append(item.dict())
            else:
                data_to_process.append(item)
    else:
        print(f"Warning: Unexpected data type for save_results: {type(data)}. Attempting direct processing.")
        data_to_process = data

    # This _prepped removal should ideally be done after model_dump if _prepped is part of the model
    # For now, assuming it's a top-level key in a dict that might have been passed directly.
    # If data_to_process is a list, this check won't apply directly.
    if isinstance(data_to_process, dict) and "_prepped" in data_to_process:
         data_to_process["_prepped"] = "Content of '_prepped' removed for JSON serialization."
    # If data_to_process is a list of dicts (e.g. from list of Pydantic models)
    elif isinstance(data_to_process, list):
        for item_idx, item_val in enumerate(data_to_process):
            if isinstance(item_val, dict) and "_prepped" in item_val:
                data_to_process[item_idx]["_prepped"] = "Content of '_prepped' removed for JSON serialization."


    def convert_types_for_json(obj):
        # If obj is a Pydantic model instance that wasn't caught above (e.g., nested)
        if hasattr(obj, 'model_dump'):
            return convert_types_for_json(obj.model_dump()) # Recursively process the dumped dict
        elif isinstance(obj, dict):
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

def compare_si_outputs(legacy_res_dict, current_res_obj_dict):
    # SI returns a dictionary of results, one for each intervention.
    # We'll compare results for the "Proposition 99" intervention.
    intervention_name = "Proposition 99" # Matches SI_INTERS_COLUMNS[0]
    summary = {f"SI_{intervention_name}": {}}
    method_name_key = f"SI_{intervention_name}"

    # Get results for the specific intervention
    # Legacy SI output is Dict[str, Dict], where str is intervention name (column name)
    legacy_intervention_data = legacy_res_dict.get(intervention_name, {})
    # Current SI output is Dict[str, BaseEstimatorResults]
    current_intervention_pydantic_obj = current_res_obj_dict.get(intervention_name) if isinstance(current_res_obj_dict, dict) else None

    if not legacy_intervention_data:
        print(f"Warning: Legacy SI results for intervention '{intervention_name}' not found.")
        summary[method_name_key]["ATT_match"] = False
        summary[method_name_key]["Counterfactual_match"] = False
        summary[method_name_key]["Weights_match"] = False
        return summary
        
    if not current_intervention_pydantic_obj:
        print(f"Warning: Current SI results for intervention '{intervention_name}' not found or not in expected Pydantic object format.")
        summary[method_name_key]["ATT_match"] = False
        summary[method_name_key]["Counterfactual_match"] = False
        summary[method_name_key]["Weights_match"] = False
        return summary

    # ATT Comparison
    legacy_att = legacy_intervention_data.get("Effects", {}).get("ATT")
    current_effects_attr = getattr(current_intervention_pydantic_obj, 'effects', None)
    current_att = getattr(current_effects_attr, 'att', None) if current_effects_attr else None

    if legacy_att is not None and current_att is not None:
        summary[method_name_key]["ATT_match"] = np.isclose(legacy_att, current_att)
    else:
        summary[method_name_key]["ATT_match"] = (legacy_att is None and current_att is None)
    summary[method_name_key]["ATT_legacy"] = legacy_att
    summary[method_name_key]["ATT_current"] = current_att

    # Counterfactual Comparison
    legacy_cf_vector = legacy_intervention_data.get("Vectors", {}).get("Counterfactual")
    current_time_series_attr = getattr(current_intervention_pydantic_obj, 'time_series', None)
    # SI's BaseEstimatorResults might use 'synthetic_outcome' or 'counterfactual_outcome'.
    # Let's try 'synthetic_outcome' first as it's more common in other estimators, then 'counterfactual_outcome'.
    current_cf_vector = None
    if current_time_series_attr:
        current_cf_vector = getattr(current_time_series_attr, 'synthetic_outcome', None)
        if current_cf_vector is None: # Fallback if synthetic_outcome is None or not present
            current_cf_vector = getattr(current_time_series_attr, 'counterfactual_outcome', None)


    if legacy_cf_vector is not None and current_cf_vector is not None:
        legacy_cf_flat = np.array(legacy_cf_vector).flatten()
        current_cf_flat = np.array(current_cf_vector).flatten()
        if legacy_cf_flat.shape == current_cf_flat.shape:
            summary[method_name_key]["Counterfactual_match"] = np.allclose(legacy_cf_flat, current_cf_flat, equal_nan=True)
        else:
            summary[method_name_key]["Counterfactual_match"] = False
            print(f"Warning: Counterfactual shapes differ for SI '{intervention_name}'. Legacy: {legacy_cf_flat.shape}, Current: {current_cf_flat.shape}")
    elif legacy_cf_vector is None and current_cf_vector is None:
        summary[method_name_key]["Counterfactual_match"] = True
    else:
        summary[method_name_key]["Counterfactual_match"] = False
        print(f"Warning: One of the counterfactuals is None for SI '{intervention_name}'. Legacy: {type(legacy_cf_vector)}, Current: {type(current_cf_vector)}")

    # Weights Comparison
    # SI typically involves PCR, so "weights" might be factor loadings or similar, not direct donor weights.
    # The legacy structure for SI weights is unknown without seeing an output.
    # Current BaseEstimatorResults has 'donor_weights'.
    legacy_weights_dict = legacy_intervention_data.get("Weights") # Assuming it's a dict if present
    
    current_weights_attr = getattr(current_intervention_pydantic_obj, 'weights', None)
    current_weights_dict = getattr(current_weights_attr, 'donor_weights', None) if current_weights_attr else None

    if legacy_weights_dict is not None and current_weights_dict is not None and isinstance(legacy_weights_dict, dict) and isinstance(current_weights_dict, dict):
        legacy_w_sorted = sorted({str(k): v for k, v in legacy_weights_dict.items()}.items())
        current_w_sorted = sorted({str(k): v for k, v in current_weights_dict.items()}.items())
        
        if len(legacy_w_sorted) == len(current_w_sorted):
            weights_match_val = all(
                d_leg == d_curr and np.isclose(w_leg, w_curr, equal_nan=True)
                for (d_leg, w_leg), (d_curr, w_curr) in zip(legacy_w_sorted, current_w_sorted)
            )
            summary[method_name_key]["Weights_match"] = weights_match_val
        else:
            summary[method_name_key]["Weights_match"] = False
            print(f"Warning: Number of donors/coefficients in weights differ for SI '{intervention_name}'.")
    elif legacy_weights_dict is None and current_weights_dict is None:
        summary[method_name_key]["Weights_match"] = True 
    else:
        summary[method_name_key]["Weights_match"] = False
        print(f"Warning: One of the weights dictionaries is None or not a dict for SI '{intervention_name}'. Legacy type: {type(legacy_weights_dict)}, Current type: {type(current_weights_dict)}")
        
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

# --- Configuration for SI Test ---
DATA_FILE = os.path.join(PROJECT_ROOT, "basedata", "smoking_data.csv")
OUTCOME_VARIABLE = "cigsale"
UNIT_ID_COLUMN_NAME = "state"
TIME_ID_COLUMN_NAME = "year"
TREATED_UNIT_ID = "California" 
TREATMENT_START_YEAR = 1989    
TREATMENT_INDICATOR_COLUMN = "treated_dummy"

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

# --- Run Legacy SI ---
print("--- Running Legacy SI ---")
# For SI, 'inters' is a list of column names that are binary indicators.
# We'll use "Proposition 99" as an example alternative intervention column.
# Ensure this column exists and is appropriately formatted (binary) in the DataFrame.
# The script already loads 'Proposition 99'. We assume it's suitable or SI handles it.
SI_INTERS_COLUMNS = ["Proposition 99"]

legacy_config_dict = {
    "df": df.copy(),
    "unitid": UNIT_ID_COLUMN_NAME,
    "time": TIME_ID_COLUMN_NAME,
    "outcome": OUTCOME_VARIABLE,
    "treat": TREATMENT_INDICATOR_COLUMN, # Main treatment
    "inters": SI_INTERS_COLUMNS, # Alternative intervention(s)
    "display_graphs": False,
}
legacy_si_estimator = legacy_si_estimator_class(config=legacy_config_dict)
legacy_si_results = None
try:
    legacy_si_results = legacy_si_estimator.fit() 
    save_results(legacy_si_results, "legacy_si_outputs.json")
except Exception as e:  # Catching generic Exception for now
    print(f"!!! Legacy SI failed to run due to an error: {e}")
    legacy_si_results = {"error": f"Legacy SI failed: {e}"}
    save_results(legacy_si_results, "legacy_si_outputs.json")

# --- Run Current SI ---
print("\n--- Running Current SI ---")
current_si_config = current_si_config_class(
    df=df.copy(),
    unitid=UNIT_ID_COLUMN_NAME,
    time=TIME_ID_COLUMN_NAME,
    outcome=OUTCOME_VARIABLE,
    treat=TREATMENT_INDICATOR_COLUMN, # Main treatment
    inters=SI_INTERS_COLUMNS, # Alternative intervention(s)
    display_graphs=False,
)
current_si_estimator = current_si_estimator_class(config=current_si_config)
current_si_results = current_si_estimator.fit() 
save_results(current_si_results, "current_si_outputs.json")

# --- Compare Results ---
if legacy_si_results and "error" not in legacy_si_results:
    comparison_summary = compare_si_outputs(legacy_si_results, current_si_results)
    print_comparison_summary(comparison_summary)
else:
    print("\n--- Comparison Summary ---")
    print("Skipping comparison due to legacy SI failure or no results.")
    print("Current SI execution results saved.")
    if current_si_results:
        current_effects_attr = getattr(current_si_results, 'effects', None)
        current_att = getattr(current_effects_attr, 'att', None) if current_effects_attr else None
        print(f"  Current SI ATT: {current_att if current_att is not None else 'N/A'}")

print("\nSI Comparison Test Completed.")
