import pandas as pd
import numpy as np
import json
import os
import sys # Add sys import
from pathlib import Path # Add Path import

# Ensure the local mlsynth package is used
project_root = Path(__file__).resolve().parent.parent.parent # Adjusted for new location
sys.path.insert(0, str(project_root))
import json
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent # Adjusted for new location
LEGACY_MLSYNTH_PATH = BASE_DIR / "mlsynth-legacy"
DATA_PATH = BASE_DIR / "basedata" / "smoking_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "comparison_outputs" # Adjusted for new location
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

# Estimator and Data Config
OUTCOME_VAR = "cigsale"
UNIT_ID_VAR = "state" # This column contains string names like "California"
TIME_VAR = "year"
TREATED_UNIT_ID = "California"  # Corrected to string
TREATMENT_START_YEAR = 1989
NUM_DRAWS = 100 # Fewer draws for faster testing, default is 500
RANDOM_SEED = 42

# Keys for extracting results (adjust if legacy version used different keys)
LEGACY_KEY_EFFECTS = "Effects"
LEGACY_KEY_ATT = "ATT"
LEGACY_KEY_WEIGHT_V = "WeightV"
LEGACY_KEY_VECTORS = "Vectors"
LEGACY_KEY_COUNTERFACTUAL = "Counterfactual"

# SC Method Names (consistent with TSSC internal constants)
SC_METHODS_TO_COMPARE = ["SIMPLEX", "MSCa", "MSCb", "MSCc"]

def load_and_prepare_data():
    """Loads smoking data and adds a treatment column."""
    df = pd.read_csv(DATA_PATH)
    # df.dropna(subset=[OUTCOME_VAR], inplace=True) # Removed for now, let estimators handle
    df["treated"] = ((df[UNIT_ID_VAR] == TREATED_UNIT_ID) & (df[TIME_VAR] >= TREATMENT_START_YEAR)).astype(int)
    return df

def run_legacy_tssc(df: pd.DataFrame):
    """Runs TSSC from the legacy mlsynth library."""
    print("\n--- Running Legacy TSSC ---")
    original_sys_path = list(sys.path)
    if str(LEGACY_MLSYNTH_PATH) in sys.path:
        sys.path.remove(str(LEGACY_MLSYNTH_PATH))
    sys.path.insert(0, str(LEGACY_MLSYNTH_PATH))

    legacy_results_extracted = {}
    try:
        from mlsynth.mlsynth import TSSC as LegacyTSSC
        
        np.random.seed(RANDOM_SEED)

        legacy_config_dict = {
            "df": df,
            "outcome": OUTCOME_VAR,
            "treat": "treated",
            "unitid": UNIT_ID_VAR,
            "time": TIME_VAR,
            "draws": NUM_DRAWS,
            "display_graphs": False 
        }
        legacy_estimator = LegacyTSSC(config=legacy_config_dict)
        legacy_raw_results_list = legacy_estimator.fit() 

        for res_dict_outer in legacy_raw_results_list:
            method_name = next(iter(res_dict_outer)) 
            if method_name in SC_METHODS_TO_COMPARE:
                res_dict_inner = res_dict_outer[method_name]
                att = res_dict_inner.get(LEGACY_KEY_EFFECTS, {}).get(LEGACY_KEY_ATT)
                weights = res_dict_inner.get(LEGACY_KEY_WEIGHT_V)
                if weights is not None:
                    weights = np.array(weights).flatten() 
                
                counterfactual = res_dict_inner.get(LEGACY_KEY_VECTORS, {}).get(LEGACY_KEY_COUNTERFACTUAL)
                if counterfactual is not None:
                    counterfactual = np.array(counterfactual).flatten()

                legacy_results_extracted[method_name] = {
                    "att": att,
                    "weights": weights.tolist() if weights is not None else None,
                    "counterfactual": counterfactual.tolist() if counterfactual is not None else None,
                }
                print(f"Legacy {method_name} ATT: {att}")
        
    except ImportError as e:
        print(f"Error importing legacy TSSC: {e}. Check LEGACY_MLSYNTH_PATH and legacy structure.")
        return None
    except Exception as e:
        print(f"Error running legacy TSSC: {e}")
        return None
    finally:
        sys.path = original_sys_path 
        if 'mlsynth.mlsynth' in sys.modules:
            del sys.modules['mlsynth.mlsynth']
        if 'mlsynth' in sys.modules:
            mlsynth_module = sys.modules['mlsynth']
            if hasattr(mlsynth_module, '__file__') and mlsynth_module.__file__ and \
               str(LEGACY_MLSYNTH_PATH) in mlsynth_module.__file__:
                del sys.modules['mlsynth']

    return legacy_results_extracted

def run_current_tssc(df: pd.DataFrame):
    """Runs TSSC from the current (refactored) mlsynth library."""
    print("\n--- Running Current TSSC ---")
    
    print("Diagnostic: 'treated' column for TREATED_UNIT_ID before TSSCConfig:")
    california_data = df[df[UNIT_ID_VAR] == TREATED_UNIT_ID]
    if not california_data.empty:
        print(california_data[[TIME_VAR, 'treated', OUTCOME_VAR]].tail(15))
        print(f"Sum of 'treated' for unit '{TREATED_UNIT_ID}': {california_data['treated'].sum()}")
        print(f"NaNs in {OUTCOME_VAR} for unit '{TREATED_UNIT_ID}' post-treatment: {california_data[(california_data[TIME_VAR] >= TREATMENT_START_YEAR)][OUTCOME_VAR].isnull().sum()}")
    else:
        print(f"No data found for TREATED_UNIT_ID '{TREATED_UNIT_ID}' in the input df to run_current_tssc.")


    current_results_extracted = {}
    try:
        from mlsynth.estimators.tssc import TSSC
        from mlsynth.config_models import TSSCConfig
        
        config = TSSCConfig(
            df=df,
            outcome=OUTCOME_VAR,
            treat="treated",
            unitid=UNIT_ID_VAR,
            time=TIME_VAR,
            draws=NUM_DRAWS,
            seed=RANDOM_SEED, 
            display_graphs=False,
        )
        current_estimator = TSSC(config=config)
        current_results_list = current_estimator.fit() 

        for res_obj in current_results_list:
            method_name = res_obj.method_details.method_name
            if method_name in SC_METHODS_TO_COMPARE:
                att = res_obj.effects.att
                weights_dict = res_obj.weights.donor_weights 
                
                weights_values = None
                if weights_dict:
                    weights_values = [weights_dict[key] for key in sorted(weights_dict.keys())]

                counterfactual = res_obj.time_series.counterfactual_outcome
                if counterfactual is not None:
                    counterfactual = np.array(counterfactual).flatten()

                current_results_extracted[method_name] = {
                    "att": att,
                    "weights_dict": weights_dict, 
                    "weights_sorted_list": weights_values, 
                    "counterfactual": counterfactual.tolist() if counterfactual is not None else None,
                }
                print(f"Current {method_name} ATT: {att}")

    except Exception as e:
        print(f"Error running current TSSC: {e}")
        return None
    return current_results_extracted

def compare_results(legacy_res, current_res):
    print("\n--- Comparison Summary ---")
    if legacy_res is None or current_res is None:
        print("Comparison aborted due to errors in one of the runs.")
        return

    all_match = True
    for method in SC_METHODS_TO_COMPARE:
        print(f"\nComparing method: {method}")
        if method not in legacy_res or method not in current_res:
            print(f"  Method {method} not found in both results. Skipping.")
            all_match = False
            continue

        leg_m = legacy_res[method]
        cur_m = current_res[method]

        # Compare ATT
        if leg_m["att"] is not None and cur_m["att"] is not None:
            att_match = np.allclose(leg_m["att"], cur_m["att"], atol=1e-5)
            print(f"  ATT match: {att_match} (Legacy: {leg_m['att']:.6f}, Current: {cur_m['att']:.6f})")
            if not att_match: all_match = False
        else:
            print(f"  ATT: One or both are None (Legacy: {leg_m['att']}, Current: {cur_m['att']})")
            if not (leg_m["att"] is None and cur_m["att"] is None): all_match = False

        # Compare Counterfactuals
        if leg_m["counterfactual"] is not None and cur_m["counterfactual"] is not None:
            if len(leg_m["counterfactual"]) == len(cur_m["counterfactual"]):
                cf_match = np.allclose(leg_m["counterfactual"], cur_m["counterfactual"], atol=1e-5)
                print(f"  Counterfactual match: {cf_match}")
                if not cf_match: all_match = False
            else:
                print(f"  Counterfactual: Length mismatch (Legacy: {len(leg_m['counterfactual'])}, Current: {len(cur_m['counterfactual'])})")
                all_match = False
        else:
            print(f"  Counterfactual: One or both are None.")
            if not (leg_m["counterfactual"] is None and cur_m["counterfactual"] is None): all_match = False
        
        # Compare Weights
        if leg_m["weights"] is not None and cur_m["weights_sorted_list"] is not None:
            if len(leg_m["weights"]) == len(cur_m["weights_sorted_list"]):
                weights_match = np.allclose(leg_m["weights"], cur_m["weights_sorted_list"], atol=1e-5)
                print(f"  Weights (sorted list) match: {weights_match}")
                if not weights_match: all_match = False
            else:
                print(f"  Weights: Length mismatch (Legacy: {len(leg_m['weights'])}, Current: {len(cur_m['weights_sorted_list'])})")
                all_match = False
        else:
            print(f"  Weights: One or both sorted lists are None.")
            if not (leg_m["weights"] is None and cur_m["weights_sorted_list"] is None):
                 all_match = False

    if all_match:
        print("\nSUCCESS: All compared numerical outputs are close!")
    else:
        print("\nWARNING: Some numerical outputs differ. Check saved JSON files for details.")

def main():
    df = load_and_prepare_data()
    
    legacy_outputs = run_legacy_tssc(df.copy())
    current_outputs = run_current_tssc(df.copy())

    if legacy_outputs:
        with open(OUTPUT_DIR / "legacy_tssc_outputs.json", "w") as f:
            json.dump(legacy_outputs, f, indent=4)
        print(f"Legacy outputs saved to {OUTPUT_DIR / 'legacy_tssc_outputs.json'}")
        
    if current_outputs:
        with open(OUTPUT_DIR / "current_tssc_outputs.json", "w") as f:
            json.dump(current_outputs, f, indent=4)
        print(f"Current outputs saved to {OUTPUT_DIR / 'current_tssc_outputs.json'}")

    if legacy_outputs and current_outputs: 
        compare_results(legacy_outputs, current_outputs)
    else:
        print("\nComparison skipped due to errors in one or both runs.")

if __name__ == "__main__":
    main()
