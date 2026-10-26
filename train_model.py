"""Rebuild the training data from "Mr Data.xlsx" and pre-train the main models.

Run this again whenever the Excel data changes (close the file in Excel first).
"""
import shutil

from mr_model import (CACHE_DIR, DATA_XLSX, EXTRA_FEATURES, FEATURES, MrPredictor,
                      build_training_data)

PRETRAIN = {
    "All requested properties": FEATURES,
    "All properties + thickness/layer type": FEATURES + EXTRA_FEATURES,
    "K1, K2, K3 only": ["RESILIENT_MODULUS_K1", "RESILIENT_MODULUS_K2", "RESILIENT_MODULUS_K3"],
    "Soil properties only (no K values)": [
        "MATL_CODE_EXP", "LIQUID_LIMIT", "PLASTIC_LIMIT", "PLASTICITY_INDEX",
        "AASHTO_SOIL_CLASS_EXP", "SPEC_GRAVITY", "MAX_LAB_DRY_DENSITY",
        "OPTIMUM_LAB_MOISTURE", "HYDRAULIC_CONDUCTIVITY"],
}

if __name__ == "__main__":
    try:
        df = build_training_data()
    except PermissionError:
        raise SystemExit(f"Cannot read {DATA_XLSX} - close it in Excel and run again.")
    print(f"Training rows with measured Mr: {len(df)}")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)  # old models belong to old data
    p = MrPredictor()
    print("\nCross-validated accuracy (5 folds, grouped by test section):")
    for name, cols in PRETRAIN.items():
        cv = p.get_model(cols)["cv"]
        print(f"  {name:40s} R2={cv['R2']:.3f}  MAE={cv['MAE']:.2f}  MAPE={cv['MAPE_%']:.1f}%")
    print("\nOther input combinations are trained automatically the first time they are used.")
