"""Predict resilient modulus (Mr) from whatever properties you have.

Usage:
  python predict_mr.py                      interactive: type values, Enter to skip
  python predict_mr.py LIQUID_LIMIT=25 MAX_LAB_DRY_DENSITY=130 ...
  python predict_mr.py --file inputs.xlsx   predict every row (blank cells = not given);
                                            writes inputs_predicted.xlsx / .csv
  python predict_mr.py --list               show allowed material / AASHTO / layer names
"""
import os
import sys

import pandas as pd

from mr_model import ALL_FEATURES, CATEGORICAL, EXTRA_FEATURES, MrPredictor


def show(result):
    lo, hi = result["range_80pct"]
    cv = result["cv_metrics"]
    for w in result["warnings"]:
        print("  note:", w)
    print(f"\n  Predicted RESILIENT_MODULUS = {result['RESILIENT_MODULUS']:.2f}")
    print(f"  Likely range (80%)          = {lo:.2f} - {hi:.2f}")
    print(f"  Inputs used ({len(result['inputs_used'])}): {', '.join(result['inputs_used'])}")
    print(f"  Accuracy of this input combination (cross-validated): "
          f"R2={cv['R2']:.2f}, MAE={cv['MAE']:.2f}, MAPE={cv['MAPE_%']:.1f}%")


def interactive(p):
    print("Enter the values you know; press Enter to skip a property.")
    print("For text properties you can type a number from the list shown.\n")
    inputs = {}
    for col in ALL_FEATURES:
        label = col + (" (optional extra)" if col in EXTRA_FEATURES else "")
        if col in CATEGORICAL:
            cats = p.categories[col]
            print(f"{label}:")
            for i, c in enumerate(cats, 1):
                print(f"   {i:2d}. {c}")
            val = input("  choice: ").strip()
            if val.isdigit() and 1 <= int(val) <= len(cats):
                val = cats[int(val) - 1]
        else:
            val = input(f"{label}: ").strip()
        if val:
            inputs[col] = val
    if not inputs:
        print("No values given.")
        return
    print("\nTraining/loading the model for this input combination...")
    show(p.predict(**inputs))


def main(argv):
    p = MrPredictor()
    if not argv:
        interactive(p)
    elif argv[0] == "--list":
        for col in CATEGORICAL:
            print(f"\n{col}:")
            for c in p.categories[col]:
                print("  ", c)
    elif argv[0] == "--file":
        path = argv[1]
        frame = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
        out = p.predict_table(frame)
        base, ext = os.path.splitext(path)
        out_path = f"{base}_predicted{ext if ext.lower() in ('.xlsx', '.csv') else '.csv'}"
        (out.to_excel if out_path.endswith(".xlsx") else out.to_csv)(out_path, index=False)
        print(f"Wrote {len(out)} predictions to {out_path}")
    else:
        inputs = dict(a.split("=", 1) for a in argv)
        show(p.predict(**inputs))


if __name__ == "__main__":
    main(sys.argv[1:])
