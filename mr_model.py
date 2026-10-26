"""Resilient modulus (Mr) predictor that works with any subset of inputs.

For each combination of inputs you provide, a gradient-boosting model is trained
on exactly those columns (rows of "Mr Data.xlsx" that have a measured Mr) and
cached in ``model_cache/``. So with all values given every property is used, and
with only some values given the prediction uses only those.
"""
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_XLSX = os.path.join(HERE, "Mr Data.xlsx")
TRAIN_DATA = os.path.join(HERE, "mr_training_data.csv")
CACHE_DIR = os.path.join(HERE, "model_cache")

TARGET = "RESILIENT_MODULUS"
CATEGORICAL = ["MATL_CODE_EXP", "AASHTO_SOIL_CLASS_EXP", "LAYER_TYPE_EXP"]
# Columns requested for the model, in the order of the data sheet.
FEATURES = [
    "MATL_CODE_EXP", "LIQUID_LIMIT", "PLASTIC_LIMIT", "PLASTICITY_INDEX",
    "AASHTO_SOIL_CLASS_EXP", "SPEC_GRAVITY", "MAX_LAB_DRY_DENSITY",
    "OPTIMUM_LAB_MOISTURE", "RESILIENT_MODULUS_K1", "RESILIENT_MODULUS_K2",
    "RESILIENT_MODULUS_K3", "POISSONS_RATIO", "HYDRAULIC_CONDUCTIVITY",
]
# Optional extras: Mr in this data also depends on the layer (thickness/position).
EXTRA_FEATURES = ["REPR_THICKNESS", "LAYER_TYPE_EXP"]
ALL_FEATURES = FEATURES + EXTRA_FEATURES

GROUP_COLS = ["STATE_CODE", "SHRP_ID"]  # keep a test section's rows in one CV fold


def build_training_data(xlsx_path=DATA_XLSX, out_path=TRAIN_DATA):
    """Extract the rows with a measured Mr from the Excel sheet and save them."""
    df = pd.read_excel(xlsx_path)
    df = df[df[TARGET].notna()]
    df = df[GROUP_COLS + ALL_FEATURES + [TARGET]].reset_index(drop=True)
    for c in CATEGORICAL:
        df[c] = df[c].astype("object").where(df[c].notna(), None)
        df[c] = df[c].map(lambda v: None if v is None else str(v).strip())
    df.to_csv(out_path, index=False)
    return df


def _norm(text):
    return " ".join(str(text).lower().split())


class MrPredictor:
    def __init__(self, train_path=TRAIN_DATA, cache_dir=CACHE_DIR):
        if not os.path.exists(train_path):
            build_training_data(out_path=train_path)
        self.data = pd.read_csv(train_path, dtype={c: str for c in GROUP_COLS + CATEGORICAL})
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.categories = {
            c: sorted(self.data[c].dropna().unique()) for c in CATEGORICAL
        }
        self._lookup = {c: {_norm(v): v for v in cats} for c, cats in self.categories.items()}
        self._models = {}

    # ---------- input handling ----------
    def clean_inputs(self, inputs):
        """Keep the known, non-empty inputs; returns (values, warnings)."""
        aliases = {_norm(f): f for f in ALL_FEATURES}
        values, warnings = {}, []
        for key, val in inputs.items():
            col = aliases.get(_norm(key))
            if col is None:
                warnings.append(f"Ignored unknown input '{key}'.")
                continue
            if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).strip() == "":
                continue
            if col in CATEGORICAL:
                match = self._lookup[col].get(_norm(val))
                if match is None:
                    warnings.append(f"'{val}' is not a known {col} value; it was ignored.")
                    continue
                values[col] = match
            else:
                try:
                    values[col] = float(val)
                except ValueError:
                    warnings.append(f"{col} must be a number (got '{val}'); it was ignored.")
        if "POISSONS_RATIO" in values:
            warnings.append("POISSONS_RATIO is 0.35 for every training row, so it cannot affect Mr.")
        return values, warnings

    def _encode(self, frame, cols):
        X = pd.DataFrame(index=frame.index)
        for c in cols:
            if c in CATEGORICAL:
                codes = {v: i for i, v in enumerate(self.categories[c])}
                X[c] = frame[c].map(codes)
            else:
                X[c] = frame[c]
        return X.astype(float)

    # ---------- model per input subset ----------
    def _new_model(self, cols):
        return HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=10,
            l2_regularization=1.0, categorical_features=[c in CATEGORICAL for c in cols],
            random_state=0,
        )

    def get_model(self, cols):
        """Model trained on exactly ``cols`` (cached in memory and on disk)."""
        cols = [c for c in ALL_FEATURES if c in cols]
        key = hashlib.md5(json.dumps(cols).encode()).hexdigest()[:12]
        if key in self._models:
            return self._models[key]
        path = os.path.join(self.cache_dir, f"mr_{key}.joblib")
        if os.path.exists(path):
            entry = joblib.load(path)
        else:
            entry = self._train(cols)
            joblib.dump(entry, path)
        self._models[key] = entry
        return entry

    def _train(self, cols):
        d = self.data
        X, y = self._encode(d, cols), np.log(d[TARGET].to_numpy())
        groups = d[GROUP_COLS].astype(str).agg("_".join, axis=1)
        # 5-fold cross-validation grouped by test section -> expected error
        pred = np.empty_like(y)
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            pred[te] = self._new_model(cols).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        true, est = np.exp(y), np.exp(pred)
        log_resid = y - pred
        cv = {
            "R2": float(1 - ((true - est) ** 2).sum() / ((true - true.mean()) ** 2).sum()),
            "MAE": float(np.abs(true - est).mean()),
            "MAPE_%": float(100 * np.mean(np.abs(true - est) / true)),
            # 10th/90th percentile of the CV error ratio -> ~80% prediction band
            "band_lo": float(np.exp(np.quantile(log_resid, 0.10))),
            "band_hi": float(np.exp(np.quantile(log_resid, 0.90))),
        }
        model = self._new_model(cols).fit(X, y)
        return {"cols": cols, "model": model, "cv": cv}

    # ---------- prediction ----------
    def predict(self, **inputs):
        values, warnings = self.clean_inputs(inputs)
        if not values:
            raise ValueError("Give at least one input value.")
        entry = self.get_model(list(values))
        X = self._encode(pd.DataFrame([values]), entry["cols"])
        mr = float(np.exp(entry["model"].predict(X)[0]))
        cv = entry["cv"]
        return {
            "RESILIENT_MODULUS": mr,
            "range_80pct": (mr * cv["band_lo"], mr * cv["band_hi"]),
            "inputs_used": values,
            "cv_metrics": {k: cv[k] for k in ("R2", "MAE", "MAPE_%")},
            "warnings": warnings,
        }

    def predict_table(self, frame):
        """Predict Mr for every row of a DataFrame; blank cells count as not given.

        Columns that are not model inputs (IDs, measured Mr, ...) are kept as they are.
        """
        inputs = {_norm(f) for f in ALL_FEATURES}
        rows = []
        for _, row in frame.iterrows():
            given = {k: v for k, v in row.items() if _norm(k) in inputs}
            try:
                r = self.predict(**given)
                rows.append({
                    "PREDICTED_MR": r["RESILIENT_MODULUS"],
                    "MR_LOW_80": r["range_80pct"][0], "MR_HIGH_80": r["range_80pct"][1],
                    "INPUTS_USED": ", ".join(r["inputs_used"]),
                    "MODEL_CV_R2": r["cv_metrics"]["R2"],
                    "MODEL_CV_MAE": r["cv_metrics"]["MAE"],
                    "NOTES": " ".join(r["warnings"]),
                })
            except ValueError as e:
                rows.append({"PREDICTED_MR": np.nan, "INPUTS_USED": str(e)})
        return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
