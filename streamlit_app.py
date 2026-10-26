"""Streamlit app: predict resilient modulus (Mr) from any available properties.

Run locally:  streamlit run streamlit_app.py
"""
import io

import pandas as pd
import streamlit as st

from mr_model import ALL_FEATURES, CATEGORICAL, MrPredictor

st.set_page_config(page_title="Mr Predictor", page_icon="🛣️", layout="wide")

# label, number format, step
NUMERIC_FIELDS = {
    "LIQUID_LIMIT": ("Liquid limit (LL)", "%.1f", 1.0),
    "PLASTIC_LIMIT": ("Plastic limit (PL)", "%.1f", 1.0),
    "PLASTICITY_INDEX": ("Plasticity index (PI)", "%.1f", 1.0),
    "SPEC_GRAVITY": ("Specific gravity", "%.3f", 0.01),
    "MAX_LAB_DRY_DENSITY": ("Max lab dry density", "%.1f", 1.0),
    "OPTIMUM_LAB_MOISTURE": ("Optimum lab moisture", "%.1f", 0.5),
    "RESILIENT_MODULUS_K1": ("K1", "%.3f", 10.0),
    "RESILIENT_MODULUS_K2": ("K2", "%.3f", 0.01),
    "RESILIENT_MODULUS_K3": ("K3", "%.3f", 0.01),
    "POISSONS_RATIO": ("Poisson's ratio", "%.2f", 0.01),
    "HYDRAULIC_CONDUCTIVITY": ("Hydraulic conductivity", "%.6f", 0.0001),
    "REPR_THICKNESS": ("Layer thickness (optional extra)", "%.1f", 0.5),
}
CATEGORY_LABELS = {
    "MATL_CODE_EXP": "Material (MATL_CODE_EXP)",
    "AASHTO_SOIL_CLASS_EXP": "AASHTO soil class",
    "LAYER_TYPE_EXP": "Layer type (optional extra)",
}
GROUPS = {
    "Material & classification": ["MATL_CODE_EXP", "AASHTO_SOIL_CLASS_EXP", "LAYER_TYPE_EXP", "REPR_THICKNESS"],
    "Index & compaction properties": ["LIQUID_LIMIT", "PLASTIC_LIMIT", "PLASTICITY_INDEX", "SPEC_GRAVITY",
                                      "MAX_LAB_DRY_DENSITY", "OPTIMUM_LAB_MOISTURE"],
    "Constitutive & other": ["RESILIENT_MODULUS_K1", "RESILIENT_MODULUS_K2", "RESILIENT_MODULUS_K3",
                             "HYDRAULIC_CONDUCTIVITY", "POISSONS_RATIO"],
}


@st.cache_resource(show_spinner="Loading training data...")
def get_predictor():
    return MrPredictor()


p = get_predictor()


def input_widget(col):
    if col in CATEGORICAL:
        return st.selectbox(CATEGORY_LABELS[col], p.categories[col], index=None,
                            placeholder="Not given", key=col)
    label, fmt, step = NUMERIC_FIELDS[col]
    return st.number_input(label, value=None, format=fmt, step=step, placeholder="Not given", key=col)


def template_bytes():
    template = pd.DataFrame(columns=["SAMPLE_ID"] + ALL_FEATURES)
    template.loc[0] = {"SAMPLE_ID": "example (all values)", "MATL_CODE_EXP": "Crushed Stone",
                       "LIQUID_LIMIT": 0, "PLASTIC_LIMIT": 0, "AASHTO_SOIL_CLASS_EXP": "A-1-a",
                       "SPEC_GRAVITY": 2.656, "MAX_LAB_DRY_DENSITY": 135, "OPTIMUM_LAB_MOISTURE": 7,
                       "RESILIENT_MODULUS_K1": 900.411, "RESILIENT_MODULUS_K2": 0.626,
                       "RESILIENT_MODULUS_K3": -0.123, "HYDRAULIC_CONDUCTIVITY": 0.05054}
    template.loc[1] = {"SAMPLE_ID": "example (only some values)", "MATL_CODE_EXP": "Fine-Grained Soils: Clay",
                       "LIQUID_LIMIT": 45, "PLASTIC_LIMIT": 20, "MAX_LAB_DRY_DENSITY": 105}
    allowed = pd.concat({c: pd.Series(p.categories[c]) for c in CATEGORICAL}, axis=1)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        template.to_excel(xw, sheet_name="Inputs", index=False)
        allowed.to_excel(xw, sheet_name="Allowed values", index=False)
    return buf.getvalue()


st.title("🛣️ Resilient Modulus (Mr) Predictor")
st.caption(f"Gradient-boosting model trained on {len(p.data):,} LTPP layers with measured Mr. "
           "Enter whatever properties you have: the model is trained on exactly those inputs.")

tab_single, tab_file, tab_about = st.tabs(["Enter values", "Upload a file", "About the model"])

with tab_single:
    with st.form("single"):
        cols = st.columns(3)
        values = {}
        for box, (title, fields) in zip(cols, GROUPS.items()):
            with box:
                st.subheader(title)
                for f in fields:
                    values[f] = input_widget(f)
        submitted = st.form_submit_button("Predict Mr", type="primary", use_container_width=True)

    if submitted:
        given = {k: v for k, v in values.items() if v is not None}
        if not given:
            st.warning("Enter at least one value.")
        else:
            with st.spinner("Training the model for this input combination (first time only, a few seconds)..."):
                r = p.predict(**given)
            lo, hi = r["range_80pct"]
            cv = r["cv_metrics"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted RESILIENT_MODULUS", f"{r['RESILIENT_MODULUS']:.2f}")
            m2.metric("Likely range (80%)", f"{lo:.2f} – {hi:.2f}")
            m3.metric("Accuracy of these inputs (CV R²)", f"{cv['R2']:.2f}",
                      help=f"Cross-validated on test sections not seen in training: "
                           f"MAE {cv['MAE']:.2f}, MAPE {cv['MAPE_%']:.1f}%")
            st.write(f"**Inputs used ({len(r['inputs_used'])}):** " + ", ".join(r["inputs_used"]))
            for w in r["warnings"]:
                st.info(w)
            if cv["R2"] < 0.4:
                st.warning("Few informative inputs were given, so this prediction is rough. "
                           "Adding K1–K3 or the soil index properties improves it a lot.")

with tab_file:
    st.markdown(
        "1. Download the template, fill in one sample per row and leave cells **blank** for values you don't have.\n"
        "2. Extra columns (e.g. `SAMPLE_ID`, a measured `RESILIENT_MODULUS`) are kept in the results.\n"
        "3. Upload the file (.xlsx or .csv) and download the predictions.")
    st.download_button("Download input template (.xlsx)", template_bytes(), "Mr_input_template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    up = st.file_uploader("Upload your values", type=["xlsx", "xls", "csv"])
    if up is not None:
        frame = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        frame = frame.dropna(how="all")
        with st.spinner(f"Predicting {len(frame)} rows..."):
            out = p.predict_table(frame)
        show = [c for c in ["SAMPLE_ID", "RESILIENT_MODULUS"] if c in out] + \
               ["PREDICTED_MR", "MR_LOW_80", "MR_HIGH_80", "MODEL_CV_R2", "INPUTS_USED", "NOTES"]
        st.dataframe(out[[c for c in show if c in out]].round(3), use_container_width=True)
        buf = io.BytesIO()
        out.to_excel(buf, index=False)
        st.download_button("Download predictions (.xlsx)", buf.getvalue(),
                           up.name.rsplit(".", 1)[0] + "_predicted.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary")

with tab_about:
    st.markdown("""
**How it works.** For each combination of inputs, a gradient-boosting model is trained on exactly those
columns using the rows of the Mr data sheet that have a measured Mr (log-transformed target). Give all values
and every property is used; give some and only those are used. Models are cached after first use.

**Accuracy** (5-fold cross-validation, grouped by LTPP test section):

| Inputs | R² | MAE | MAPE |
|---|---|---|---|
| All requested properties | 0.68 | 1.87 | 13.5% |
| All + thickness/layer type | 0.72 | 1.73 | 12.5% |
| K1, K2, K3 only | 0.57 | 2.26 | 15.9% |
| Soil properties only (no K) | 0.59 | 2.26 | 17.9% |
| Only 1–2 properties | 0.1–0.3 | 3–3.6 | 25%+ |

**Notes.** Poisson's ratio is 0.35 in every training row, so it cannot change the prediction.
Mr also depends on layer thickness/position, so the same material can have different Mr values;
that limits accuracy to roughly ±13% even with all inputs. Values use the same units as the Mr data sheet.
""")
