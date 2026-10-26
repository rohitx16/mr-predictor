# Mr Predictor

Predicts `RESILIENT_MODULUS` (Mr) from any combination of these properties
(same units as `Mr Data.xlsx`):

`MATL_CODE_EXP, LIQUID_LIMIT, PLASTIC_LIMIT, PLASTICITY_INDEX, AASHTO_SOIL_CLASS_EXP,
SPEC_GRAVITY, MAX_LAB_DRY_DENSITY, OPTIMUM_LAB_MOISTURE, RESILIENT_MODULUS_K1,
RESILIENT_MODULUS_K2, RESILIENT_MODULUS_K3, POISSONS_RATIO, HYDRAULIC_CONDUCTIVITY`
plus optional `REPR_THICKNESS` and `LAYER_TYPE_EXP`.

A gradient-boosting model is trained on exactly the inputs you give (the 6,636 rows
of the sheet with a measured Mr), so all values -> all properties used, fewer values ->
only those used. Each new combination takes a few seconds the first time, then is
cached in `model_cache/`.

## Web app (Streamlit)

```
.venv\Scripts\python -m streamlit run streamlit_app.py
```

Enter values in the form (leave unknown ones empty), or upload an Excel/CSV file of samples
(download the template from the app). Deployed from this repo on Streamlit Community Cloud
(`streamlit_app.py`, `requirements.txt`, training rows in `mr_training_data.csv`).

Google Colab: `Mr_Predictor_Colab.ipynb`.

## Command line

```
.venv\Scripts\python predict_mr.py                         # interactive, Enter skips a value
.venv\Scripts\python predict_mr.py LIQUID_LIMIT=25 MAX_LAB_DRY_DENSITY=130
.venv\Scripts\python predict_mr.py --file my_inputs.xlsx   # one prediction per row
.venv\Scripts\python predict_mr.py --list                  # valid material / AASHTO names
```

For `--file`, use the column names above; blank cells count as "not given".

After changing `Mr Data.xlsx` (kept locally, not in the repo), close it in Excel and run `.venv\Scripts\python train_model.py`. That rewrites `mr_training_data.csv`, which the app uses.

## Accuracy (5-fold CV, grouped by test section)

| Inputs | R² | MAE | MAPE |
|---|---|---|---|
| All requested properties | 0.68 | 1.87 | 13.5% |
| All + thickness/layer type | 0.72 | 1.73 | 12.5% |
| K1, K2, K3 only | 0.57 | 2.26 | 15.9% |
| Soil properties only (no K) | 0.59 | 2.26 | 17.9% |

Every prediction prints the CV accuracy for its own input combination and an ~80% range.
`POISSONS_RATIO` is 0.35 in every row, so it has no effect.
