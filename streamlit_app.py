from __future__ import annotations

import __main__
import json
import re
import tempfile
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import predict_one as po


def _col_key(name: str) -> str:
    """Normalize a column name for lookup.

    Examples:
    - "Carcinoma nodule" -> "carcinoma_nodule"
    - "MSI-H" -> "msi_h"
    """

    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _pretty_var_name(name: str) -> str:
    """Pretty display name: underscores -> spaces, remove trailing brackets/parentheses content."""

    s = str(name).replace("_", " ")
    # Remove any trailing type hints like "（categorical）"/"（numerical）" or "(xxx)".
    s = re.sub(r"\s*[（(].*?[）)]\s*$", "", s)
    return s


# Categorical value -> English label mapping (display only; raw value is preserved).
_CAT_VALUE_LABELS: dict[str, dict[str, str]] = {
    _col_key("Carcinoma_nodule"): {
        "0": "None detected",
        "1": "1–3 carcinoma nodules (mild)",
        "2": "≥4 carcinoma nodules (severe)",
    },
    _col_key("Differentiation_grade"): {
        "1": "Well differentiated",
        "2": "Well-to-moderately differentiated",
        "3": "Moderately differentiated",
        "4": "Moderately-to-poorly differentiated",
        "5": "Poorly differentiated",
        "6": "Poorly/undifferentiated",
    },
    _col_key("Vascular_invasion"): {"1": "Present", "0": "Absent"},
    _col_key("Perineural_invasion"): {"1": "Present", "0": "Absent"},
    _col_key("MLH1"): {"1": "Positive", "0": "Negative"},
    _col_key("MSH2"): {"1": "Positive", "0": "Negative"},
    _col_key("MSH6"): {"1": "Positive", "0": "Negative"},
    _col_key("PMS2"): {"1": "Positive", "0": "Negative"},
    _col_key("Family_history"): {"1": "Yes", "0": "No"},
    _col_key("Colonic_obstruction"): {"1": "Yes", "0": "No"},
    _col_key("Hypertension"): {"1": "Yes", "0": "No"},
    _col_key("Diabetes"): {"1": "Yes", "0": "No"},
    _col_key("Coronary_artery_disease"): {"1": "Yes", "0": "No"},
    _col_key("Hyperlipidemia"): {"1": "Yes", "0": "No"},
    _col_key("BRAF_mutant"): {"1": "Yes", "0": "No"},
    _col_key("KRAS_mutant"): {"1": "Yes", "0": "No"},
    _col_key("NRAS_mutant"): {"1": "Yes", "0": "No"},
    _col_key("MSI-H"): {"1": "Yes", "0": "No"},
}


def _cat_format_func(col_name: str):
    mapping = _CAT_VALUE_LABELS.get(_col_key(col_name), {})

    def _fmt(x: Any) -> str:
        sx = str(x)
        return mapping.get(sx, sx)

    return _fmt


# Units for numerical (continuous) variables (display only).
_NUM_UNITS: dict[str, str] = {
    _col_key("IgA"): "g/L",
    _col_key("IgG"): "g/L",
    _col_key("IgM"): "g/L",
    _col_key("IgE"): "kU/L",
    _col_key("Albumin"): "g/L",
    _col_key("ALP"): "U/L",
    _col_key("ALT"): "U/L",
    _col_key("AST"): "U/L",
    _col_key("Direct_bilirubin"): "μmol/L",
    _col_key("Glucose"): "mmol/L",
    _col_key("Potassium"): "mmol/L",
    _col_key("LDH"): "U/L",
    _col_key("Prealbumin"): "mg/L",
    _col_key("RBP"): "mg/L",
    _col_key("Total_bile_acids"): "μmol/L",
    _col_key("Total_bilirubin"): "μmol/L",
    _col_key("Triglycerides"): "mmol/L",
    _col_key("Total_protein"): "g/L",
    _col_key("mAST"): "U/L",
    _col_key("GLDH"): "U/L",
    _col_key("A_G_ratio"): "%",
    _col_key("Globulin"): "%",
    _col_key("Total_cholesterol"): "%",
    _col_key("Ast_alt_ratio"): "%",
    _col_key("Indirect_bilirubin"): "μmol/L",
    _col_key("EGFR"): "mL/min/1.73m²",
    _col_key("IL10"): "pg/mL",
    _col_key("IL4"): "pg/mL",
    _col_key("IL5"): "pg/mL",
    _col_key("IL8"): "pg/mL",
    _col_key("IL12_p70"): "pg/mL",
    _col_key("IFN_gamma"): "pg/mL",
    _col_key("IFN_alpha"): "pg/mL",
    _col_key("IL1_beta"): "pg/mL",
    _col_key("IL6"): "pg/mL",
    _col_key("IL17"): "pg/mL",
    _col_key("TNF_alpha"): "pg/mL",
    _col_key("IL2"): "pg/mL",
    _col_key("Total_T_lymphocytes"): "%",
    _col_key("CD4_T_cells"): "%",
    _col_key("CD8_T_cells"): "%",
    _col_key("Total_B_lymphocytes"): "%",
    _col_key("NK_cells"): "%",
    _col_key("CD29_pos_helper_T_cells"): "%",
    _col_key("Early_activated_T_cells"): "%",
    _col_key("Regulatory_T_cells"): "%",
    _col_key("CD8_CD28_pos"): "%",
    _col_key("CD8_CD25_over_CD8_percent"): "%",
    _col_key("CD4_CD25_over_CD4_percent"): "%",
    _col_key("CD4_count"): "个/ul",
    _col_key("CD8_count"): "个/ul",
    _col_key("CD19_count"): "个/ul",
    _col_key("NK_count"): "个/ul",
    _col_key("CD3_count"): "个/ul",
    _col_key("CD4_CD45RO_memory_percent_of_helper_T"): "%",
    _col_key("CD8_CD45RO_memory_percent_of_cytotoxic_T"): "%",
    _col_key("CD4_CD45RA_naive_percent_of_helper_T"): "%",
    _col_key("CD8_CD45RA_naive_percent_of_cytotoxic_T"): "%",
    _col_key("CD3_HLA_DR_pos"): "%",
    _col_key("VitA"): "μmol/L",
    _col_key("VitB1"): "μg/L",
    _col_key("VitB2"): "μg/L",
    _col_key("VitB6"): "μg/L",
    _col_key("VitC"): "μmol/L",
    _col_key("VitE"): "μmol/L",
    _col_key("SCCA"): "U/mL",
    _col_key("CA724"): "U/mL",
    _col_key("NSE"): "U/mL",
    _col_key("CYFRA21_1"): "U/mL",
    _col_key("CA242"): "U/mL",
    _col_key("CA50"): "U/mL",
    _col_key("CA199"): "U/mL",
    _col_key("CEA"): "U/mL",
    _col_key("AFP"): "U/mL",
    _col_key("CA153"): "U/mL",
    _col_key("CA125"): "U/mL",
    _col_key("WBC"): "10^9/L",
    _col_key("RBC"): "10^12/L",
    _col_key("Neutrophil_percent"): "%",
    _col_key("Hemoglobin"): "g/L",
    _col_key("Lymphocyte_percent"): "%",
    _col_key("Hematocrit"): "%",
    _col_key("Monocyte_percent"): "%",
    _col_key("MCV"): "fL",
    _col_key("MCH"): "pg",
    _col_key("Neutrophil_absolute"): "10^9/L",
    _col_key("RDW_CV"): "%",
    _col_key("Lymphocyte_absolute"): "10^9/L",
    _col_key("Platelet_count"): "10^9/L",
    _col_key("Monocyte_absolute"): "10^9/L",
    _col_key("MPV"): "fL",
    _col_key("Plateletcrit"): "%",
    _col_key("PDW"): "%",
    _col_key("MCHC"): "g/L",
    _col_key("CRP"): "mg/L",
    _col_key("Iron"): "μmol/L",
    _col_key("Reticulocyte_percent"): "%",
    _col_key("Tumor volume"): "cm³",
    _col_key("Tumor size"): "cm",
    _col_key("Ki67"): "%",
}


def _display_label(col_name: str, *, is_categorical: bool) -> str:
    pretty = _pretty_var_name(col_name)
    if is_categorical:
        return pretty
    unit = _NUM_UNITS.get(_col_key(col_name))
    return f"{pretty} ({unit})" if unit else pretty


def _inject_custom_pickle_types() -> None:
    """Ensure joblib can unpickle custom transformers saved under __main__."""
    __main__.ColumnSelector = po.ColumnSelector
    __main__.ToDense = po.ToDense


@st.cache_resource
def load_artifacts(load_dir: str) -> tuple[Any, dict[str, Any]]:
    _inject_custom_pickle_types()
    pre, models = po._load_training_artifacts(Path(load_dir))
    return pre, models


def _iter_column_transformers(pre: Any):
    """Yield (name, transformer, cols) from a ColumnTransformer.

    Prefer fitted transformers_ (contains fitted OneHotEncoder with categories_).
    """
    trs = getattr(pre, "transformers_", None)
    if trs:
        for t in trs:
            # transformers_ may include ('remainder', 'drop', ...) entries
            if isinstance(t, tuple) and len(t) == 3:
                yield t
        return
    for t in getattr(pre, "transformers", []) or []:
        if isinstance(t, tuple) and len(t) == 3:
            yield t


def _get_transformer_columns(pre: Any) -> tuple[set[str], set[str]]:
    """Return (numeric_cols, categorical_cols) from the saved ColumnTransformer."""
    num_cols: set[str] = set()
    cat_cols: set[str] = set()
    for name, _trans, cols in _iter_column_transformers(pre):
        if not isinstance(cols, list):
            continue
        if name == "num":
            num_cols.update([str(c) for c in cols])
        elif name == "cat":
            cat_cols.update([str(c) for c in cols])
    return num_cols, cat_cols


def _find_step(obj: Any, class_name: str) -> Any | None:
    """Best-effort finder for a step inside a sklearn Pipeline-like object."""
    if obj is None:
        return None
    if obj.__class__.__name__ == class_name:
        return obj
    if hasattr(obj, "steps"):
        for _name, step in getattr(obj, "steps", []):
            found = _find_step(step, class_name)
            if found is not None:
                return found
    return None


def _categorical_options(pre: Any) -> dict[str, list[str]]:
    """Map categorical raw column -> list of known categories (strings)."""
    options: dict[str, list[str]] = {}
    cat_transformer = None
    cat_cols: list[str] = []
    for name, trans, cols in _iter_column_transformers(pre):
        if name == "cat" and isinstance(cols, list):
            cat_transformer = trans
            cat_cols = [str(c) for c in cols]
            break
    if cat_transformer is None or not cat_cols:
        return options

    enc = _find_step(cat_transformer, "OneHotEncoder")
    if enc is None or not hasattr(enc, "categories_"):
        return options

    try:
        cats = enc.categories_
        for col, col_cats in zip(cat_cols, cats):
            options[col] = [str(x) for x in col_cats.tolist()]
    except Exception:
        pass
    return options


def _load_defaults_from_text(*, base_dir: Path) -> dict[str, Any]:
    """Load raw-column defaults from a text file (JSON) without model introspection."""
    defaults_path = base_dir / "row0_defaults.json"
    data = json.loads(defaults_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Invalid defaults format in {defaults_path}: expected JSON object")
    return data


def _required_raw_columns_for_model(pre: Any, model: Any) -> list[str]:
    """Derive the minimal set of raw columns used by the model's selected features."""
    feature_names = po.get_feature_names(pre)
    if not feature_names:
        # fallback: if names are unavailable, require all raw inputs
        return [str(x) for x in getattr(pre, "feature_names_in_", [])]

    if hasattr(model, "named_steps") and "select" in model.named_steps and hasattr(model.named_steps["select"], "indices"):
        selected_idx = np.asarray(model.named_steps["select"].indices, dtype=int)
    else:
        selected_idx = np.arange(len(feature_names), dtype=int)

    selected_idx = selected_idx[(selected_idx >= 0) & (selected_idx < len(feature_names))]
    selected_feature_names = [feature_names[int(i)] for i in selected_idx.tolist()]

    num_cols, cat_cols = _get_transformer_columns(pre)
    cat_cols_list = sorted(cat_cols, key=len, reverse=True)  # prefer longest match

    required: set[str] = set()
    for fn in selected_feature_names:
        # Examples: num__Creatinine ; cat__Sex_M
        if fn.startswith("num__"):
            required.add(fn[len("num__") :])
            continue
        if fn.startswith("cat__"):
            rest = fn[len("cat__") :]
            # Find the categorical raw column whose name prefixes rest + "_"
            matched = None
            for c in cat_cols_list:
                if rest == c or rest.startswith(c + "_"):
                    matched = c
                    break
            if matched is not None:
                required.add(matched)
            continue

        # Fallback: if name doesn't have prefixes, try raw-col match
        if fn in num_cols or fn in cat_cols:
            required.add(fn)

    # Keep stable ordering by feature_names_in_
    order = [str(x) for x in getattr(pre, "feature_names_in_", [])]
    return [c for c in order if c in required]


def _build_full_row_df(*, pre: Any, user_values: dict[str, Any], defaults: dict[str, Any]) -> pd.DataFrame:
    cols = [str(x) for x in getattr(pre, "feature_names_in_", [])]
    row: dict[str, Any] = {}
    for c in cols:
        if c in user_values and user_values[c] is not None:
            row[c] = user_values[c]
        else:
            row[c] = defaults.get(c, 0.0)
    return pd.DataFrame([row], columns=cols)


def _df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def _auto_export_categorical_levels(pre: Any, *, output_path: Path) -> None:
    """Export fitted OneHotEncoder categories_ to CSV (best-effort, silent)."""
    try:
        cat_trans = None
        cat_cols: list[str] | None = None
        for name, trans, cols in _iter_column_transformers(pre):
            if name == "cat" and isinstance(cols, list):
                cat_trans = trans
                cat_cols = [str(c) for c in cols]
                break
        if cat_trans is None or not cat_cols:
            return

        enc = _find_step(cat_trans, "OneHotEncoder")
        if enc is None or not hasattr(enc, "categories_"):
            return

        rows: list[dict[str, Any]] = []
        for col, cats in zip(cat_cols, enc.categories_):
            cats_list = [str(x) for x in getattr(cats, "tolist", lambda: list(cats))()]
            rows.append({"column": col, "n_categories": len(cats_list), "categories": "|".join(cats_list)})
        out_df = pd.DataFrame(rows).sort_values(["n_categories", "column"], ascending=[False, True])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except Exception:
        # Intentionally silent: user requested no visible UI changes.
        return


st.set_page_config(page_title="Metastasis Predictor", layout="centered")

st.title("Predicting the occurrence of metastases after colorectal cancer surgery")

load_dir = str(Path("ml_results") / "smoke_run_jlx")

try:
    pre, models = load_artifacts(load_dir)
    _auto_export_categorical_levels(pre, output_path=Path(__file__).resolve().parent / "categorical_levels_from_encoder.csv")
    model_keys = sorted(models.keys())
    
    # Define the desired model order with SGBT as optimal
    desired_order = ["SGBT", "RF", "SVM", "XGB", "LR", "DT", "NNET", "NB", "KNN"]
    ordered_keys = [m for m in desired_order if m in model_keys]
    # Add any remaining models not in desired_order
    ordered_keys.extend([m for m in model_keys if m not in desired_order])
    model_keys = ordered_keys
    
    # Create display labels with optimal model annotation
    model_labels = [f"{m} (Optimal model)" if m == "SGBT" else m for m in model_keys]
    
    exists = True
except Exception as e:
    exists = False
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

model_key = st.selectbox("Select model", options=model_labels, index=0)

# Extract actual model key from label (remove " (Optimal model)" suffix if present)
actual_model_key = model_key.replace(" (Optimal model)", "")

threshold = 0.5

model = models[actual_model_key]

defaults = _load_defaults_from_text(base_dir=Path(__file__).resolve().parent)
num_cols, cat_cols = _get_transformer_columns(pre)
cat_options = _categorical_options(pre)

required_cols = _required_raw_columns_for_model(pre, model)
st.caption(f"The model will use some of the preprocessed features; the current inference requires input for {len(required_cols)} raw columns.")

user_values: dict[str, Any] = {}
with st.form("input_form"):
    for c in required_cols:
        if c in cat_cols:
            opts = cat_options.get(c, [])
            default_val = str(defaults.get(c, ""))
            # Ensure default is in options if possible
            if default_val and default_val not in opts:
                opts = [default_val] + opts
            if not opts:
                # If encoder categories aren't available, fall back to a single safe option.
                opts = [default_val] if default_val != "" else [""]
            val = st.selectbox(
                _display_label(c, is_categorical=True),
                options=opts,
                index=opts.index(default_val) if default_val in opts else 0,
                format_func=_cat_format_func(c),
            )
            user_values[c] = val
        else:
            val = st.number_input(_display_label(c, is_categorical=False), value=float(defaults.get(c, 0.0)))
            user_values[c] = float(val)

    submit = st.form_submit_button("Predict")

if submit:
    X_df = _build_full_row_df(pre=pre, user_values=user_values, defaults=defaults)

    # Internal: generate an Excel file for predict_one.py to consume
    excel_bytes = _df_to_excel_bytes(X_df)

    # Run prediction via the same logic used in predict_one.py (reads from xlsx)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(excel_bytes)
        tmp_path = Path(tmp.name)

    try:
        proba, pred = po.predict_one_row(
            input_xlsx=tmp_path,
            target_col="Metastasis",
            row_index=0,
            pre=pre,
            model=model,
            threshold=threshold,
        )
        st.subheader("Prediction Results")

        desc = f"The probability of metastasis is {proba*100:.2f}%."
        st.markdown(f'<h3 style="font-weight: 400; margin: 0;">{escape(desc)}</h3>', unsafe_allow_html=True)
        st.progress(min(max(float(proba), 0.0), 1.0))
        st.caption(f"Model: {model_key}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
