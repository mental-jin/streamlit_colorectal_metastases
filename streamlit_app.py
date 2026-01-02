from __future__ import annotations

import __main__
import json
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
    exists = True
except Exception as e:
    exists = False
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

model_key = st.selectbox("Select model", options=model_keys, index=model_keys.index("RF") if "RF" in model_keys else 0)

threshold = 0.5

model = models[model_key]

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
            val = st.selectbox(f"{c}（categorical）", options=opts, index=opts.index(default_val) if default_val in opts else 0)
            user_values[c] = val
        else:
            val = st.number_input(f"{c}（numerical）", value=float(defaults.get(c, 0.0)))
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
