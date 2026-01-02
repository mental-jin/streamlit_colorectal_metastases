from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class ColumnSelector(TransformerMixin, BaseEstimator):
	def __init__(self, indices: np.ndarray):
		self.indices = np.asarray(indices, dtype=int)

	def fit(self, X: Any, y: Any = None):
		return self

	def transform(self, X: Any):
		if sp.issparse(X):
			return X[:, self.indices]
		return np.asarray(X)[:, self.indices]


class ToDense(TransformerMixin, BaseEstimator):
	def fit(self, X: Any, y: Any = None):
		return self

	def transform(self, X: Any):
		if sp.issparse(X):
			return X.toarray()
		return np.asarray(X)


def _safe_float32_matrix(X: Any) -> Any:
	if sp.issparse(X):
		X = X.tocsr(copy=False)
		return X.astype(np.float32) if X.dtype != np.float32 else X
	X = np.asarray(X)
	return X.astype(np.float32) if X.dtype != np.float32 else X


def _load_training_artifacts(load_dir: Path) -> tuple[ColumnTransformer, dict[str, Pipeline]]:
	art_dir = load_dir / "artifacts"
	pre_path = art_dir / "preprocessor.joblib"
	if not pre_path.exists():
		raise FileNotFoundError(f"找不到预处理器文件: {pre_path}（请先用 train 模式运行并保存 artifacts）")
	pre = joblib.load(pre_path)
	models: dict[str, Pipeline] = {}
	for p in art_dir.glob("model_*.joblib"):
		key = p.stem.replace("model_", "", 1)
		models[key] = joblib.load(p)
	return pre, models


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
	try:
		names = preprocessor.get_feature_names_out()
		return [str(x) for x in names]
	except Exception:
		return []


def _to_jsonable(obj: Any) -> Any:
	"""Best-effort conversion for printing (avoid crashes on non-JSON objects)."""
	try:
		json.dumps(obj)
		return obj
	except Exception:
		return str(obj)


def _read_one_row(*, input_xlsx: Path, target_col: str, row_index: int) -> pd.DataFrame:
	if not input_xlsx.exists():
		raise FileNotFoundError(f"找不到输入文件: {input_xlsx}")
	df = pd.read_excel(input_xlsx)
	if target_col in df.columns:
		X_df = df.drop(columns=[target_col])
	else:
		X_df = df

	if row_index < 0 or row_index >= X_df.shape[0]:
		raise IndexError(f"row_index 越界：{row_index}，有效范围是 [0, {X_df.shape[0]-1}]")

	return X_df.iloc[[row_index]].copy()


def _print_one_row_verbose(*, input_xlsx: Path, target_col: str, row_index: int, pre: ColumnTransformer, model: Pipeline) -> None:
	X_row_df = _read_one_row(input_xlsx=input_xlsx, target_col=target_col, row_index=row_index)
	row_series = X_row_df.iloc[0]

	print(f"[INPUT] file={input_xlsx} row={row_index} n_cols={X_row_df.shape[1]}")
	for c in X_row_df.columns:
		v = row_series[c]
		if pd.isna(v):
			vs = "<NA>"
		elif isinstance(v, (float, np.floating)):
			vs = f"{float(v):.6g}"
		else:
			vs = str(v)
		print(f"  {c} = {vs}")

	Xt = _safe_float32_matrix(pre.transform(X_row_df))
	all_names = get_feature_names(pre)
	if not all_names:
		all_names = [f"f{i}" for i in range(int(Xt.shape[1]))]

	if "select" in model.named_steps and hasattr(model.named_steps["select"], "indices"):
		selected_idx = np.asarray(model.named_steps["select"].indices, dtype=int)
	else:
		selected_idx = np.arange(int(Xt.shape[1]), dtype=int)

	selected_idx = selected_idx[(selected_idx >= 0) & (selected_idx < int(Xt.shape[1]))]
	selected_names = [all_names[i] if i < len(all_names) else f"f{i}" for i in selected_idx]

	# Non-zero values among selected features (avoid printing大量0)
	nonzero_map: dict[int, float] = {}
	if sp.issparse(Xt):
		row = Xt.tocsr()[0]
		for j, val in zip(row.indices.tolist(), row.data.tolist()):
			nonzero_map[int(j)] = float(val)
	else:
		arr = np.asarray(Xt).ravel()
		nz = np.flatnonzero(arr)
		for j in nz.tolist():
			nonzero_map[int(j)] = float(arr[j])

	print(
		f"[FEATURES_USED] n_after_preprocess={int(Xt.shape[1])} selected_k={int(selected_idx.size)} "
		f"nonzero_in_selected={sum(1 for i in selected_idx if int(i) in nonzero_map)} (仅打印非零项)"
	)
	printed = 0
	for idx, name in zip(selected_idx.tolist(), selected_names):
		idx_i = int(idx)
		if idx_i in nonzero_map:
			print(f"  {name} = {nonzero_map[idx_i]:.6g}")
			printed += 1
	if printed == 0:
		print("  <all selected features are zero>")

	# Model hyper-parameters
	try:
		mdl = model.named_steps.get("model", model)
		params = mdl.get_params(deep=False) if hasattr(mdl, "get_params") else {}
		params = {k: _to_jsonable(v) for k, v in params.items()}
		print("[MODEL_PARAMS] " + json.dumps(params, ensure_ascii=False, indent=2))
	except Exception:
		pass


def predict_one_row(*, input_xlsx: Path, target_col: str, row_index: int, pre: ColumnTransformer, model: Pipeline, threshold: float) -> tuple[float, int]:
	X_row_df = _read_one_row(input_xlsx=input_xlsx, target_col=target_col, row_index=row_index)
	X_row = _safe_float32_matrix(pre.transform(X_row_df))
	proba = float(model.predict_proba(X_row)[:, 1].ravel()[0])
	pred = int(proba >= threshold)
	return proba, pred


def main() -> int:
	parser = argparse.ArgumentParser(description="加载训练 artifacts，对Excel指定行进行预测（并输出输入/特征/参数信息）")
	parser.add_argument("--input", default=str(Path("mice") / "combined_imputed.xlsx"), help="输入Excel文件")
	parser.add_argument("--target", default="Metastasis", help="因变量列名（0/1；若不存在则忽略）")
	parser.add_argument("--load_dir", required=True, help="训练输出目录（包含 artifacts/），例如 ml_results/smoke_run4")
	parser.add_argument("--predict_model", default="RF", help="要使用的模型key，例如 RF/LR/DT/KNN/SVM/NB/SGBT/NNET/XGB")
	parser.add_argument("--predict_row", type=int, default=0, help="要预测的行号（0-based，与 pandas iloc 一致）")
	parser.add_argument("--threshold", type=float, default=0.5, help="将概率转为0/1的阈值，默认0.5")
	args = parser.parse_args()

	load_dir = Path(str(args.load_dir))
	pre, models = _load_training_artifacts(load_dir)
	model_key = str(args.predict_model).strip()
	if model_key not in models:
		available = ", ".join(sorted(models.keys()))
		raise KeyError(f"在 {load_dir / 'artifacts'} 中找不到模型 {model_key}。可用模型：{available}")

	input_xlsx = Path(str(args.input))
	target_col = str(args.target)
	row_index = int(args.predict_row)
	threshold = float(args.threshold)
	model = models[model_key]

	_print_one_row_verbose(
		input_xlsx=input_xlsx,
		target_col=target_col,
		row_index=row_index,
		pre=pre,
		model=model,
	)
	proba, pred = predict_one_row(
		input_xlsx=input_xlsx,
		target_col=target_col,
		row_index=row_index,
		pre=pre,
		model=model,
		threshold=threshold,
	)
	print(f"[PREDICT] model={model_key} row={row_index} threshold={threshold}")
	print(f"[PREDICT] P(y=1)={proba:.6f}  y_hat={pred}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())