import warnings
warnings.filterwarnings("ignore")

import os
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from uppercase_features import UPPERCASE_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_PATH = os.path.join(BASE_DIR, "sixhos_NC_3.xlsx")
TARGET_COL = "age"

META_COLS = ["ID", "age", "Diagnosis_simple"]

N_SPLITS = 10
RANDOM_STATE = 42

PCA_VAR_THRESHOLD = 0.95


def build_kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
    )


def build_gpr():
    return GaussianProcessRegressor(
        kernel=build_kernel(),
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=RANDOM_STATE,
    )


def calc_metrics(y_true, y_pred):
    r, p = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        "Pearson_r": r,
        "Pearson_p": p,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }


def save_metrics_csv(metrics_dict, out_name, extra_dict=None):
    row = dict(metrics_dict)
    if extra_dict:
        row.update(extra_dict)
    pd.DataFrame([row]).to_csv(
        os.path.join(BASE_DIR, out_name), index=False, encoding="utf-8-sig"
    )
    print(f"已保存: {out_name}")


def save_scatter(y_true, y_pred, title, out_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], "--")
    plt.xlabel("Chronological Age")
    plt.ylabel("Predicted Brain Age")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, out_name), dpi=300)
    plt.close()
    print(f"已保存: {out_name}")


def load_dataset():
    df = pd.read_excel(FILE_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"目标列 {TARGET_COL} 不存在，请检查 Excel。")

    meta_cols_present = [c for c in META_COLS if c in df.columns]
    
    # 只使用36个大写开头的特征
    feature_cols = [c for c in UPPERCASE_FEATURES if c in df.columns]
    
    print(f"使用36个大写开头特征中的 {len(feature_cols)} 个")
    missing = set(UPPERCASE_FEATURES) - set(df.columns)
    if missing:
        print(f"警告: 以下特征在数据中不存在: {missing}")

    if len(feature_cols) == 0:
        raise ValueError("没有可用的特征列，请检查 UPPERCASE_FEATURES 列表。")

    X_all = df[feature_cols].copy()
    y = df[TARGET_COL].values.astype(float)
    return df, X_all, y, meta_cols_present, feature_cols


def pca_to_tables(pca, feature_names):
    n_pc = int(pca.n_components_)
    pc_names = [f"PC{i+1}" for i in range(n_pc)]

    summary_df = pd.DataFrame({
        "PC": pc_names,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    loading_wide_df = pd.DataFrame(loadings, index=feature_names, columns=pc_names)
    loading_wide_df.index.name = "feature"
    loading_wide_df = loading_wide_df.reset_index()

    loading_long_df = loading_wide_df.melt(
        id_vars="feature", var_name="PC", value_name="loading"
    )

    return summary_df, loading_wide_df, loading_long_df, loadings, pc_names


def make_prediction_df(df, y_true, y_pred, fold_assignments=None):
    result_df = pd.DataFrame({
        "age": y_true,
        "predicted_age_cv": y_pred,
        "brain_PAD_cv": y_pred - y_true,
    })

    if "ID" in df.columns:
        result_df.insert(0, "ID", df["ID"].values)

    if "Diagnosis_simple" in df.columns:
        result_df["Diagnosis_simple"] = df["Diagnosis_simple"].values

    if fold_assignments is not None:
        result_df["cv_fold"] = fold_assignments

    return result_df


SCRIPT_TAG = "PCAt_to_GPR_36upper"


def main():
    df, X_all, y, meta_cols_present, feature_cols = load_dataset()

    print("目标3: PCA降维 -> GPR")
    print("原始特征数:", X_all.shape[1])
    print("样本数:", X_all.shape[0])

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    y_pred_cv = np.zeros(len(y), dtype=float)
    fold_assignments = np.zeros(len(y), dtype=int)
    fold_metrics_rows = []
    fold_pc_summary_rows = []
    fold_pc_loading_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_all, y), start=1):
        X_train = X_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_VAR_THRESHOLD, svd_solver="full")),
            ("gpr", build_gpr()),
        ])

        model.fit(X_train, y_train)
        y_pred_fold = model.predict(X_test)

        y_pred_cv[test_idx] = y_pred_fold
        fold_assignments[test_idx] = fold

        fold_metrics = calc_metrics(y_test, y_pred_fold)
        fold_metrics_rows.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **fold_metrics,
        })

        pca = model.named_steps["pca"]
        summary_df, _, loading_long_df, _, _ = pca_to_tables(pca, feature_cols)

        summary_df["fold"] = fold
        summary_df["n_pc_retained"] = int(pca.n_components_)
        fold_pc_summary_rows.append(summary_df)

        loading_long_df["fold"] = fold
        fold_pc_loading_rows.append(loading_long_df)

    overall_metrics = calc_metrics(y, y_pred_cv)

    print("result")
    print(f"Pearson r = {overall_metrics['Pearson_r']:.4f} (p = {overall_metrics['Pearson_p']:.4e})")
    print(f"R^2       = {overall_metrics['R2']:.4f}")
    print(f"MAE       = {overall_metrics['MAE']:.4f} years")
    print(f"RMSE      = {overall_metrics['RMSE']:.4f} years")

    # 计算平均保留的PC数量
    mean_pc_retained = float(pd.concat(fold_pc_summary_rows).groupby('fold')['PC'].count().mean())
    print(f"平均保留PC数: {mean_pc_retained:.1f}")

    pred_df = make_prediction_df(df, y, y_pred_cv, fold_assignments=fold_assignments)
    pred_name = f"{SCRIPT_TAG}_cv_predictions.csv"
    pred_df.to_csv(os.path.join(BASE_DIR, pred_name), index=False, encoding="utf-8-sig")
    print(f"已保存: {pred_name}")

    fold_metrics_df = pd.DataFrame(fold_metrics_rows)
    fold_metrics_name = f"{SCRIPT_TAG}_fold_metrics.csv"
    fold_metrics_df.to_csv(os.path.join(BASE_DIR, fold_metrics_name), index=False, encoding="utf-8-sig")
    print(f"已保存: {fold_metrics_name}")

    save_metrics_csv(
        overall_metrics,
        f"{SCRIPT_TAG}_metrics.csv",
        extra_dict={
            "method": "PCA_transform_36upper_to_GPR",
            "n_samples": len(y),
            "n_raw_features": X_all.shape[1],
            "mean_retained_pcs_cv": mean_pc_retained,
        },
    )

    save_scatter(
        y,
        y_pred_cv,
        f"{SCRIPT_TAG} 10-fold CV | r={overall_metrics['Pearson_r']:.3f}, MAE={overall_metrics['MAE']:.2f}",
        f"{SCRIPT_TAG}_scatter.png",
    )

    fold_pc_summary_df = pd.concat(fold_pc_summary_rows, axis=0, ignore_index=True)
    fold_pc_summary_name = f"{SCRIPT_TAG}_cv_fold_pc_summary.csv"
    fold_pc_summary_df.to_csv(os.path.join(BASE_DIR, fold_pc_summary_name), index=False, encoding="utf-8-sig")
    print(f"已保存: {fold_pc_summary_name}")

    fold_pc_loading_df = pd.concat(fold_pc_loading_rows, axis=0, ignore_index=True)
    fold_pc_loading_name = f"{SCRIPT_TAG}_cv_fold_pc_loadings_long.csv"
    fold_pc_loading_df.to_csv(os.path.join(BASE_DIR, fold_pc_loading_name), index=False, encoding="utf-8-sig")
    print(f"已保存: {fold_pc_loading_name}")

    print("使用全部数据拟合模型......")
    full_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_VAR_THRESHOLD, svd_solver="full")),
        ("gpr", build_gpr()),
    ])
    full_model.fit(X_all, y)

    pca = full_model.named_steps["pca"]
    full_pc_summary_df, full_loading_wide_df, full_loading_long_df, _, pc_names = pca_to_tables(pca, feature_cols)

    full_pc_summary_name = f"{SCRIPT_TAG}_full_pc_summary.csv"
    full_pc_summary_df.to_csv(os.path.join(BASE_DIR, full_pc_summary_name), index=False, encoding="utf-8-sig")
    print(f"已保存: {full_pc_summary_name}")

    full_loading_wide_name = f"{SCRIPT_TAG}_full_pc_loadings_wide.csv"
    full_loading_wide_df.to_csv(os.path.join(BASE_DIR, full_loading_wide_name), index=False, encoding="utf-8-sig")
    print(f"已保存: {full_loading_wide_name}")

    full_loading_long_name = f"{SCRIPT_TAG}_full_pc_loadings_long.csv"
    full_loading_long_df.to_csv(os.path.join(BASE_DIR, full_loading_long_name), index=False, encoding="utf-8-sig")
    print(f"已保存: {full_loading_long_name}")

    X_all_imp = full_model.named_steps["imputer"].transform(X_all)
    X_all_scaled = full_model.named_steps["scaler"].transform(X_all_imp)
    pc_scores = pca.transform(X_all_scaled)

    pc_scores_df = pd.DataFrame(pc_scores, columns=pc_names)
    if "ID" in df.columns:
        pc_scores_df.insert(0, "ID", df["ID"].values)
    pc_scores_df["age"] = y
    if "Diagnosis_simple" in df.columns:
        pc_scores_df["Diagnosis_simple"] = df["Diagnosis_simple"].values

    pc_scores_name = f"{SCRIPT_TAG}_full_pc_scores.csv"
    pc_scores_df.to_csv(os.path.join(BASE_DIR, pc_scores_name), index=False, encoding="utf-8-sig")
    print(f"已保存: {pc_scores_name}")

    artifact = {
        "model": full_model,
        "feature_cols": feature_cols,
        "meta_cols_present": meta_cols_present,
        "file_path": FILE_PATH,
        "method": "PCA_transform_36upper_to_GPR",
        "n_pc_retained": int(pca.n_components_),
    }
    model_name = f"{SCRIPT_TAG}_model.joblib"
    joblib.dump(artifact, os.path.join(BASE_DIR, model_name))
    print(f"已保存: {model_name}")

    print("finish!!!")


if __name__ == "__main__":
    main()
