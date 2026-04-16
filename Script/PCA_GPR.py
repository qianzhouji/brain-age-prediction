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
    feature_cols = [c for c in df.columns if c not in meta_cols_present]

    if len(feature_cols) == 0:
        raise ValueError("没有可用的特征列，请检查 META_COLS 是否写对。")

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


SCRIPT_TAG = "PCA_GPR"

# 特征筛选配置：
# score可选: "communality" / "max_abs_loading" / "weighted_abs_loading"
FEATURE_SCORE_METHOD = "communality"
# mode可选: "top_ratio" / "top_k"
FEATURE_SELECT_MODE = "top_ratio"
TOP_RATIO = 0.50
TOP_K = 20


def build_feature_score_df(feature_names, loadings, explained_variance_ratio):
    communality = np.sum(loadings ** 2, axis=1)
    max_abs_loading = np.max(np.abs(loadings), axis=1)
    weighted_abs_loading = np.sum(np.abs(loadings) * explained_variance_ratio, axis=1)

    score_df = pd.DataFrame({
        "feature": feature_names,
        "communality": communality,
        "max_abs_loading": max_abs_loading,
        "weighted_abs_loading": weighted_abs_loading,
    })
    return score_df.sort_values(FEATURE_SCORE_METHOD, ascending=False).reset_index(drop=True)


def select_features(score_df):
    n_features = len(score_df)

    if FEATURE_SELECT_MODE == "top_ratio":
        k = max(1, int(math.ceil(n_features * TOP_RATIO)))
    elif FEATURE_SELECT_MODE == "top_k":
        k = min(max(1, int(TOP_K)), n_features)
    else:
        raise ValueError("FEATURE_SELECT_MODE 只支持 'top_ratio' 或 'top_k'。")

    selected_df = score_df.head(k).copy()
    filtered_df = score_df.iloc[k:].copy()
    selected_features = selected_df["feature"].tolist()
    filtered_features = filtered_df["feature"].tolist()
    return selected_features, filtered_features, selected_df, filtered_df


def main():
    df, X_all, y, meta_cols_present, feature_cols = load_dataset()

    print("方法2: PCA辅助筛选原始特征 -> GPR")
    print("模型的原始特征数:", X_all.shape[1])
    print("样本数:", X_all.shape[0])
    print(f"筛选规则: score={FEATURE_SCORE_METHOD}, mode={FEATURE_SELECT_MODE}, top_ratio={TOP_RATIO}, top_k={TOP_K}")

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    y_pred_cv = np.zeros(len(y), dtype=float)
    fold_assignments = np.zeros(len(y), dtype=int)

    fold_metrics_rows = []
    fold_pc_summary_rows = []
    fold_pc_loading_rows = []
    fold_feature_score_rows = []
    fold_selected_rows = []
    fold_filtered_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_all, y), start=1):
        X_train = X_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        pca = PCA(n_components=PCA_VAR_THRESHOLD, svd_solver="full")
        pca.fit(X_train_scaled)

        summary_df, _, loading_long_df, loadings, _ = pca_to_tables(pca, feature_cols)
        score_df = build_feature_score_df(feature_cols, loadings, pca.explained_variance_ratio_)
        selected_features, filtered_features, selected_df, filtered_df = select_features(score_df)

        selected_idx = [feature_cols.index(f) for f in selected_features]
        X_train_selected = X_train_scaled[:, selected_idx]
        X_test_selected = X_test_scaled[:, selected_idx]

        gpr = build_gpr()
        gpr.fit(X_train_selected, y_train)
        y_pred_fold = gpr.predict(X_test_selected)

        y_pred_cv[test_idx] = y_pred_fold
        fold_assignments[test_idx] = fold

        fold_metrics = calc_metrics(y_test, y_pred_fold)
        fold_metrics_rows.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_selected_features": len(selected_features),
            **fold_metrics,
        })

        summary_df["fold"] = fold
        summary_df["n_pc_retained"] = int(pca.n_components_)
        fold_pc_summary_rows.append(summary_df)

        loading_long_df["fold"] = fold
        fold_pc_loading_rows.append(loading_long_df)

        score_df["fold"] = fold
        score_df["selected"] = score_df["feature"].isin(selected_features)
        fold_feature_score_rows.append(score_df)

        if len(selected_df) > 0:
            selected_df = selected_df.copy()
            selected_df["fold"] = fold
            selected_df["selected_rank"] = np.arange(1, len(selected_df) + 1)
            fold_selected_rows.append(selected_df)

        if len(filtered_df) > 0:
            filtered_df = filtered_df.copy()
            filtered_df["fold"] = fold
            filtered_df["filtered_rank"] = np.arange(1, len(filtered_df) + 1)
            fold_filtered_rows.append(filtered_df)

    overall_metrics = calc_metrics(y, y_pred_cv)

    print("result")
    print(f"Pearson r = {overall_metrics['Pearson_r']:.4f} (p = {overall_metrics['Pearson_p']:.4e})")
    print(f"R^2       = {overall_metrics['R2']:.4f}")
    print(f"MAE       = {overall_metrics['MAE']:.4f} years")
    print(f"RMSE      = {overall_metrics['RMSE']:.4f} years")

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
            "method": "PCA_select_raw_features_then_GPR",
            "n_samples": len(y),
            "n_raw_features": X_all.shape[1],
            "feature_score_method": FEATURE_SCORE_METHOD,
            "feature_select_mode": FEATURE_SELECT_MODE,
            "top_ratio": TOP_RATIO,
            "top_k": TOP_K,
            "mean_selected_features_cv": float(fold_metrics_df["n_selected_features"].mean()),
        },
    )

    save_scatter(
        y,
        y_pred_cv,
        f"{SCRIPT_TAG} 10-fold CV | r={overall_metrics['Pearson_r']:.3f}, MAE={overall_metrics['MAE']:.2f}",
        f"{SCRIPT_TAG}_scatter.png",
    )

    pd.concat(fold_pc_summary_rows, axis=0, ignore_index=True).to_csv(
        os.path.join(BASE_DIR, f"{SCRIPT_TAG}_cv_fold_pc_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: {SCRIPT_TAG}_cv_fold_pc_summary.csv")

    pd.concat(fold_pc_loading_rows, axis=0, ignore_index=True).to_csv(
        os.path.join(BASE_DIR, f"{SCRIPT_TAG}_cv_fold_pc_loadings_long.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: {SCRIPT_TAG}_cv_fold_pc_loadings_long.csv")

    pd.concat(fold_feature_score_rows, axis=0, ignore_index=True).to_csv(
        os.path.join(BASE_DIR, f"{SCRIPT_TAG}_cv_fold_feature_scores.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: {SCRIPT_TAG}_cv_fold_feature_scores.csv")

    if fold_selected_rows:
        pd.concat(fold_selected_rows, axis=0, ignore_index=True).to_csv(
            os.path.join(BASE_DIR, f"{SCRIPT_TAG}_cv_fold_selected_features.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print(f"已保存: {SCRIPT_TAG}_cv_fold_selected_features.csv")

    if fold_filtered_rows:
        pd.concat(fold_filtered_rows, axis=0, ignore_index=True).to_csv(
            os.path.join(BASE_DIR, f"{SCRIPT_TAG}_cv_fold_filtered_features.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print(f"已保存: {SCRIPT_TAG}_cv_fold_filtered_features.csv")

    print("使用全部数据拟合 PCA辅助筛选 -> GPR 模型......")
    full_imputer = SimpleImputer(strategy="median")
    full_scaler = StandardScaler()

    X_all_imp = full_imputer.fit_transform(X_all)
    X_all_scaled = full_scaler.fit_transform(X_all_imp)

    full_pca = PCA(n_components=PCA_VAR_THRESHOLD, svd_solver="full")
    full_pca.fit(X_all_scaled)

    full_pc_summary_df, full_loading_wide_df, full_loading_long_df, full_loadings, _ = pca_to_tables(full_pca, feature_cols)
    full_score_df = build_feature_score_df(feature_cols, full_loadings, full_pca.explained_variance_ratio_)
    selected_features, filtered_features, full_selected_df, full_filtered_df = select_features(full_score_df)

    selected_idx = [feature_cols.index(f) for f in selected_features]
    X_all_selected = X_all_scaled[:, selected_idx]

    full_gpr = build_gpr()
    full_gpr.fit(X_all_selected, y)

    full_pc_summary_df.to_csv(
        os.path.join(BASE_DIR, f"{SCRIPT_TAG}_full_pc_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: {SCRIPT_TAG}_full_pc_summary.csv")

    full_loading_wide_df.to_csv(
        os.path.join(BASE_DIR, f"{SCRIPT_TAG}_full_pc_loadings_wide.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: {SCRIPT_TAG}_full_pc_loadings_wide.csv")

    full_loading_long_df.to_csv(
        os.path.join(BASE_DIR, f"{SCRIPT_TAG}_full_pc_loadings_long.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: {SCRIPT_TAG}_full_pc_loadings_long.csv")

    full_score_df["selected"] = full_score_df["feature"].isin(selected_features)
    full_score_df.to_csv(
        os.path.join(BASE_DIR, f"{SCRIPT_TAG}_full_feature_scores.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: {SCRIPT_TAG}_full_feature_scores.csv")

    full_selected_df.to_csv(
        os.path.join(BASE_DIR, f"{SCRIPT_TAG}_full_selected_features.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: {SCRIPT_TAG}_full_selected_features.csv")

    full_filtered_df.to_csv(
        os.path.join(BASE_DIR, f"{SCRIPT_TAG}_full_filtered_features.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: {SCRIPT_TAG}_full_filtered_features.csv")

    selected_scaled_df = pd.DataFrame(X_all_selected, columns=selected_features)
    if "ID" in df.columns:
        selected_scaled_df.insert(0, "ID", df["ID"].values)
    selected_scaled_df["age"] = y
    if "Diagnosis_simple" in df.columns:
        selected_scaled_df["Diagnosis_simple"] = df["Diagnosis_simple"].values
    selected_scaled_df.to_csv(
        os.path.join(BASE_DIR, f"{SCRIPT_TAG}_full_selected_features_scaled.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: {SCRIPT_TAG}_full_selected_features_scaled.csv")

    artifact = {
        "imputer": full_imputer,
        "scaler": full_scaler,
        "pca_for_selection": full_pca,
        "gpr": full_gpr,
        "feature_cols": feature_cols,
        "selected_features": selected_features,
        "filtered_features": filtered_features,
        "meta_cols_present": meta_cols_present,
        "file_path": FILE_PATH,
        "method": "PCA_select_raw_features_then_GPR",
        "feature_score_method": FEATURE_SCORE_METHOD,
        "feature_select_mode": FEATURE_SELECT_MODE,
        "top_ratio": TOP_RATIO,
        "top_k": TOP_K,
    }
    model_name = f"{SCRIPT_TAG}_model.joblib"
    joblib.dump(artifact, os.path.join(BASE_DIR, model_name))
    print(f"已保存: {model_name}")

    print("finish!!!")


if __name__ == "__main__":
    main()
