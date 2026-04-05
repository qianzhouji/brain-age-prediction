import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FILE_PATH = os.path.join(BASE_DIR, "sixhos_NC_3.xlsx")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_COL = "age"
META_COLS = ["ID", "age", "Diagnosis_simple"]

N_SPLITS = 10
RANDOM_STATE = 42

# 中间结果保存文件
RESULT_PATH = os.path.join(BASE_DIR, "NC_all_feature_ablation_results.csv")

df = pd.read_excel(FILE_PATH)

meta_cols_present = [c for c in META_COLS if c in df.columns]
feature_cols = [c for c in df.columns if c not in meta_cols_present]

X_all = df[feature_cols].copy()
y = df[TARGET_COL].values.astype(float)

print("样本数:", X_all.shape[0])
print("特征数:", X_all.shape[1])
print("输出目录:", BASE_DIR)

cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_splits = list(cv.split(X_all, y))  # 固定同一套分折，保证每个特征删除时公平比较

kernel = (
    ConstantKernel(1.0, (1e-3, 1e3)) *
    RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3)) +
    WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
)

def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=RANDOM_STATE
        ))
    ])


def evaluate_feature_set(X, y, cv_splits):
    model = build_model()
    y_pred_cv = cross_val_predict(
        model,
        X,
        y,
        cv=cv_splits,
        method="predict",
        n_jobs=1
    )

    r, _ = pearsonr(y, y_pred_cv)
    r2 = r2_score(y, y_pred_cv)
    mae = mean_absolute_error(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))

    return {
        "r": float(r),
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse)
    }


print("开始跑基线模型...")
t0 = time.time()
base_metrics = evaluate_feature_set(X_all, y, cv_splits)
t1 = time.time()

print("基线结果:")
print(base_metrics)
print(f"基线耗时: {(t1 - t0)/60:.2f} 分钟")

results = []

for i, feat in enumerate(feature_cols, start=1):
    print(f"[{i}/{len(feature_cols)}] 删除特征: {feat}")

    start_time = time.time()

    X_drop = X_all.drop(columns=[feat])
    metrics = evaluate_feature_set(X_drop, y, cv_splits)

    result = {
        "removed_feature": feat,
        "r_after_drop": metrics["r"],
        "R2_after_drop": metrics["R2"],
        "MAE_after_drop": metrics["MAE"],
        "RMSE_after_drop": metrics["RMSE"],

        "r_base": base_metrics["r"],
        "R2_base": base_metrics["R2"],
        "MAE_base": base_metrics["MAE"],
        "RMSE_base": base_metrics["RMSE"],

        "delta_r": base_metrics["r"] - metrics["r"],
        "delta_R2": base_metrics["R2"] - metrics["R2"],
        "delta_MAE": metrics["MAE"] - base_metrics["MAE"],
        "delta_RMSE": metrics["RMSE"] - base_metrics["RMSE"],

        "n_features_after_drop": X_drop.shape[1],
        "elapsed_sec": time.time() - start_time
    }

    results.append(result)

    # 每次都保存一次，防止中途断掉
    temp_df = pd.DataFrame(results).sort_values(
        by="delta_MAE", ascending=False
    ).reset_index(drop=True)
    temp_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    print(
        f"耗时={result['elapsed_sec']:.1f}s"
    )

print("finish")


result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    by="delta_MAE", ascending=False
).reset_index(drop=True)

result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")