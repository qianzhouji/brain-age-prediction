import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


NC_FILE_PATH = os.path.join(BASE_DIR, "sixhos_NC_3.xlsx")
SCD_FILE_PATH = os.path.join(BASE_DIR, "sixhos_SCD.xlsx")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "SCD_feature_ablation.csv")

TARGET_COL = "age"
META_COLS = ["ID", "age", "Diagnosis_simple"]

RANDOM_STATE = 42

df_nc = pd.read_excel(NC_FILE_PATH)
df_scd = pd.read_excel(SCD_FILE_PATH)

print("NC 数据:", df_nc.shape)
print("SCD 数据:", df_scd.shape)


meta_cols_present_nc = [c for c in META_COLS if c in df_nc.columns]
feature_cols = [c for c in df_nc.columns if c not in meta_cols_present_nc]

missing_in_scd = [c for c in feature_cols if c not in df_scd.columns]
if len(missing_in_scd) > 0:
    raise ValueError(
        "SCD 表缺少 NC 训练所需特征列。前10个缺失列如下：\n"
        + "\n".join(missing_in_scd[:10])
    )

X_nc_all = df_nc[feature_cols].copy()
y_nc_all = pd.to_numeric(df_nc[TARGET_COL], errors="coerce").values.astype(float)

X_scd_all = df_scd[feature_cols].copy()
y_scd_all = pd.to_numeric(df_scd[TARGET_COL], errors="coerce").values.astype(float)

# 去掉年龄缺失（防御性处理）
valid_nc = ~np.isnan(y_nc_all)
X_nc_all = X_nc_all.loc[valid_nc].reset_index(drop=True)
y_nc_all = y_nc_all[valid_nc]
df_nc = df_nc.loc[valid_nc].reset_index(drop=True)

valid_scd = ~np.isnan(y_scd_all)
X_scd_all = X_scd_all.loc[valid_scd].reset_index(drop=True)
y_scd_all = y_scd_all[valid_scd]
df_scd = df_scd.loc[valid_scd].reset_index(drop=True)

print("NC 样本数:", X_nc_all.shape[0])
print("SCD 样本数:", X_scd_all.shape[0])
print("特征数一共有:", X_nc_all.shape[1])

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

def evaluate_on_scd(model, X_scd, y_scd):
    pred = model.predict(X_scd)

    r, p = pearsonr(y_scd, pred)
    r2 = r2_score(y_scd, pred)
    mae = mean_absolute_error(y_scd, pred)
    rmse = np.sqrt(mean_squared_error(y_scd, pred))
    brain_pad = pred - y_scd

    return {
        "predicted_age": pred,
        "brain_PAD": brain_pad,
        "r": float(r),
        "p": float(p),
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "brain_PAD_mean": float(np.mean(brain_pad)),
        "brain_PAD_std": float(np.std(brain_pad, ddof=1))
    }

print("开始训练基线 NC 模型，并应用到 SCD...")
t0 = time.time()

base_model = build_model()
base_model.fit(X_nc_all, y_nc_all)

base_metrics = evaluate_on_scd(base_model, X_scd_all, y_scd_all)

t1 = time.time()

print("基线结果（NC训练 -> SCD预测）:")
print(f"Pearson r           = {base_metrics['r']:.4f}")
print(f"R^2                 = {base_metrics['R2']:.4f}")
print(f"MAE                 = {base_metrics['MAE']:.4f}")
print(f"RMSE                = {base_metrics['RMSE']:.4f}")
print(f"brain_PAD mean      = {base_metrics['brain_PAD_mean']:.4f}")
print(f"brain_PAD std       = {base_metrics['brain_PAD_std']:.4f}")
print(f"基线耗时            = {(t1 - t0)/60:.2f} 分钟")

results = []

for i, feat in enumerate(feature_cols, start=1):
    print(f"[{i}/{len(feature_cols)}] 删除 NC 特征: {feat}")
    start_time = time.time()

    # 在 NC 和 SCD 上删同一个列，保证输入维度一致
    X_nc_drop = X_nc_all.drop(columns=[feat])
    X_scd_drop = X_scd_all.drop(columns=[feat])

    # 重训 NC 模型
    model_drop = build_model()
    model_drop.fit(X_nc_drop, y_nc_all)

    # 应用到 SCD
    metrics = evaluate_on_scd(model_drop, X_scd_drop, y_scd_all)

    result = {
        "removed_feature": feat,

        # 删除后，在 SCD 上的表现
        "r_SCD_after_drop": metrics["r"],
        "R2_SCD_after_drop": metrics["R2"],
        "MAE_SCD_after_drop": metrics["MAE"],
        "RMSE_SCD_after_drop": metrics["RMSE"],
        "brain_PAD_mean_SCD_after_drop": metrics["brain_PAD_mean"],
        "brain_PAD_std_SCD_after_drop": metrics["brain_PAD_std"],

        # 基线
        "r_SCD_base": base_metrics["r"],
        "R2_SCD_base": base_metrics["R2"],
        "MAE_SCD_base": base_metrics["MAE"],
        "RMSE_SCD_base": base_metrics["RMSE"],
        "brain_PAD_mean_SCD_base": base_metrics["brain_PAD_mean"],
        "brain_PAD_std_SCD_base": base_metrics["brain_PAD_std"],

        # 变化量
        "delta_r_SCD": base_metrics["r"] - metrics["r"],
        "delta_R2_SCD": base_metrics["R2"] - metrics["R2"],
        "delta_MAE_SCD": metrics["MAE"] - base_metrics["MAE"],
        "delta_RMSE_SCD": metrics["RMSE"] - base_metrics["RMSE"],
        "delta_brain_PAD_mean_SCD": metrics["brain_PAD_mean"] - base_metrics["brain_PAD_mean"],

        "n_features_after_drop": X_nc_drop.shape[1],
        "elapsed_sec": time.time() - start_time
    }

    results.append(result)

    # 每次都保存一次，防止程序中断
    temp_df = pd.DataFrame(results).sort_values(
        by="delta_MAE_SCD", ascending=False
    ).reset_index(drop=True)
    temp_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    print(
        f"  完成 | delta_MAE_SCD={result['delta_MAE_SCD']:.4f}, "
        f"delta_R2_SCD={result['delta_R2_SCD']:.4f}, "
        f"耗时={result['elapsed_sec']:.1f}s"
    )

print("finished!")


result_df = pd.DataFrame(results).sort_values(
    by="delta_MAE_SCD", ascending=False
).reset_index(drop=True)

result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

print("已保存结果到:")
print(RESULT_PATH)
