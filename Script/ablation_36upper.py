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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from uppercase_features import UPPERCASE_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径
NC_FILE_PATH = os.path.join(BASE_DIR, "sixhos_NC_3.xlsx")

# 结果输出路径
RESULT_PATH = os.path.join(BASE_DIR, "NC_36upper_feature_ablation.csv")

TARGET_COL = "age"
META_COLS = ["ID", "age", "Diagnosis_simple"]

RANDOM_STATE = 42


def build_kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
    )


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=build_kernel(),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=RANDOM_STATE
        ))
    ])


def calc_metrics(y_true, y_pred):
    r, p = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


def main():
    print("=" * 70)
    print("目标5: 特征删除法消融实验 (Leave-One-Feature-Out Ablation)")
    print("=" * 70)
    
    # 1. 加载NC数据
    print(f"\n1. 加载NC数据: {NC_FILE_PATH}")
    df_nc = pd.read_excel(NC_FILE_PATH)
    
    meta_cols_present = [c for c in META_COLS if c in df_nc.columns]
    feature_cols = [c for c in UPPERCASE_FEATURES if c in df_nc.columns]
    
    print(f"   NC样本数: {len(df_nc)}")
    print(f"   特征数: {len(feature_cols)}")
    
    # 2. 准备数据
    X_nc_all = df_nc[feature_cols].copy()
    y_nc_all = pd.to_numeric(df_nc[TARGET_COL], errors="coerce").values.astype(float)
    
    # 去掉年龄缺失
    valid = ~np.isnan(y_nc_all)
    X_nc_all = X_nc_all.loc[valid].reset_index(drop=True)
    y_nc_all = y_nc_all[valid]
    df_nc = df_nc.loc[valid].reset_index(drop=True)
    
    print(f"   有效样本数: {len(df_nc)}")
    
    # 3. 训练基线模型（使用全部特征）
    print(f"\n2. 训练基线模型（全部{len(feature_cols)}个特征）...")
    t0 = time.time()
    
    base_model = build_model()
    base_model.fit(X_nc_all, y_nc_all)
    y_pred_base = base_model.predict(X_nc_all)
    base_metrics = calc_metrics(y_nc_all, y_pred_base)
    
    t1 = time.time()
    
    print(f"   基线结果:")
    print(f"     Pearson r = {base_metrics['pearson_r']:.4f}")
    print(f"     R² = {base_metrics['R2']:.4f}")
    print(f"     MAE = {base_metrics['MAE']:.4f} years")
    print(f"     耗时: {(t1-t0)/60:.2f} 分钟")
    
    # 4. 逐个删除特征，重新训练
    print(f"\n3. 开始逐个删除特征消融实验（共{len(feature_cols)}个特征）...")
    print(f"   预计总耗时: ~{(len(feature_cols) * (t1-t0))/60:.1f} 分钟")
    print(f"   中间结果将实时保存到: {RESULT_PATH}")
    print()
    
    results = []
    
    for i, feat in enumerate(feature_cols, start=1):
        print(f"[{i}/{len(feature_cols)}] 删除特征: {feat}")
        start_time = time.time()
        
        # 删除该特征
        X_nc_drop = X_nc_all.drop(columns=[feat])
        
        # 重训模型
        model_drop = build_model()
        model_drop.fit(X_nc_drop, y_nc_all)
        y_pred_drop = model_drop.predict(X_nc_drop)
        
        # 计算指标
        metrics = calc_metrics(y_nc_all, y_pred_drop)
        
        # 计算变化量
        result = {
            "removed_feature": feat,
            
            # 删除后的性能
            "r_after_drop": metrics["pearson_r"],
            "R2_after_drop": metrics["R2"],
            "MAE_after_drop": metrics["MAE"],
            "RMSE_after_drop": metrics["RMSE"],
            
            # 基线性能
            "r_base": base_metrics["pearson_r"],
            "R2_base": base_metrics["R2"],
            "MAE_base": base_metrics["MAE"],
            "RMSE_base": base_metrics["RMSE"],
            
            # 变化量（正值表示该特征重要）
            "delta_r": base_metrics["pearson_r"] - metrics["pearson_r"],
            "delta_R2": base_metrics["R2"] - metrics["R2"],
            "delta_MAE": metrics["MAE"] - base_metrics["MAE"],
            "delta_RMSE": metrics["RMSE"] - base_metrics["RMSE"],
            
            "n_features_after_drop": X_nc_drop.shape[1],
            "elapsed_sec": time.time() - start_time,
        }
        
        results.append(result)
        
        # 实时保存中间结果
        temp_df = pd.DataFrame(results).sort_values(
            by="delta_MAE", ascending=False
        ).reset_index(drop=True)
        temp_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
        
        print(f"    完成 | delta_MAE={result['delta_MAE']:.4f}, "
              f"delta_r={result['delta_r']:.4f}, "
              f"耗时={result['elapsed_sec']:.1f}s")
    
    # 5. 最终结果整理
    print(f"\n4. 实验完成! 最终结果:")
    result_df = pd.DataFrame(results).sort_values(
        by="delta_MAE", ascending=False
    ).reset_index(drop=True)
    
    # 添加排名
    result_df["importance_rank"] = np.arange(1, len(result_df) + 1)
    
    result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"   结果已保存: {RESULT_PATH}")
    
    # 6. 输出Top 10最重要特征
    print(f"\n5. 特征重要性排名（Top 10，按delta_MAE）:")
    print("-" * 70)
    for idx, row in result_df.head(10).iterrows():
        print(f"{row['importance_rank']:2d}. {row['removed_feature']:<35} "
              f"delta_MAE={row['delta_MAE']:+.4f}, "
              f"delta_r={row['delta_r']:+.4f}")
    
    # 7. 输出Bottom 5最不重要特征
    print(f"\n   特征重要性排名（Bottom 5，按delta_MAE）:")
    print("-" * 70)
    for idx, row in result_df.tail(5).iterrows():
        print(f"{row['importance_rank']:2d}. {row['removed_feature']:<35} "
              f"delta_MAE={row['delta_MAE']:+.4f}, "
              f"delta_r={row['delta_r']:+.4f}")
    
    print("\n" + "=" * 70)
    print("特征删除法消融实验完成!")
    print("=" * 70)
    
    print(f"\n说明:")
    print(f"  - delta_MAE > 0: 删除该特征后MAE增加，说明该特征重要")
    print(f"  - delta_MAE < 0: 删除该特征后MAE减少，说明该特征可能引入噪声")
    print(f"  - delta_r > 0: 删除该特征后r下降，说明该特征重要")


if __name__ == "__main__":
    main()
