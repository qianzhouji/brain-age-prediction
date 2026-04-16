import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from uppercase_features import UPPERCASE_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NC训练数据路径
NC_FILE_PATH = os.path.join(BASE_DIR, "sixhos_NC_3.xlsx")

# 输出模型路径
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "NC_36upper_GPR_model.joblib")

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
            random_state=RANDOM_STATE,
        ))
    ])


def main():
    print("=" * 60)
    print("在NC数据上训练36个大写特征的GPR模型")
    print("=" * 60)
    
    # 1. 加载NC数据
    print(f"\n1. 加载NC数据: {NC_FILE_PATH}")
    df_nc = pd.read_excel(NC_FILE_PATH)
    
    meta_cols_present = [c for c in META_COLS if c in df_nc.columns]
    feature_cols = [c for c in UPPERCASE_FEATURES if c in df_nc.columns]
    
    print(f"   NC样本数: {len(df_nc)}")
    print(f"   使用特征数: {len(feature_cols)}")
    
    missing = set(UPPERCASE_FEATURES) - set(df_nc.columns)
    if missing:
        print(f"   警告: 缺失特征: {missing}")
    
    # 2. 准备训练数据
    X_nc = df_nc[feature_cols].copy()
    y_nc = df_nc[TARGET_COL].values.astype(float)
    
    print(f"\n2. 训练数据准备完成")
    print(f"   X shape: {X_nc.shape}")
    print(f"   y shape: {y_nc.shape}")
    print(f"   年龄范围: {y_nc.min():.1f} - {y_nc.max():.1f} 岁")
    
    # 3. 训练模型
    print(f"\n3. 开始训练GPR模型...")
    model = build_model()
    model.fit(X_nc, y_nc)
    print("   训练完成!")
    
    # 4. 在NC上评估（训练集性能参考）
    y_pred_nc = model.predict(X_nc)
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score, mean_absolute_error
    
    r, p = pearsonr(y_nc, y_pred_nc)
    r2 = r2_score(y_nc, y_pred_nc)
    mae = mean_absolute_error(y_nc, y_pred_nc)
    
    print(f"\n4. NC训练集性能（参考，非泛化性能）:")
    print(f"   Pearson r = {r:.4f} (p = {p:.4e})")
    print(f"   R² = {r2:.4f}")
    print(f"   MAE = {mae:.4f} years")
    
    # 5. 保存模型
    print(f"\n5. 保存模型到: {MODEL_OUTPUT_PATH}")
    import joblib
    
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "meta_cols_present": meta_cols_present,
        "nc_file_path": NC_FILE_PATH,
        "nc_n_samples": len(df_nc),
        "nc_age_range": (float(y_nc.min()), float(y_nc.max())),
        "training_metrics": {
            "pearson_r": float(r),
            "pearson_p": float(p),
            "r2": float(r2),
            "mae": float(mae),
        },
        "method": "NC_36upper_GPR",
        "description": "Trained on NC data with 36 uppercase brain structure features",
    }
    
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print("   模型保存成功!")
    
    # 6. 保存特征列表（便于跨组预测时使用）
    feature_list_path = os.path.join(BASE_DIR, "NC_36upper_feature_list.csv")
    pd.DataFrame({"feature": feature_cols}).to_csv(
        feature_list_path, index=False, encoding="utf-8-sig"
    )
    print(f"   特征列表保存到: {feature_list_path}")
    
    print("\n" + "=" * 60)
    print("训练完成! 模型可用于预测SCD/SCS/MCI/AD等人群")
    print("=" * 60)
    
    print(f"\n使用示例:")
    print(f"  from sklearn.externals import joblib")
    print(f"  artifact = joblib.load('{MODEL_OUTPUT_PATH}')")
    print(f"  model = artifact['model']")
    print(f"  predictions = model.predict(X_new)")


if __name__ == "__main__":
    main()
