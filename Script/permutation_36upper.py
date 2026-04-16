import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.inspection import permutation_importance

from uppercase_features import UPPERCASE_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径
NC_FILE_PATH = os.path.join(BASE_DIR, "sixhos_NC_3.xlsx")

# 模型路径（使用已训练的NC模型）
MODEL_PATH = os.path.join(BASE_DIR, "NC_36upper_GPR_model.joblib")

# 结果输出路径
RESULT_PATH = os.path.join(BASE_DIR, "NC_36upper_permutation_importance.csv")

TARGET_COL = "age"
META_COLS = ["ID", "age", "Diagnosis_simple"]

RANDOM_STATE = 42
N_REPEATS = 20  # 置换重复次数


def main():
    print("=" * 70)
    print("目标6: 置换重要性评估 (Permutation Importance)")
    print("=" * 70)
    
    # 1. 加载NC数据
    print(f"\n1. 加载NC数据: {NC_FILE_PATH}")
    df_nc = pd.read_excel(NC_FILE_PATH)
    
    meta_cols_present = [c for c in META_COLS if c in df_nc.columns]
    feature_cols = [c for c in UPPERCASE_FEATURES if c in df_nc.columns]
    
    print(f"   NC样本数: {len(df_nc)}")
    print(f"   特征数: {len(feature_cols)}")
    
    # 2. 准备数据
    X_nc = df_nc[feature_cols].copy()
    y_nc = pd.to_numeric(df_nc[TARGET_COL], errors="coerce").values.astype(float)
    
    # 去掉年龄缺失
    valid = ~np.isnan(y_nc)
    X_nc = X_nc.loc[valid].reset_index(drop=True)
    y_nc = y_nc[valid]
    
    print(f"   有效样本数: {len(X_nc)}")
    
    # 3. 加载已训练的模型
    print(f"\n2. 加载已训练模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"   错误: 模型文件不存在，请先运行 train_NC_model.py")
        return
    
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    
    print(f"   模型加载成功!")
    print(f"   模型训练时使用的特征数: {len(artifact['feature_cols'])}")
    
    # 4. 计算置换重要性
    print(f"\n3. 计算置换重要性...")
    print(f"   评估指标: neg_mean_absolute_error")
    print(f"   置换重复次数: {N_REPEATS}")
    print(f"   随机种子: {RANDOM_STATE}")
    print(f"   计算中，请稍候...")
    
    result = permutation_importance(
        model,
        X_nc,
        y_nc,
        scoring="neg_mean_absolute_error",  # 负MAE，越大越好
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1  # 单线程，避免内存问题
    )
    
    # 5. 整理结果
    print(f"\n4. 整理结果...")
    
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False
    ).reset_index(drop=True)
    
    # 添加排名
    importance_df["importance_rank"] = np.arange(1, len(importance_df) + 1)
    
    # 保存结果
    importance_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"   结果已保存: {RESULT_PATH}")
    
    # 6. 输出Top 10最重要特征
    print(f"\n5. 置换重要性排名（Top 10）:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Std':<10}")
    print("-" * 70)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['importance_rank']:<6} {row['feature']:<35} "
              f"{row['importance_mean']:<12.6f} {row['importance_std']:<10.6f}")
    
    # 7. 输出Bottom 5最不重要特征
    print(f"\n   置换重要性排名（Bottom 5）:")
    print("-" * 70)
    for idx, row in importance_df.tail(5).iterrows():
        print(f"{row['importance_rank']:<6} {row['feature']:<35} "
              f"{row['importance_mean']:<12.6f} {row['importance_std']:<10.6f}")
    
    # 8. 统计信息
    print(f"\n6. 统计信息:")
    print(f"   平均重要性: {importance_df['importance_mean'].mean():.6f}")
    print(f"   重要性标准差: {importance_df['importance_mean'].std():.6f}")
    print(f"   最大重要性: {importance_df['importance_mean'].max():.6f} "
          f"({importance_df.iloc[0]['feature']})")
    print(f"   最小重要性: {importance_df['importance_mean'].min():.6f} "
          f"({importance_df.iloc[-1]['feature']})")
    
    print("\n" + "=" * 70)
    print("置换重要性评估完成!")
    print("=" * 70)
    
    print(f"\n说明:")
    print(f"  - importance_mean: 置换该特征后模型性能下降的平均值")
    print(f"  - 值越大表示该特征越重要")
    print(f"  - 基于neg_mean_absolute_error，正值表示性能下降（特征重要）")


if __name__ == "__main__":
    main()
