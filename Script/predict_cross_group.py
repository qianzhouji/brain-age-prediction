import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from uppercase_features import UPPERCASE_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径（NC训练模型）
MODEL_PATH = os.path.join(BASE_DIR, "NC_36upper_GPR_model.joblib")

# 数据路径
NC_FILE_PATH = os.path.join(BASE_DIR, "sixhos_NC_3.xlsx")
SCD_FILE_PATH = os.path.join(BASE_DIR, "sixhos_SCD.xlsx")
SCS_FILE_PATH = os.path.join(BASE_DIR, "sixhos_SCS.xlsx")

# 结果输出路径
RESULT_PATH = os.path.join(BASE_DIR, "cross_group_prediction_results.csv")

TARGET_COL = "age"
META_COLS = ["ID", "age", "Diagnosis_simple"]


def calc_metrics(y_true, y_pred):
    r, p = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    brain_pad = y_pred - y_true
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "brain_PAD_mean": float(np.mean(brain_pad)),
        "brain_PAD_std": float(np.std(brain_pad, ddof=1)),
    }


def predict_group(model, feature_cols, group_name, file_path):
    """对指定人群进行预测"""
    print(f"\n{'='*60}")
    print(f"预测人群: {group_name}")
    print(f"{'='*60}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"  警告: 文件不存在 {file_path}")
        print(f"  跳过该人群预测")
        return None
    
    # 加载数据
    print(f"  加载数据: {file_path}")
    df = pd.read_excel(file_path)
    print(f"  样本数: {len(df)}")
    
    # 检查特征列
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"  错误: 缺少以下特征: {missing_features[:5]}...")
        print(f"  跳过该人群预测")
        return None
    
    # 准备数据
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").values.astype(float)
    
    # 去掉年龄缺失
    valid = ~np.isnan(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid]
    df = df.loc[valid].reset_index(drop=True)
    
    print(f"  有效样本数: {len(df)}")
    print(f"  年龄范围: {y.min():.1f} - {y.max():.1f} 岁")
    
    # 预测
    print(f"  进行预测...")
    y_pred = model.predict(X)
    brain_pad = y_pred - y
    
    # 计算指标
    metrics = calc_metrics(y, y_pred)
    
    print(f"  预测结果:")
    print(f"    Pearson r = {metrics['pearson_r']:.4f} (p = {metrics['pearson_p']:.4e})")
    print(f"    R² = {metrics['R2']:.4f}")
    print(f"    MAE = {metrics['MAE']:.4f} years")
    print(f"    RMSE = {metrics['RMSE']:.4f} years")
    print(f"    Brain-PAD mean = {metrics['brain_PAD_mean']:.4f} years")
    print(f"    Brain-PAD std = {metrics['brain_PAD_std']:.4f} years")
    
    # 准备结果DataFrame
    result_df = pd.DataFrame({
        "ID": df["ID"] if "ID" in df.columns else range(len(df)),
        "age": y,
        "predicted_age": y_pred,
        "brain_PAD": brain_pad,
    })
    
    if "Diagnosis_simple" in df.columns:
        result_df["Diagnosis_simple"] = df["Diagnosis_simple"].values
    
    return {
        "group": group_name,
        "n_samples": len(df),
        "metrics": metrics,
        "predictions": result_df,
    }


def plot_comparison(results_list):
    """绘制各组brain-PAD分布对比图"""
    fig, axes = plt.subplots(1, len(results_list), figsize=(6*len(results_list), 5))
    
    if len(results_list) == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results_list):
        group = result["group"]
        brain_pad = result["predictions"]["brain_PAD"]
        mean_pad = result["metrics"]["brain_PAD_mean"]
        
        ax.hist(brain_pad, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', label='PAD=0')
        ax.axvline(x=mean_pad, color='g', linestyle='-', label=f'Mean={mean_pad:.2f}')
        ax.set_xlabel('Brain-PAD (years)')
        ax.set_ylabel('Count')
        ax.set_title(f'{group}\n(n={result["n_samples"]}, MAE={result["metrics"]["MAE"]:.2f})')
        ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "cross_group_brain_PAD_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n  Brain-PAD分布对比图已保存: {plot_path}")


def main():
    print("=" * 70)
    print("目标7: 跨组预测 (NC训练模型 → SCD/SCS预测)")
    print("=" * 70)
    
    # 1. 加载NC训练模型
    print(f"\n1. 加载NC训练模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"   错误: 模型文件不存在，请先运行 train_NC_model.py")
        return
    
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    
    print(f"   模型加载成功!")
    print(f"   特征数: {len(feature_cols)}")
    
    # 2. 在NC上验证（参考）
    print(f"\n2. NC组验证（参考，训练集）:")
    nc_result = predict_group(model, feature_cols, "NC", NC_FILE_PATH)
    
    # 3. 在SCD上预测
    scd_result = predict_group(model, feature_cols, "SCD", SCD_FILE_PATH)
    
    # 4. 在SCS上预测
    scs_result = predict_group(model, feature_cols, "SCS", SCS_FILE_PATH)
    
    # 5. 汇总结果
    print(f"\n{'='*70}")
    print("跨组预测结果汇总")
    print(f"{'='*70}")
    
    results_list = []
    if nc_result:
        results_list.append(nc_result)
    if scd_result:
        results_list.append(scd_result)
    if scs_result:
        results_list.append(scs_result)
    
    # 创建汇总表
    summary_rows = []
    for result in results_list:
        summary_rows.append({
            "Group": result["group"],
            "N": result["n_samples"],
            "Pearson_r": result["metrics"]["pearson_r"],
            "R2": result["metrics"]["R2"],
            "MAE": result["metrics"]["MAE"],
            "RMSE": result["metrics"]["RMSE"],
            "Brain_PAD_mean": result["metrics"]["brain_PAD_mean"],
            "Brain_PAD_std": result["metrics"]["brain_PAD_std"],
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))
    
    # 保存汇总结果
    summary_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n汇总结果已保存: {RESULT_PATH}")
    
    # 6. 保存各组详细预测结果
    for result in results_list:
        group = result["group"]
        pred_path = os.path.join(BASE_DIR, f"{group}_predictions.csv")
        result["predictions"].to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"{group}详细预测结果: {pred_path}")
    
    # 7. 绘制对比图
    if len(results_list) > 0:
        print(f"\n3. 生成Brain-PAD分布对比图...")
        plot_comparison(results_list)
    
    print("\n" + "=" * 70)
    print("跨组预测完成!")
    print("=" * 70)
    
    print(f"\n说明:")
    print(f"  - Brain-PAD = Predicted Age - Chronological Age")
    print(f"  - 正值: 脑年龄 > 实际年龄（脑老化加速）")
    print(f"  - 负值: 脑年龄 < 实际年龄（脑老化延缓）")


if __name__ == "__main__":
    main()
