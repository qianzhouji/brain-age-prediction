import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

FILE_PATH = os.path.join(BASE_DIR, "sixhos_NC_3.xlsx")
TARGET_COL = "age"
META_COLS = ["ID", "age", "Diagnosis_simple"]

df = pd.read_excel(FILE_PATH)

meta_cols_present = [c for c in META_COLS if c in df.columns]
candidate_feature_cols = [c for c in df.columns if c not in meta_cols_present]

X_all = df[candidate_feature_cols]
y = df[TARGET_COL].values.astype(float)

artifact = joblib.load(os.path.join(BASE_DIR, "PCAt_to_GPR_model.joblib"))
model = artifact["model"]
#找到在训练中使用的数据
feature_cols = artifact["feature_cols"]

# 确保列顺序一致
X_all = X_all[feature_cols]

result = permutation_importance(
    model,
    X_all,
    y,
    scoring="neg_mean_absolute_error",
    n_repeats=20,
    random_state=42,
    n_jobs=1
)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
})

importance_df = importance_df.sort_values(
    by="importance_mean",
    ascending=False
).reset_index(drop=True)

importance_df.to_csv(
    os.path.join(BASE_DIR, "NC_feature_importance.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("已保存")