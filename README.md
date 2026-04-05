# Brain Age Prediction with GPR

基于高斯过程回归（GPR）的脑年龄预测项目，使用36个脑结构特征（大写开头特征）进行建模和特征重要性分析。

---

## 项目背景

本项目基于2025年4月4日会议纪要，目标是完成"36个feature预测7个目标值"的任务：
- **3种建模方法对比**：直接GPR、PCA筛选→GPR、PCA降维→GPR
- **2种特征重要性评估**：特征删除法、置换重要性
- **跨组预测验证**：NC训练模型→SCD/SCS预测

---

## 数据说明

**输入数据**：`sixhos_NC_3.xlsx`
- 来源：认知障碍研究（NC=正常认知组）
- 样本数：1025
- 特征数：247个脑区指标
- 本研究使用：36个大写开头特征（脑体积、脑室、皮层等结构指标）

**36个特征列表**：见 `uppercase_features.py`

---

## 脚本构成

### 核心依赖
```python
pandas, numpy, scikit-learn, matplotlib, joblib, openpyxl
```

### 特征定义
| 脚本 | 功能 |
|------|------|
| `uppercase_features.py` | 定义36个大写开头特征列表 |

### 目标1-3：三种建模方法对比

| 脚本 | 方法 | 输出 |
|------|------|------|
| `GPR_36upper.py` | 36特征直接输入GPR | R=0.7511, MAE=4.44y |
| `PCA_GPR_36upper.py` | PCA载荷筛选Top50%→GPR | R=0.6904, MAE=4.87y |
| `PCAt_to_GPR_36upper.py` | PCA降维(18PC)→GPR | R=0.7244, MAE=4.65y |

**结论**：直接GPR性能最优，PCA降维次之，PCA筛选最差。

### 目标4：NC模型训练

| 脚本 | 功能 | 输出 |
|------|------|------|
| `train_NC_model.py` | 在全部NC数据上训练最终模型 | `NC_36upper_GPR_model.joblib` |

**结果**：训练集R=0.8133，用于后续跨组预测。

### 目标5-6：特征重要性评估

| 脚本 | 方法 | 最重要特征 | 输出文件 |
|------|------|-----------|---------|
| `ablation_36upper.py` | 特征删除法 | Brain-Stem | `NC_36upper_feature_ablation.csv` |
| `permutation_36upper.py` | 置换重要性 | EstimatedTotalIntraCranialVol | `NC_36upper_permutation_importance.csv` |

**关键发现**：
- 删除法：Brain-Stem最重要（delta_MAE=+0.143）
- 置换法：颅内体积最重要（重要性=1.329）
- Left-Pallidum可能引入噪声（删除后性能提升）

### 目标7：跨组预测

| 脚本 | 功能 | 状态 |
|------|------|------|
| `predict_cross_group.py` | NC模型→SCD/SCS预测 | NC验证通过，SCD/SCS数据待补充 |

---

## 快速开始

### 1. 安装依赖
```bash
pip install pandas numpy scikit-learn matplotlib joblib openpyxl
```

### 2. 运行实验
```bash
# 目标1：直接GPR
python3 GPR_36upper.py

# 目标2：PCA筛选
python3 PCA_GPR_36upper.py

# 目标3：PCA降维
python3 PCAt_to_GPR_36upper.py

# 目标4：训练NC模型
python3 train_NC_model.py

# 目标5：特征删除法（~2分钟）
python3 ablation_36upper.py

# 目标6：置换重要性
python3 permutation_36upper.py

# 目标7：跨组预测
python3 predict_cross_group.py
```

### 3. 查看结果
所有结果保存在 `~/Desktop/Yellow/`：
- `*_metrics.csv` - 性能指标
- `*_scatter.png` - 预测散点图
- `*_model.joblib` - 训练好的模型
- `NC_36upper_feature_ablation.csv` - 特征删除法结果
- `NC_36upper_permutation_importance.csv` - 置换重要性结果

---

## 核心结果

### 三种建模方法对比

| 方法 | Pearson r | R² | MAE | 特征数 |
|------|-----------|-----|-----|--------|
| 直接GPR | **0.7511** | **0.5640** | **4.44y** | 36 |
| PCA降维→GPR | 0.7244 | 0.5247 | 4.65y | 18 PC |
| PCA筛选→GPR | 0.6904 | 0.4767 | 4.87y | 18 |

### 特征重要性Top 5

**特征删除法**：
1. Brain-Stem (delta_MAE=+0.143)
2. EstimatedTotalIntraCranialVol (+0.112)
3. Left-vessel (+0.109)
4. Right-Inf-Lat-Vent (+0.104)
5. Left-VentralDC (+0.085)

**置换重要性**：
1. EstimatedTotalIntraCranialVol (1.329)
2. Brain-Stem (0.421)
3. Left-Thalamus (0.344)
4. Left-Accumbens-area (0.225)
5. Right-VentralDC (0.131)

---

## 项目结构

```
~/Desktop/Yellow/
├── uppercase_features.py          # 36个特征定义
├── GPR_36upper.py                 # 目标1：直接GPR
├── PCA_GPR_36upper.py             # 目标2：PCA筛选
├── PCAt_to_GPR_36upper.py         # 目标3：PCA降维
├── train_NC_model.py              # 目标4：NC模型训练
├── ablation_36upper.py            # 目标5：特征删除法
├── permutation_36upper.py         # 目标6：置换重要性
├── predict_cross_group.py         # 目标7：跨组预测
├── sixhos_NC_3.xlsx               # 输入数据
├── NC_36upper_GPR_model.joblib    # 训练好的模型
├── *_metrics.csv                  # 性能指标
├── *_scatter.png                  # 散点图
├── NC_36upper_feature_ablation.csv    # 消融结果
├── NC_36upper_permutation_importance.csv  # 置换重要性
└── 实验执行记录.md                # 详细实验记录
```

---

## GitHub仓库

https://github.com/qianzhouji/brain-age-prediction

---

## 作者

虞珂

## 日期

2026-04-06
