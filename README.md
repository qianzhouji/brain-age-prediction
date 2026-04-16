# Brain Age Prediction with GPR

基于高斯过程回归（GPR）的脑龄预测项目，围绕 36 个脑结构特征开展建模、PCA 对比、特征重要性分析和跨组预测。

## 项目概览

本项目目前包含 4 类核心任务：

- 3 种建模方法对比：直接 GPR、PCA 特征筛选后 GPR、PCA 降维后 GPR
- 最终 NC 模型训练：在全部 NC 数据上训练可复用模型
- 2 种特征重要性分析：特征删除法、置换重要性
- 跨组预测：使用 NC 模型对 SCD/SCS 组进行预测

## 数据说明

输入数据为 `sixhos_NC_3.xlsx`。

- 来源：认知障碍研究数据（NC = 正常认知组）
- 样本数：1025
- 原始特征数：247 个脑区指标
- 当前建模使用特征：36 个大写开头的脑结构特征

36 个特征列表定义在 `Script/uppercase_features.py`。

说明：当前仓库中未看到 `sixhos_NC_3.xlsx`，如果要完整复现实验，需要先将该数据文件放到项目约定的位置。

## 环境依赖

核心依赖如下：

```bash
pip install pandas numpy scikit-learn matplotlib joblib openpyxl
```

建议使用 Python 3.10+。

## 主要脚本

### 特征定义

| 脚本 | 功能 |
|------|------|
| `Script/uppercase_features.py` | 定义 36 个用于建模的大写开头特征 |

### 三种建模方法对比

| 脚本 | 方法 | 结果摘要 |
|------|------|------|
| `Script/GPR_36upper.py` | 36 特征直接输入 GPR | R = 0.7511, MAE = 4.44y |
| `Script/PCA_GPR_36upper.py` | PCA 载荷筛选 Top 50% 后再做 GPR | R = 0.6904, MAE = 4.87y |
| `Script/PCAt_to_GPR_36upper.py` | PCA 降维为 18 个主成分后做 GPR | R = 0.7244, MAE = 4.65y |

结论：直接 GPR 性能最佳，PCA 降维次之，PCA 筛选相对较弱。

### 最终模型训练

| 脚本 | 功能 | 输出 |
|------|------|------|
| `Script/train_NC_model.py` | 在全部 NC 数据上训练最终模型 | `results/models/NC_36upper_GPR_model.joblib` |

结果：训练集 R = 0.8133，可用于后续跨组预测。

### 特征重要性分析

| 脚本 | 方法 | 最重要特征 | 输出 |
|------|------|------|------|
| `Script/ablation_36upper.py` | 特征删除法 | Brain-Stem | `results/features/NC_36upper_feature_ablation.csv` |
| `Script/permutation_36upper.py` | 置换重要性 | EstimatedTotalIntraCranialVol | `results/features/NC_36upper_permutation_importance.csv` |

关键发现：

- 删除法中 Brain-Stem 最重要，`delta_MAE = +0.143`
- 置换重要性中颅内总体积最重要，重要性为 `1.329`
- `Left-Pallidum` 可能引入噪声，删除后模型性能略有改善

### 跨组预测

| 脚本 | 功能 | 当前状态 |
|------|------|------|
| `Script/predict_cross_group.py` | 使用 NC 模型对 SCD/SCS 进行预测 | 已生成跨组预测结果，SCD/SCS 依赖对应输入数据是否齐备 |

## 运行方式

在项目根目录执行以下命令。

### 1. 三种模型对比

```bash
python3 Script/GPR_36upper.py
python3 Script/PCA_GPR_36upper.py
python3 Script/PCAt_to_GPR_36upper.py
```

### 2. 训练最终 NC 模型

```bash
python3 Script/train_NC_model.py
```

### 3. 运行特征重要性分析

```bash
python3 Script/ablation_36upper.py
python3 Script/permutation_36upper.py
```

### 4. 运行跨组预测

```bash
python3 Script/predict_cross_group.py
```

## 结果目录

目前结果文件已统一整理到 `results/` 目录下：

- `results/features/`：特征列表、特征删除法结果、置换重要性结果
- `results/metrics/`：各模型性能指标和折次评估结果
- `results/models/`：训练后的模型文件 `.joblib`
- `results/pca_analysis/`：PCA 主成分载荷、解释方差、主成分得分等
- `results/plots/`：散点图和跨组比较图
- `results/predictions/`：交叉验证预测结果、NC 预测结果及部分中间数据

## 当前核心结果

### 三种建模方法对比

| 方法 | Pearson r | R² | MAE | 特征数 |
|------|-----------|-----|-----|--------|
| 直接 GPR | **0.7511** | **0.5640** | **4.44y** | 36 |
| PCA 降维后 GPR | 0.7244 | 0.5247 | 4.65y | 18 PC |
| PCA 筛选后 GPR | 0.6904 | 0.4767 | 4.87y | 18 |

### 特征重要性 Top 5

特征删除法：

1. Brain-Stem (`delta_MAE = +0.143`)
2. EstimatedTotalIntraCranialVol (`+0.112`)
3. Left-vessel (`+0.109`)
4. Right-Inf-Lat-Vent (`+0.104`)
5. Left-VentralDC (`+0.085`)

置换重要性：

1. EstimatedTotalIntraCranialVol (`1.329`)
2. Brain-Stem (`0.421`)
3. Left-Thalamus (`0.344`)
4. Left-Accumbens-area (`0.225`)
5. Right-VentralDC (`0.131`)

## 项目结构

```text
Yellow/
├── README.md
├── Script/
│   ├── uppercase_features.py
│   ├── GPR_36upper.py
│   ├── PCA_GPR_36upper.py
│   ├── PCAt_to_GPR_36upper.py
│   ├── train_NC_model.py
│   ├── ablation_36upper.py
│   ├── permutation_36upper.py
│   └── predict_cross_group.py
├── results/
│   ├── features/
│   ├── metrics/
│   ├── models/
│   ├── pca_analysis/
│   ├── plots/
│   └── predictions/
├── 文献/
├── 实验执行记录.md
└── 其他研究记录与文档
```

## 仓库地址

[GitHub 仓库](https://github.com/qianzhouji/brain-age-prediction)

## 作者

虞珂
