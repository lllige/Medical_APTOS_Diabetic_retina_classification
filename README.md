# Medical_AI

**模型**：`tf_efficientnetv2_s`（参数量 20.8M）  
**设备**：CUDA  
**数据集规模**：Train 2930 | Val 366 | Test 366  

### 1. 类别分布
**训练集**：  
0 (No DR) 1434 | 1 (Mild) 300 | 2 (Moderate) 808 | 3 (Severe) 154 | 4 (Proliferative DR) 234  

**验证集**：  
0 172 | 1 40 | 2 104 | 3 22 | 4 28  

### 2. 训练概况
- 共训练 **44 个 epoch**  
- **早停**于 epoch 44（连续 7 个 epoch 验证集无改善）  
- **最佳模型**：epoch 37，验证集 Kappa = **0.9189**  

### 3. 验证集最终性能（Best checkpoint）
- **Accuracy**：83.88%  
- **Quadratic Weighted Kappa**：**0.9189**  

**分类报告（Val）**

| Class              | Precision | Recall | F1-score | Support |
|--------------------|-----------|--------|----------|---------|
| No DR              | 0.98      | 0.99   | 0.99     | 172     |
| Mild               | 0.74      | 0.65   | 0.69     | 40      |
| Moderate           | 0.78      | 0.77   | 0.77     | 104     |
| Severe             | 0.44      | 0.50   | 0.47     | 22      |
| Proliferative DR   | 0.66      | 0.68   | 0.67     | 28      |
| **macro avg**      | **0.72**  | **0.72**| **0.72** | 366     |
| **weighted avg**   | **0.84**  | **0.84**| **0.84** | 366     |

**混淆矩阵（Val）**

| True \ Pred        | No DR | Mild | Moderate | Severe | Proliferative DR |
|--------------------|-------|------|----------|--------|------------------|
| **No DR**          | 171   | 1    | 0        | 0      | 0                |
| **Mild**           | 2     | 26   | 11       | 0      | 1                |
| **Moderate**       | 1     | 8    | 80       | 11     | 4                |
| **Severe**         | 0     | 0    | 6        | 11     | 5                |
| **Proliferative DR**| 0    | 0    | 6        | 3      | 19               |

### 4. 测试集最终性能
- **Accuracy**：**84.43%**  
- **Quadratic Weighted Kappa**：**0.9163**  

**分类报告（Test）**

| Class              | Precision | Recall | F1-score | Support |
|--------------------|-----------|--------|----------|---------|
| No DR              | 0.99      | 0.98   | 0.99     | 199     |
| Mild               | 0.61      | 0.57   | 0.59     | 30      |
| Moderate           | 0.74      | 0.82   | 0.78     | 87      |
| Severe             | 0.30      | 0.35   | 0.32     | 17      |
| Proliferative DR   | 0.77      | 0.61   | 0.68     | 33      |
| **macro avg**      | **0.68**  | **0.66**| **0.67** | 366     |
| **weighted avg**   | **0.85**  | **0.84**| **0.85** | 366     |
