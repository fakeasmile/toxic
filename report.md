# report

## 基础配置
基础模型：Tags:v1.3


| 配置项                           | 参数值                   |
|:------------------------------|:----------------------|
| 数据集名称 (dataset_name)          | TOXICN/COLD           |
| 预训练模型bert-base-chinese        | ModifiedBert          |
| 随机种子 (seed)                   | 1                     |
| 确定性模式 (use_deterministic)     | true                  |
| 批次大小 (batch_size)             | 64                    |
| 训练轮数 (epochs)                 | 8                     |
| 学习率 (lr)                      | 1e-05                 |
| 最大序列长度 (max_len)              | 80                    |
| Dropout率 (dropout_rate)       | 0.5                   |
| 冻结BERT层数 (freeze_bert_layers) | 0                     |
| 毒性类别数 (num_toxic_types)       | 6                     |
| 双融合                           | toxic_ids, stance_ids |
| 数据预处理 | 更新版                   |




## 实验

### 实验1：冻结额外嵌入层
* 变量：默认冻结0层freeze_bert_layers = 0(都不冻结，全部更新参数)，
把toxic_ids和stance_ids嵌入层参数冻结，其余不变。
`self.toxic_embeddings.weight.requires_grad = False`
`self.stance_embeddings.weight.requires_grad = False`

* 已知不冻结的情况下：双融合>不融合>单一融合

冻结实验：

| 冻结 |         |                       |               |                       |
|:---|:-------:|:---------------------:|:-------------:|:---------------------:|
| **评估指标** | **双融合** | **单一融合(融合toxic_ids)** | **解冻所有(双融合)** | **解冻所有(融合toxic_ids)** |
| 精确率 (Precision) | 0.8029  |        0.8015         |    0.8136     |        0.8037         |
| 召回率 (Recall) | 0.8030  |        0.7987         |    0.8142     |        0.8045         |
| F1 分数 (F1 Score) | 0.8005  |        0.7995         |    0.8138     |        0.8039         |
| 准确率 (Accuracy) | 0.8005  |        0.8009         |    0.8142     |        0.8042         |

发现：
* 冻结后性能下降，说明额外嵌入层必须可训练
* 单一融合效果差是因为信号不足导致过拟合，双融合通过信号互补形成正则化

结论：
* toxic_ids为6维融合，stance_ids为2维融合，即使双融合，相比于纯BERT也提升极小，
说明上述融合方式maybe相比于BERT强大的理解能力是不足够的。
* 额外添加的嵌入层必须可训练。

### 实验2：不同随机种子试验
* 变量：不同随机试验种子，其余不变

在测试集上：

|Seed|F1 Score|Precision|Recall|Accuracy
|:---|:----:|:----:|:----:|:-------:|
1|0.8138|0.8136|0.8142|0.8142
123|0.8041|0.8040|0.8042|0.8046
2024|0.8018|0.8016|0.8024|0.8022
42|0.8043|0.8070|0.8034|0.8059
7777|0.8168|0.8170|0.8167|0.8175

* 对于F1，在0.8018~0.8168，最大差距1.5%，说明随机性带来的波动不可忽视。

结论：
* 不要报告单次试验，应该多次取平均值
* maybe需要进行统计显著性检验

### 实验3：消融实验
实验数据在experiments/总结(0411-164206~185624)/compare.md中

结论：
* toxic_ids和stance_ids在BERT嵌入层的融合可能是不足够的










### 实验二：修改融合方式
* 变量：融合方式修改为

### 实验三：添加类似MLP的门控机制
gate = torch.sigmoid(self.fusion_gate(embeddings))
embeddings = embeddings + gate * toxic_emb + (1-gate) * stance_emb