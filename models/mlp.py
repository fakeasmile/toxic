import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5, hidden_features=96):
        """
        概念瓶颈分类器，使用矩阵门控。

        结构: 门控(sigmoid(W@x)) → Dropout → FC(in→hidden) → ReLU → Dropout → FC(hidden→2)

        Args:
            in_features: 输入特征维度（概念向量维度）
            dropout_rate: dropout 比率
            hidden_features: 隐藏层维度
        """
        super(MLP, self).__init__()

        # ========== 门控单元 ==========
        # 矩阵门控：gate = sigmoid(W @ x)，样本相关
        self.gate_layer = nn.Linear(in_features, in_features)

        # ========== dropout ==========
        self.dropout = nn.Dropout(dropout_rate)

        # ========== 分类层 ==========
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, feature_vector):
        # 门控
        gate_weights = torch.sigmoid(self.gate_layer(feature_vector))
        x = feature_vector * gate_weights

        # dropout
        x = self.dropout(x)

        # 分类
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class FormConditionedMLP(nn.Module):
    """Form-Conditioned Gate概念瓶颈分类器。

    核心思路：文本形式特征(form)不参与分类，而是调节语义概念的门控。
    gate = sigmoid(W_sem @ sem + W_form @ form)
    form作为"旁证"修正语义gate的倾向，分类路径仍为177维概念。

    Args:
        in_features: 语义概念向量维度（177）
        form_dim: 文本形式特征维度（10）
        dropout_rate: dropout比率
        hidden_features: 隐藏层维度
    """

    def __init__(self, in_features, form_dim, dropout_rate=0.5, hidden_features=96):
        super(FormConditionedMLP, self).__init__()

        # ========== 门控单元 ==========
        # 语义自门控 + form条件化调整
        # form_gate_layer无bias：让gate_layer的bias统一承担偏置，避免冗余
        self.gate_layer = nn.Linear(in_features, in_features)
        self.form_gate_layer = nn.Linear(form_dim, in_features, bias=False)

        # ========== dropout ==========
        self.dropout = nn.Dropout(dropout_rate)

        # ========== 分类层 ==========
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, sem_vector, form_vector):
        # 门控：sem自门控 + form条件化调整
        gate_weights = torch.sigmoid(
            self.gate_layer(sem_vector) + self.form_gate_layer(form_vector)
        )
        x = sem_vector * gate_weights

        # dropout
        x = self.dropout(x)

        # 分类
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_gate_values(self, sem_vector, form_vector):
        """返回门控值，用于诊断分析。

        Returns:
            gate_total: sigmoid(W_sem @ sem + W_form @ form) 完整门控
            gate_sem_only: sigmoid(W_sem @ sem) 纯语义门控
            gate_form_raw: W_form @ form form对门控的原始贡献（sigmoid前）
        """
        with torch.no_grad():
            gate_sem = self.gate_layer(sem_vector)
            gate_form = self.form_gate_layer(form_vector)
            gate_total = torch.sigmoid(gate_sem + gate_form)
            gate_sem_only = torch.sigmoid(gate_sem)
        return gate_total, gate_sem_only, gate_form


class SimpleMLP(nn.Module):
    """无门控的简单MLP分类器。

    结构: FC(in→hidden) → ReLU → Dropout → FC(hidden→2)

    Args:
        in_features: 输入特征维度
        dropout_rate: dropout比率
        hidden_features: 隐藏层维度
    """

    def __init__(self, in_features, dropout_rate=0.5, hidden_features=96):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FormConditionedSimpleMLP(nn.Module):
    """Form-Conditioned无门控MLP。

    核心思路：form特征作为隐藏层的偏置调整，条件化语义特征的解读。
    h = ReLU(W @ sem + W_form @ form + b)
    form只影响隐藏层激活阈值，不改变分类路径的输入维度。

    Args:
        in_features: 语义概念向量维度（177）
        form_dim: 文本形式特征维度（10）
        dropout_rate: dropout比率
        hidden_features: 隐藏层维度
    """

    def __init__(self, in_features, form_dim, dropout_rate=0.5, hidden_features=96):
        super(FormConditionedSimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.form_bias = nn.Linear(form_dim, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sem_vector, form_vector):
        # form特征调整隐藏层偏置，条件化语义解读
        x = self.fc1(sem_vector) + self.form_bias(form_vector)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
