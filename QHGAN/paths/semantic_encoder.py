# semantic_encoder.py
import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
from qiskit_aer.primitives import Sampler as AerSampler
from sklearn.preprocessing import MinMaxScaler

class QuantumSemanticEncoder(nn.Module):
    def __init__(self, n_qubit=4, n_layer=1, shots=1024):
        super().__init__()
        self.n_qubit = n_qubit
        self.n_layer = n_layer
        self.shots = shots

        # 初始化参数（可训练）
        self.theta = nn.Parameter(torch.randn(n_layer * n_qubit * 3))  # RX/RY/RZ 参数

        # 后端和采样器
        self.backend = AerSimulator()
        self.sampler = AerSampler()

    def amplitude_encode(self, features):
        """特征归一化用于振幅编码"""
        norm = np.linalg.norm(features)
        return features / norm if norm != 0 else features

    def build_circuit(self, encoded_vector, params):
        """构建量子电路"""
        qc = QuantumCircuit(self.n_qubit)

        # 用于初始特征嵌入的 Ry 旋转
        for i in range(self.n_qubit):
            angle = encoded_vector[i] * np.pi
            qc.ry(angle, i)

        # 加入参数化量子层
        param_index = 0
        for _ in range(self.n_layer):
            for q in range(self.n_qubit):
                qc.rx(params[param_index], q); param_index += 1
                qc.ry(params[param_index], q); param_index += 1
                qc.rz(params[param_index], q); param_index += 1
            # 纠缠层
            for q in range(self.n_qubit - 1):
                qc.cz(q, q + 1)

        qc.measure_all()
        return qc

    def forward(self, node_feat_a, node_feat_b):
        """两个节点的特征输入，输出量子相似度分数"""
        assert node_feat_a.shape[0] == node_feat_b.shape[0]

        a_encoded = self.amplitude_encode(node_feat_a.detach().cpu().numpy())
        b_encoded = self.amplitude_encode(node_feat_b.detach().cpu().numpy())
        combined = (a_encoded + b_encoded) / 2

        # 构建电路并运行采样
        circuit = self.build_circuit(combined, self.theta.detach().cpu().numpy())
        result = self.sampler.run([circuit], shots=self.shots).result()

        # 获取测量结果并计算平均汉明权重作为相似度 proxy
        counts = result.quasi_dists[0]
        similarity_score = 0
        for bitstring, prob in counts.items():
            bitstring_bin = format(bitstring, f"0{self.n_qubit}b")  # 转换为固定宽度的二进制字符串
            hamming_weight = bitstring_bin.count("1")
            similarity_score += (1 - hamming_weight / self.n_qubit) * prob

        return torch.tensor(similarity_score, dtype=torch.float32).to(node_feat_a.device)

def train_encoder(data, meta_path_scores, epochs=10, lr=1e-3, device="cpu"):
    """
    使用路径终点的特征和其语义相似度训练量子编码器。
    输入:
      - data: 包含 'features' 和 'n_qubit'
      - meta_path_scores: {meta_path: [(start, end, prob, semantic_score)]}
    """
    import torch.optim as optim

    model = QuantumSemanticEncoder(n_qubit=data["n_qubit"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    features = data["features"]

    all_train_data = []
    for path_type, entries in meta_path_scores.items():
        for (start, end, _, score) in entries:
            all_train_data.append((start, end, score))

    for epoch in range(epochs):
        total_loss = 0.0

        for start, end, target_score in all_train_data:
            feat_a = torch.tensor(features[start], dtype=torch.float32).to(device)
            feat_b = torch.tensor(features[end], dtype=torch.float32).to(device)
            target = torch.tensor(target_score, dtype=torch.float32).to(device)

            pred = model(feat_a, feat_b)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(all_train_data)
        print(f"[Epoch {epoch + 1}] Avg Loss = {avg_loss:.6f}")

    return model


