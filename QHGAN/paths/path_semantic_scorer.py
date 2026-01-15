import torch
from paths.semantic_encoder import QuantumSemanticEncoder

def compute_path_semantics(data, meta_path_dict, device="cpu"):
    """
    为每条路径对 (start, end) 计算语义相似度分数
    输入:
      - data: 包含节点特征的原始图数据
      - meta_path_dict: 来自 quantum_walk.py 的路径集合
    输出:
      - meta_path_scores: dict，键为元路径，值为列表 [(start, end, prob, semantic_score)]
    """
    node_features = data["features"]
    encoder = QuantumSemanticEncoder(n_qubit=data["n_qubit"])
    encoder.to(device)
    encoder.eval()

    meta_path_scores = {}

    for meta_path, path_list in meta_path_dict.items():
        scored_paths = []

        for (start, end, prob) in path_list:
            # 获取节点特征（转 float + to device）
            feat_a = torch.tensor(node_features[start], dtype=torch.float32, device=device)
            feat_b = torch.tensor(node_features[end], dtype=torch.float32, device=device)

            with torch.no_grad():
                sim_score = encoder(feat_a, feat_b).item()

            scored_paths.append((start, end, prob, sim_score))

        meta_path_scores[meta_path] = scored_paths

    return meta_path_scores