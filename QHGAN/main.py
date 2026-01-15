from data.load_data import load_dblp_dataset
from utils.graph_utils import build_hetero_graph, extract_meta_paths
from paths.quantum_walk import quantum_random_walk
import networkx as nx
import random
from collections import defaultdict
from paths.path_semantic_scorer import compute_path_semantics
from paths.semantic_encoder import train_encoder
import torch


def sample_fixed_size_subgraph(G, num_nodes=100):
    """
    从图中随机选择 num_nodes 个节点及其相邻边，形成子图。
    """

    selected_nodes = random.sample(list(G.nodes()), num_nodes)
    subgraph = G.subgraph(selected_nodes).copy()
    print(f"[INFO] 固定采样子图：{subgraph.number_of_nodes()} 节点，{subgraph.number_of_edges()} 边")
    return subgraph


def sample_subgraph(G, center_nodes, hops=2):
    """
    从异质图中采样多个中心节点的 ego-subgraph
    """
    nodes = set()
    for node in center_nodes:
        ego_nodes = nx.single_source_shortest_path_length(G, node, cutoff=hops).keys()
        nodes.update(ego_nodes)

    subgraph = G.subgraph(nodes).copy()
    print(f"[INFO] 采样子图包含 {subgraph.number_of_nodes()} 个节点，{subgraph.number_of_edges()} 条边")
    return subgraph


if __name__ == "__main__":
    data = load_dblp_dataset("data/dblp_processed.pt")
    G = build_hetero_graph(data['features'], data['years'], data['original_features'])

    # ✅ 固定采样100个节点构成的子图
    sampled_G = G.subgraph(random.sample(list(G.nodes), 100)).copy()
    print(f"[INFO] 子图节点数: {sampled_G.number_of_nodes()}，边数: {sampled_G.number_of_edges()}")

    meta_paths = extract_meta_paths(sampled_G, max_hops=3)
    print(f"[INFO] 元路径种类数：{len(meta_paths)}")
    for path, count in list(meta_paths.items())[:10]:
        print(f"  Path {path}: {count} 次")

# 从 sampled_G 中选 3 个起点测试 QWalk

    meta_path_dict = defaultdict(list)
    start_nodes = list(sampled_G.nodes())[:3]  # 你可以改多一点

    for node in start_nodes:
        paths = quantum_random_walk(
            sampled_G,
            start_node=node,

            max_hops=3,
            walk_per_node=100,
            coin_bias={"F": 0.7, "Y": 0.3}
        )
        for path_key, triplets in paths.items():
            meta_path_dict[path_key].extend(triplets)

    # semantic scoring
    semantic_path_scores = compute_path_semantics(data, meta_path_dict,
                                                  device="cuda" if torch.cuda.is_available() else "cpu")

    for meta_path, entries in semantic_path_scores.items():
        print(f"[Path {meta_path}] Top examples:")
        for (s, e, p, sim) in entries[:3]:
            print(f"  {s} -> {e} | prob={p:.3f}, semantic_sim={sim:.3f}")

    trained_encoder = train_encoder(data, semantic_path_scores, epochs=20, lr=1e-3, device="cuda")