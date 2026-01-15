import numpy as np
import networkx as nx
from collections import defaultdict

def build_hetero_graph(features, years, original_features):
    """
    构建异质图结构：节点之间基于原始特征或年份信息构建边
    返回 NetworkX 的多类型图（MultiDiGraph）
    """
    G = nx.MultiDiGraph()

    num_nodes = features.shape[0]
    for i in range(num_nodes):
        G.add_node(i, feature=features[i], year=years[i], raw=original_features[i])

    # 示例边构造策略1：按年份相近度连接
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if abs(years[i] - years[j]) <= 1:
                G.add_edge(i, j, key='Y')  # Y表示年份近似关系
                G.add_edge(j, i, key='Y')

    # 示例边构造策略2：原始特征欧式距离小于阈值
    threshold = 3.5
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(original_features[i] - original_features[j])
            if dist < threshold:
                G.add_edge(i, j, key='F')  # F表示特征相似
                G.add_edge(j, i, key='F')

    print(f"[INFO] 构建完成异质图，共有 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
    return G

def extract_meta_paths(graph, max_hops=3):
    meta_paths = defaultdict(int)

    for start in graph.nodes():
        for path in nx.single_source_shortest_path(graph, start, cutoff=max_hops).values():
            if len(path) <= 1:
                continue
            edge_types = []
            for i in range(len(path) - 1):
                edge_data = graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    edge_type = list(edge_data.keys())[0]
                    edge_types.append(edge_type)
            if edge_types:
                meta_paths[tuple(edge_types)] += 1

    return dict(meta_paths)