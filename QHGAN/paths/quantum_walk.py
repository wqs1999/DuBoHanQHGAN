# paths/quantum_walk.py

import random
from collections import defaultdict

def quantum_random_walk(graph, start_node, max_hops=3, walk_per_node=100, coin_bias=None):
    """
    模拟量子随机游走生成路径及其终点信息（简化模拟）
    返回：
      meta_path_dict: dict
        { meta_path_tuple: [ (start_node, end_node, prob), ... ] }
    """
    if coin_bias is None:
        coin_bias = {"F": 0.5, "Y": 0.5}

    coin_types = list(coin_bias.keys())
    path_dict = defaultdict(list)
    path_counter = defaultdict(int)

    for _ in range(walk_per_node):
        current_node = start_node
        path = []

        for _ in range(max_hops):
            neighbors = list(graph[current_node])
            valid_choices = []

            for nbr in neighbors:
                edge_data = graph.get_edge_data(current_node, nbr)
                for edge_type in edge_data.keys():
                    if edge_type in coin_types:
                        valid_choices.append((edge_type, nbr))

            if not valid_choices:
                break

            weights = [coin_bias[et] for et, _ in valid_choices]
            total = sum(weights)
            weights = [w / total for w in weights]
            edge_type, next_node = random.choices(valid_choices, weights=weights, k=1)[0]

            path.append(edge_type)
            current_node = next_node

        if path:
            path_tuple = tuple(path)
            path_counter[path_tuple] += 1
            path_dict[path_tuple].append((start_node, current_node))  # (start, end)

    # 转换为带概率的形式
    meta_path_dict = defaultdict(list)
    for path_tuple, pairs in path_dict.items():
        total_count = len(pairs)
        for start, end in pairs:
            prob = 1.0 / total_count
            meta_path_dict[path_tuple].append((start, end, prob))

    return dict(meta_path_dict)
