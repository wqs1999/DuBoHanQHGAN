import torch
from torch_geometric.data import HeteroData

# 加载图数据
graph = torch.load("dblp_sample.pt")

# 打印图结构基本信息
print("=" * 50)
print("图结构基本信息:")
print("=" * 50)
print(graph)

# 打印节点数量
print("\n节点类型及数量:")
for node_type in graph.node_types:
    print(f"{node_type}: {graph[node_type].num_nodes}")

# 打印边关系及数量
print("\n边关系及数量:")
for edge_type in graph.edge_types:
    src, rel, dst = edge_type
    num_edges = graph[edge_type].edge_index.shape[1]
    print(f"({src} → {rel} → {dst}): {num_edges} 条边")

# 查看元数据 - 修正部分
print("\n元数据内容:")
# 获取元数据元组
metadata_tuple = graph.metadata()

# 首先打印元数据的结构和类型
print(f"元数据类型: {type(metadata_tuple)}")
print(f"元数据长度: {len(metadata_tuple)}")
print("元数据内容:")
for i, item in enumerate(metadata_tuple):
    print(f"索引 {i}: 类型={type(item)}, 内容={item}")

# 尝试提取可能的编码器信息
if len(metadata_tuple) > 0:
    # 第一个元素通常是节点类型列表
    print(f"\n节点类型列表: {metadata_tuple[0]}")

if len(metadata_tuple) > 1:
    # 第二个元素通常是边类型列表
    print(f"边类型列表: {metadata_tuple[1]}")

# 检查是否有额外的元数据存储位置
if hasattr(graph, 'author_encoder'):
    print("\n在graph对象上找到 author_encoder 属性")
    print(f"作者编码器类型: {type(graph.author_encoder)}")

if hasattr(graph, 'venue_encoder'):
    print(f"会议编码器类型: {type(graph.venue_encoder)}")

if hasattr(graph, 'term_encoder'):
    print(f"术语编码器类型: {type(graph.term_encoder)}")

if hasattr(graph, 'paper_mapping'):
    print(f"论文映射数量: {len(graph.paper_mapping)} 条映射")
else:
    print("\n没有找到预期的编码器属性")

# 查看论文节点特征示例
print("\n论文节点特征示例:")
print(f"特征维度: {graph['paper'].x.shape}")
print("前5篇论文特征:")
print(graph['paper'].x[:5])

# 查看作者-论文边示例
if ('author', 'writes', 'paper') in graph.edge_types:
    edge_index = graph['author', 'writes', 'paper'].edge_index
    print("\n作者-论文边示例 (前5条):")
    print(edge_index[:, :5])