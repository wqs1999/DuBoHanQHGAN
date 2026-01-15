import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def process_dblp_data(file_path, n_qubit=3, train_ratio=0.8, random_state=42):
    """
    处理DBLP图数据，转换为量子图神经网络可用的格式

    参数:
        file_path (str): 图数据文件路径
        n_qubit (int): 量子比特数量
        train_ratio (float): 训练集比例
        random_state (int): 随机种子

    返回:
        dict: 包含处理后的数据和元信息
    """
    # 加载图数据
    graph = torch.load(file_path)

    # 提取论文特征和年份信息
    paper_features = graph['paper'].x.numpy()
    years = paper_features[:, 0]

    # 根据年份创建标签 (3个类别)
    labels = np.zeros(len(years), dtype=int)
    labels[years < 2000] = 0  # 2000年之前
    labels[(years >= 2000) & (years < 2005)] = 1  # 2000-2004
    labels[(years >= 2005) & (years < 2010)] = 2  # 2005-2009
    labels[years >= 2010] = 3  # 2010年之后

    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(paper_features)

    # 将特征投影到2^n_qubit维空间
    target_dim = 2 ** n_qubit
    if scaled_features.shape[1] < target_dim:
        # 随机投影到更高维度
        projection_matrix = np.random.randn(scaled_features.shape[1], target_dim)
        high_dim_features = scaled_features @ projection_matrix
    else:
        # 使用PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim)
        high_dim_features = pca.fit_transform(scaled_features)

    # 归一化量子态
    norms = np.linalg.norm(high_dim_features, axis=1, keepdims=True)
    quantum_states = high_dim_features / norms

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        quantum_states, labels,
        train_size=train_ratio,
        random_state=random_state,
        stratify=labels
    )

    # 准备完整数据集
    full_dataset = {
        "init": quantum_states,  # 所有量子态
        "train_init": X_train,  # 训练量子态
        "test_init": X_test,  # 测试量子态
        "labels": labels,  # 所有标签
        "train_labels": y_train,  # 训练标签
        "test_labels": y_test,  # 测试标签
        "n_qubit": n_qubit,  # 量子比特数
        "n_class": len(np.unique(labels)),  # 类别数
        "feature_scaler": scaler,  # 特征缩放器
        "original_features": paper_features,  # 原始特征
        "years": years  # 年份信息
    }

    # 打印数据集信息
    print("=" * 50)
    print("处理后的数据集信息:")
    print("=" * 50)
    print(f"总样本数: {len(quantum_states)}")
    print(f"训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    print(f"量子态维度: {quantum_states.shape[1]} (2^{n_qubit})")
    print(f"类别分布: {np.bincount(labels)}")
    print(f"训练集类别分布: {np.bincount(y_train)}")
    print(f"测试集类别分布: {np.bincount(y_test)}")

    return full_dataset


def get_dynamic_map(n_class, n_qubit):
    """
    动态生成量子态映射关系

    参数:
        n_class (int): 类别数量
        n_qubit (int): 量子比特数量

    返回:
        dict: 类别到量子态的映射
    """
    total_states = 2 ** n_qubit
    map_dict = {}

    # 每个类别分配大致相等数量的量子态
    states_per_class = total_states // n_class
    extra_states = total_states % n_class

    start_idx = 0
    for i in range(n_class):
        end_idx = start_idx + states_per_class
        if i < extra_states:
            end_idx += 1

        # 确保不超过总状态数
        end_idx = min(end_idx, total_states)

        # 分配连续的状态索引
        state_indices = list(range(start_idx, end_idx))
        map_dict[i] = state_indices

        start_idx = end_idx

    return map_dict


if __name__ == "__main__":
    # 处理数据
    dataset = process_dblp_data("dblp_sample.pt", n_qubit=3)

    # 示例：获取动态映射
    n_class = dataset["n_class"]
    n_qubit = dataset["n_qubit"]
    dynamic_map = get_dynamic_map(n_class, n_qubit)

    print("\n动态量子态映射:")
    for cls, states in dynamic_map.items():
        print(f"类别 {cls}: 量子态 {states}")

    # 示例：保存处理后的数据
    torch.save(dataset, "dblp_processed.pt")
    print("\n处理后的数据已保存到 'dblp_processed.pt'")