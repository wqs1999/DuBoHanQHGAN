import torch


def load_dblp_dataset(file_path):
    """
    加载已处理的 DBLP 数据集（.pt 文件）
    返回字典形式，包括 features, labels 等
    """
    data = torch.load(file_path)
    print("[INFO] 数据字段:", list(data.keys()))

    return {
        'features': data['init'],  # shape: (10000, 8)
        'train_features': data['train_init'],
        'test_features': data['test_init'],
        'labels': data['labels'],  # shape: (10000,)
        'train_labels': data['train_labels'],
        'test_labels': data['test_labels'],
        'original_features': data['original_features'],  # shape: (10000, 3)
        'years': data['years'],
        'n_qubit': data['n_qubit'],
        'n_class': data['n_class'],
        'scaler': data['feature_scaler']
    }