import gzip
import pickle
import torch
import numpy as np
import scipy as sp
from scipy.sparse import spmatrix  # 导入稀疏矩阵基类


def inspect_dblp_file(file_path):
    try:
        # 尝试直接加载（未压缩的pt文件）
        try:
            data = torch.load(file_path)
            print("文件加载成功（普通pt格式）")
            return data
        except:
            # 尝试作为gzip压缩的pickle文件加载
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            print("文件加载成功（gzip压缩格式）")
            return data
    except Exception as e:
        print(f"文件加载失败: {str(e)}")
        return None


def analyze_structure(data, indent=0):
    """递归分析数据结构"""
    indent_str = "  " * indent

    if isinstance(data, dict):
        print(f"{indent_str}字典 (长度: {len(data)})")
        for key, value in data.items():
            print(f"{indent_str}├─ 键: {key} ({type(value).__name__})")
            analyze_structure(value, indent + 1)

    elif isinstance(data, list):
        print(f"{indent_str}列表 (长度: {len(data)})")
        if len(data) > 0:
            print(f"{indent_str}└─ 首元素类型: {type(data[0]).__name__}")
            analyze_structure(data[0], indent + 1)

    elif isinstance(data, tuple):
        print(f"{indent_str}元组 (长度: {len(data)})")
        for i, item in enumerate(data):
            print(f"{indent_str}├─ [{i}]: {type(item).__name__}")
            analyze_structure(item, indent + 1)

    elif isinstance(data, torch.Tensor):
        print(f"{indent_str}Torch张量 (形状: {data.shape}, 类型: {data.dtype})")

    elif isinstance(data, np.ndarray):
        print(f"{indent_str}NumPy数组 (形状: {data.shape}, 类型: {data.dtype})")

    # 修复：使用正确的稀疏矩阵检查方法
    elif isinstance(data, spmatrix):  # 使用导入的spmatrix基类
        print(f"{indent_str}稀疏矩阵 (形状: {data.shape}, 格式: {type(data).__name__})")

    # 或者使用属性检查方法（更通用）
    # elif hasattr(data, 'format') and hasattr(data, 'shape') and hasattr(data, 'nnz'):
    #     print(f"{indent_str}稀疏矩阵 (形状: {data.shape}, 格式: {data.format})")

    else:
        print(f"{indent_str}{type(data).__name__}")


if __name__ == "__main__":
    file_path = "dblp_processed.pt"
    data = inspect_dblp_file(file_path)

    if data is not None:
        print("\n文件内容结构分析:")
        analyze_structure(data)

        print("\n可能的键列表:")
        if isinstance(data, dict):
            print(list(data.keys()))
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            print("索引访问: data[0], data[1], etc.")

        print("\n保存示例内容到 sample_output.txt...")
        with open("sample_output.txt", "w") as f:
            f.write("文件结构:\n")
            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"{key}: {type(value).__name__}\n")
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        f.write(f"  形状: {value.shape}\n")
                    elif isinstance(value, spmatrix):  # 使用导入的spmatrix基类
                        f.write(f"  形状: {value.shape}\n")
                    # 或者使用属性检查方法
                    # elif hasattr(value, 'format') and hasattr(value, 'shape') and hasattr(value, 'nnz'):
                    #     f.write(f"  形状: {value.shape}\n")
            f.write("\n前10个标签(如果存在):\n")
            if "label" in data or "labels" in data:
                labels = data.get("label", data.get("labels", None))
                f.write(str(labels[:10]) + "\n")
            elif isinstance(data, tuple) and len(data) > 1:
                f.write(str(data[1][:10]) + "\n")

        print("分析完成! 请查看 sample_output.txt 获取详细信息")