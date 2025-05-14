from typing import Dict
import pandas as pd

def preprocess_classification_labels(
    df: pd.DataFrame,
    threshold_map: Dict[str, float],
    output_suffix: str = "_class"
) -> pd.DataFrame:
    """为指定列生成二分类标签基于阈值

    参数:
        df: 输入 DataFrame，包含需要处理的列
        threshold_map: 字典，键为列名（如 'y60_duo'），值为阈值（如 0.0020）
        output_suffix: 新生成列的后缀，默认为 '_class'

    返回:
        pd.DataFrame: 包含新生成二分类标签列的 DataFrame
    """
    df = df.copy()  # 避免修改原始数据
    for col, threshold in threshold_map.items():
        output_col = f"{col}{output_suffix}"
        df[output_col] = df[col].apply(lambda x: 1 if x > threshold else 0).astype(int)
    return df