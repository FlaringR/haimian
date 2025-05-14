import pandas as pd  
import os  

def load_and_merge_predictions(root_folder: str, feather_file_path: str) -> pd.DataFrame:  
    """  
    加载指定根文件夹下的所有 predictions.csv 文件，并与指定的 feather 文件合并。  

    参数:  
    root_folder (str): 存放子文件夹的路径。  
    feather_file_path (str): 要合并的 feather 文件的完整路径。  

    返回:  
    pd.DataFrame: 合并后的 DataFrame。  
    """  
    # 创建一个空的列表来存储 DataFrame  
    dataframes = []  

    # 遍历根文件夹下的所有子文件夹  
    for folder in sorted(os.listdir(root_folder)):  
        folder_path = os.path.join(root_folder, folder)  

        # 检查当前路径是否是文件夹  
        if os.path.isdir(folder_path):  
            # 构建 predictions.csv 的完整路径  
            csv_file_path = os.path.join(folder_path, 'predictions.csv')  

            # 检查文件是否存在  
            if os.path.isfile(csv_file_path):  
                # 读取 CSV 文件并添加到 DataFrame 列表  
                df = pd.read_csv(csv_file_path)  
                dataframes.append(df)  
    print("file num:", len(dataframes))
    # 合并所有 DataFrame  
    combined_df = pd.concat(dataframes, ignore_index=True)  

    # 转换 Time 列为 datetime 类型，并设置时区  
    combined_df["Time"] = pd.to_datetime(combined_df['Time'], format='ISO8601').dt.tz_convert('Asia/Shanghai')  

    # 读取 feather 文件  
    y_tag = pd.read_feather(feather_file_path)  

    # 合并 DataFrame  
    result_df = pd.merge(combined_df, y_tag, on="Time", how="left")  

    # 过滤结果，保留 y_action 为 1 的行  
    # result_df = result_df[(result_df['y60_duo_class_action'] == 1) & (result_df['y180_duo_class_action'] == 1)]  
    result_df = result_df[result_df['y60_duo_class_action'] == 1]

    return result_df  

# 使用示例  
# 设置参数  
root_folder = '/data/home/lichengzhang/zhoujun/Rehaimian/logs/TransformerModel_5'  
feather_file_path = '/data/home/lichengzhang/zhoujun/HaimianData/20250422/y_tag_496_300750.SZ.ftr'  

# 调用函数  
final_result = load_and_merge_predictions(root_folder, feather_file_path)  

# 输出结果描述  
print(final_result.describe())  