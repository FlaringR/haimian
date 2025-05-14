import pandas as pd 

def rankic(factor: pd.Series, y_f: pd.Series) -> float:
        """计算因子与目标变量的斯皮尔曼相关系数（绝对值）。"""
        df = pd.concat([factor.loc[factor.notna()], y_f], axis=1)
        return abs(df.corr(method='spearman').iloc[0, 1])

def rankic_v2(factor: pd.Series, y_f: pd.Series, perc: float = 0.4) -> float:
    """计算因子与目标变量的部分数据斯皮尔曼相关系数。"""
    df = pd.concat([factor.loc[factor.notna()], y_f], axis=1)
    n = int(len(df) * perc)
    name = factor.name
    sorted_df = df.sort_values(by=[name], ascending=True)

    if name.startswith('AskPrice1_diff_1_over_LastPrice_dayk_nega_perc_ewm'):
        return -sorted_df.iloc[:n].corr(method='spearman').iloc[0, 1]
    else:
        return sorted_df.iloc[-n:].corr(method='spearman').iloc[0, 1]
