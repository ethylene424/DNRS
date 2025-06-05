import pandas as pd
import time


# 局部邻域
def local_neighbor_group(dataset: pd.DataFrame, delta, x):
    # 构造论域U和信息表R
    U = dataset.index.tolist()
    # 初始化领域粒子族为空
    group: dict = {}
    for i in x:
        delta_granule = []
        for j in U:
            if dis(dataset.iloc[j, :], dataset.iloc[i, :]) <= delta:
                delta_granule.append(j)
        group[i] = delta_granule
    return group


def neighbor_local_low_apr(df, x, a, delta) -> list:
    granules = local_neighbor_group(df, delta, x)
    U = df.index.tolist()
    local_low = []
    for i in granules.keys():
        if len(list(set(granules[i]) & (set(x))))/len(U) > a:
            local_low.append(i)
    return local_low


def neighbor_local_up_apr(df, x, b) -> list:
    granules = local_neighbor_group(df, delta, x)
    U = df.index.tolist()
    local_up = []
    for i in granules.keys():
        if len(list(set(granules[i]) & (set(x))))/len(U) > b:
            local_up.append(i)
    return local_up