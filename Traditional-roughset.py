import pandas as pd


# 计算欧氏距离
def dis(x: list, y: list):
    if len(x) != len(y):
        return 'error'
    import math
    d = 0
    for i in range(len(x)):
        d = d + (x[i] - y[i]) ** 2
    return math.sqrt(d)


# 计算一个信息表的邻域粒子族
def delta_neighbor_group(dataset: pd.DataFrame, delta):
    # 构造论域U和信息表R
    U = dataset.index.tolist()
    # 初始化领域粒子族为空
    group: dict = {}
    for i in U:
        delta_granule = []
        for j in U:
            if dis(dataset.iloc[j, :], dataset.iloc[i, :]) <= delta:
                delta_granule.append(j)
        group[i] = delta_granule
    return group


# 计算X的邻域下近似集
def neighbor_low_apr(df, delta, X):
    # 获取df的所有粒子族
    group = delta_neighbor_group(df, delta)
    # 计算X的邻域下近似集
    low_apr_set = set()
    for g in group:
        if set(g).issubset(set(X)):
            low_apr_set = low_apr_set.union(set(g))
    return list(low_apr_set)


# 计算X的邻域上近似集
def neighbor_up_apr(df, delta, X):
    # 获取df的所有粒子族
    group = delta_neighbor_group(df, delta)
    # 计算X的邻域上近似集
    up_apr_set = set()
    for g in group:
        if not set(g).isdisjoint(set(X)):
            up_apr_set = up_apr_set.union(set(g))
    return list(up_apr_set)


# 计算x的邻域粒子
def x_granule(df, delta, x):
    # 构造论域U和信息表R
    U = df.index.tolist()
    R: dict = {}
    for i in range(len(U)):
        R[U[i]] = list(df.iloc[i, :])
    # 遍历R中向量，计算x的领域粒子
    granule = []
    for i in U:
        if dis(R[x], R[i]) <= delta:
            granule.append(i)
    return granule
