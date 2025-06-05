import pandas as pd
import time


# k 近邻
def mergesort(lst, inf_dict) -> list:
    # 合并左右子序列函数
    def merge(arr, left, mid, right):
        temp = []  # 中间数组
        i = left  # 左段子序列起始
        j = mid + 1  # 右段子序列起始
        while i <= mid and j <= right:
            if inf_dict[arr[i]] <= inf_dict[arr[j]]:
                temp.append(arr[i])
                i += 1
            else:
                temp.append(arr[j])
                j += 1
        while i <= mid:
            temp.append(arr[i])
            i += 1
        while j <= right:
            temp.append(arr[j])
            j += 1
        for i in range(left, right + 1):
            arr[i] = temp[i - left]

    # 递归调用归并排序
    def mSort(arr, left, right):
        if left >= right:
            return
        mid = (left + right) // 2
        mSort(arr, left, mid)
        mSort(arr, mid + 1, right)
        merge(arr, left, mid, right)

    n = len(lst)
    if n <= 1:
        return lst
    mSort(lst, 0, n - 1)
    return lst


def kn_neighbor_group(dataset: pd.DataFrame, delta):
    U = dataset.index.tolist()
    group: dict = {}
    for i in U:
        delta_granule = []
        dis_dict: dict = {}
        for j in U:
            dis_temp = dis(i, j)
            dis_dict[j] = dis_temp
            if dis_temp <= delta:
                delta_granule.append(j)
        if len(delta_granule) > k:
            delta_granule = mergesort(delta_granule, dis_dict)
            delta_granule = delta_granule[: k]
        group[i] = delta_granule
    return group


# 计算X的kn下近似集
def kn_neighbor_low_apr(df, delta, X):
    # 获取df的所有粒子族
    group = kn_neighbor_group(df, delta)
    # 计算X的邻域下近似集
    low_apr_set = set()
    for g in group:
        if set(g).issubset(set(X)):
            low_apr_set = low_apr_set.union(set(g))
    return list(low_apr_set)


# 计算X的kn上近似集
def kn_neighbor_up_apr(df, delta, X):
    # 获取df的所有粒子族
    group = kn_neighbor_group(df, delta)
    # 计算X的邻域上近似集
    up_apr_set = set()
    for g in group:
        if not set(g).isdisjoint(set(X)):
            up_apr_set = up_apr_set.union(set(g))
    return list(up_apr_set)