import pandas as pd
import time


# 多粒度
def multi_dfs(df, n):
    dfs = divide(df, n)
    s = []
    for i in dfs:
        eq_dict = equivalence_class(i)
        s.append(eq_dict)
    return s


def equivalence_multi(df, n):
    result: dict = {}
    multi_df = multi_dfs(df, n)
    for i in df.index.tolist():
        i_class = []
        for j in multi_df:
            i_class.append(j[i])
        result[i] = i_class
    return result


# 下近似：且， 上近似：或
def lowapr_multi(df, X, n):
    LowAprSet = []
    multi_dict = equivalence_multi(df, n)
    for i in df.index.tolist():
        panduan: bool = True
        for j in multi_dict[i]:
            if not set(j).issubset(set(X)):
                panduan = False
        if panduan:
            LowAprSet.append(i)
    return LowAprSet


def upapr_multi(df, X, n):
    UpAprSet = []
    multi_dict = equivalence_multi(df,n)
    for i in df.index.tolist():
        panduan: bool = False
        for j in multi_dict[i]:
            if not set(j).isdisjoint(set(X)):
                panduan = True
        if panduan:
            UpAprSet.append(i)
    return UpAprSet