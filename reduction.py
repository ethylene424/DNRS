import pandas as pd


# 基于X的下近似集不变的约简
def AR_L(df, X):
    A = df.columns.tolist()
    uselessA = []
    for i in A:
        dfi = df.drop(i, axis=1)
        L_df = classic_low_apr(df, X)
        L_dfi = classic_low_apr(dfi, X)
        if set(L_df) == set(L_dfi):
            uselessA.append(i)
            df = dfi
    return list(set(A).difference(set(uselessA)))
