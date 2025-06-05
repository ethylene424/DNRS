import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import nonzero, array
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, rand_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


df = pd.read_excel(f"D:\\Articles\\Main\\2.Neighborhood DMF\\NeighborDMF.dataset\\maxminscaler\\Segmentation.xlsx", header=(None))  # 设置要读取的数据集

columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns) - 1]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
attributes = len(df.columns) - 1  # 属性数量（数据集维度）
original_labels = list(df[columns[-1]])  # 原始标签


def initialize_centroids(data, k):
    # 从数据集中随机选择k个点作为初始质心
    centers = data[np.random.choice(data.shape[0], k, replace=False)]
    return centers


def get_clusters(data, centroids):
    # 计算数据点与质心之间的距离，并将数据点分配给最近的质心
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels


def update_centroids(data, cluster_labels, k):
    # 计算每个簇的新质心，即簇内数据点的均值
    new_centroids = np.array([data[cluster_labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


def k_means(data, k, T, epsilon):
    start = time.time()  # 开始时间，计时
    # 初始化质心
    centroids = initialize_centroids(data, k)
    t = 0
    while t <= T:
        # 分配簇
        cluster_labels = get_clusters(data, centroids)

        # 更新质心
        new_centroids = update_centroids(data, cluster_labels, k)

        # 检查收敛条件
        if np.linalg.norm(new_centroids - centroids) < epsilon:
            break
        centroids = new_centroids
        print("第", t, "次迭代")
        t += 1
    print("用时：{0}".format(time.time() - start))
    return cluster_labels, centroids


# 计算聚类指标
def clustering_indicators(labels_true, labels_pred):
    if type(labels_true[0]) != int:
        labels_true = LabelEncoder().fit_transform(df[columns[len(columns) - 1]])  # 如果数据集的标签为文本类型，把文本标签转换为数字标签
    f_measure = f1_score(labels_true, labels_pred, average='macro')  # F值
    accuracy = accuracy_score(labels_true, labels_pred)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI
    ARI = adjusted_rand_score(labels_true, labels_pred)
    return f_measure, accuracy, normalized_mutual_information, rand_index, ARI


if __name__ == "__main__":
    k = 20  # 聚类簇数
    T = 100  # 最大迭代数
    n = len(dataset)  # 样本数
    epsilon = 1e-5
    # 预测全部数据
    labels, centers = k_means(np.array(dataset), k, T, epsilon)
    cluster_dict: dict = {}
    for i in range(k):
        cluster_dict[i] = []
    for i in range(len(labels)):
        cluster_dict[labels[i]].append(i)
    print(cluster_dict)

