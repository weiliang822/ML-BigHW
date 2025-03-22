import numpy as np


class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters  # 聚类数量
        self.max_iter = max_iter  # 最大迭代次数
        self.random_state = random_state  # 随机数种子，确保结果可重现

    def initialize_centroids(self, X):
        # 随机初始化聚类中心
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[: self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        # 计算新的聚类中心
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        # 计算每个点到各个聚类中心的距离
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = np.linalg.norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        # 找到距离每个点最近的聚类中心
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        # 计算聚类的总平方误差（SSE）
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = np.linalg.norm(
                X[labels == k] - centroids[k], axis=1
            )
        return np.sum(np.square(distance))

    def fit(self, X):
        # 训练模型，执行 K-Means 聚类
        self.centroids = self.initialize_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels_ = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels_)
            if np.all(old_centroids == self.centroids):
                break
        self.inertia_ = self.compute_sse(X, self.labels_, self.centroids)

    def predict(self, X):
        # 预测数据点的聚类标签
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)


class MyAgglomerativeClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters  # 最终聚类的数量

    def fit(self, X):
        # 初始化每个点作为一个聚类
        clusters = list(range(X.shape[0]))
        self.labels_ = np.zeros(X.shape[0], dtype=np.int32)

        while len(set(clusters)) > self.n_clusters:
            # 计算所有聚类对之间的距离
            cluster_distance = np.inf * np.ones((len(X), len(X)))
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    if clusters[i] != clusters[j]:
                        cluster_distance[i, j] = np.linalg.norm(X[i] - X[j])

            # 找到距离最近的两个聚类
            min_distance_idx = np.unravel_index(
                np.argmin(cluster_distance, axis=None), cluster_distance.shape
            )
            min_i, min_j = min_distance_idx

            # 合并这两个聚类
            min_cluster = min(clusters[min_i], clusters[min_j])
            max_cluster = max(clusters[min_i], clusters[min_j])
            for k in range(len(clusters)):
                if clusters[k] == max_cluster:
                    clusters[k] = min_cluster

        # 为每个点分配最终的聚类标签
        unique_clusters = list(set(clusters))
        for i, cluster in enumerate(clusters):
            self.labels_[i] = unique_clusters.index(cluster)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


from sklearn.metrics.pairwise import euclidean_distances


class MyDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        # 计算所有点之间的距离
        distances = euclidean_distances(X)

        # 寻找核心点
        neighbors = [np.where(distances[i] <= self.eps)[0] for i in range(len(X))]
        core_points = np.array(
            [i for i in range(len(X)) if len(neighbors[i]) >= self.min_samples]
        )

        # 初始化聚类标签为 -1（未分类）
        labels = -1 * np.ones(X.shape[0], dtype=int)

        # 为每个核心点及其密度可达点分配聚类标签
        cluster_id = 0
        for point in core_points:
            if labels[point] == -1:  # 如果该核心点尚未被分类
                labels[point] = cluster_id
                self._expand_cluster(labels, neighbors, point, cluster_id)
                cluster_id += 1

        self.labels_ = labels

    def _expand_cluster(self, labels, neighbors, point, cluster_id):
        # 将所有密度可达的点分配到当前聚类
        points_to_check = [point]
        while points_to_check:
            current_point = points_to_check.pop()
            if labels[current_point] == -1 or labels[current_point] == cluster_id:
                labels[current_point] = cluster_id
                points_to_check.extend(
                    neighbors[current_point][labels[neighbors[current_point]] == -1]
                )

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


from sklearn.neighbors import KDTree


class MyMeanShift:
    def __init__(self, bandwidth=2, max_iter=300, tol=1e-3):
        self.bandwidth = bandwidth  # 窗口大小
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛容忍度

    def fit(self, X):
        # 初始化
        points = np.copy(X)
        for _ in range(self.max_iter):
            for i, point in enumerate(points):
                # 使用 KDTree 找到邻近点
                tree = KDTree(points)
                indices = tree.query_radius([point], r=self.bandwidth)[0]

                # 计算窗口内点的均值
                if len(indices) > 0:
                    points[i] = np.mean(X[indices], axis=0)

        # 聚类中心
        self.cluster_centers_ = np.array(
            [point for point in set(tuple(p) for p in points)]
        )

        # 聚类标签
        self.labels_ = np.array(
            [
                np.argmin(np.linalg.norm(point - self.cluster_centers_, axis=1))
                for point in X
            ]
        )

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
