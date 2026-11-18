import numpy as np
import random

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            features = [float(x) for x in parts[:4]]
            data.append(features)
    return np.array(data)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def init_centroids(data, k):
    random.seed(42)
    indices = random.sample(range(len(data)), k)
    return data[indices]

def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, c) for c in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        points = data[clusters == i]
        if len(points) == 0:
            new_centroids.append(data[random.randint(0, len(data)-1)])
        else:
            new_centroids.append(np.mean(points, axis=0))
    return np.array(new_centroids)

def kmeans(data, k, max_iters=100, tol=1e-6):
    centroids = init_centroids(data, k)
    for i in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        diff = np.sum(np.abs(new_centroids - centroids))
        if diff < tol:
            break
        centroids = new_centroids
    return centroids, clusters

if __name__ == '__main__':
    data_path = '../data/iris.data'
    k = 3
    data = load_data(data_path)
    centroids, clusters = kmeans(data, k)

    output = []
    output.append("=== 单机K‑Means结果 ===")
    output.append(f"聚类数: {k}")
    output.append(f"样本数: {len(data)}")
    output.append("\n最终质心坐标:")
    for i, c in enumerate(centroids):
        output.append(f"  簇{i}: {np.round(c, 4).tolist()}")
    output.append("\n各簇样本数量:")
    for i in range(k):
        count = np.sum(clusters == i)
        output.append(f"  簇{i}: {count} 个样本")
    output.append("\n前10个样本的聚类结果:")
    for i in range(10):
        output.append(f"  样本{i}: 簇{clusters[i]}")

    result_str = '\n'.join(output)
    print(result_str)
    with open('../results/single_result.txt', 'w') as f:
        f.write(result_str)

