from pyspark import SparkContext, SparkConf
import numpy as np
import random

def parse_line(line):
    parts = line.strip().split(',')
    return np.array([float(x) for x in parts[:4]])

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def closest_centroid(point, centroids):
    distances = [euclidean_distance(point, c) for c in centroids]
    return int(np.argmin(distances))

def init_centroids(data_rdd, k):
    # 为和单机版本结果完全对齐：把全部数据拉到本地，用完全一样的采样逻辑
    data_local = data_rdd.collect()
    random.seed(42)
    indices = random.sample(range(len(data_local)), k)
    samples = [data_local[i] for i in indices]
    return np.array(samples)


def kmeans_spark(data_rdd, k, max_iters=100, tol=1e-6):
    centroids = init_centroids(data_rdd, k)
    for i in range(max_iters):
        assigned = data_rdd.map(lambda p: (closest_centroid(p, centroids), (p, 1)))
        cluster_sums = assigned.reduceByKey(
            lambda a, b: (a[0] + b[0], a[1] + b[1])
        ).mapValues(lambda x: x[0] / x[1]).collect()

        new_centroids = np.zeros_like(centroids)
        for idx, centroid in cluster_sums:
            new_centroids[idx] = centroid

        cluster_idx_list = [c[0] for c in cluster_sums]
        for idx in range(k):
            if idx not in cluster_idx_list:
                sample = data_rdd.takeSample(False, 1, seed=42)[0]
                new_centroids[idx] = sample

        diff = np.sum(np.abs(new_centroids - centroids))
        if diff < tol:
            break
        centroids = new_centroids

    final_clusters = data_rdd.map(lambda p: closest_centroid(p, centroids)).collect()
    return centroids, np.array(final_clusters)

if __name__ == '__main__':
    conf = SparkConf().setAppName("KMeansSpark").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    data_path = '../data/iris.data'
    k = 3
    lines = sc.textFile(data_path).filter(lambda x: x.strip() != '')
    data_rdd = lines.map(parse_line)

    centroids, clusters = kmeans_spark(data_rdd, k)

    output = []
    output.append("=== Spark并行化K‑Means结果 ===")
    output.append(f"聚类数: {k}")
    output.append(f"样本数: {clusters.shape[0]}")
    output.append("\n最终质心坐标:")
    for i, c in enumerate(centroids):
        output.append(f"  簇{i}: {np.round(c,4).tolist()}")
    output.append("\n各簇样本数量:")
    for i in range(k):
        count = np.sum(clusters == i)
        output.append(f"  簇{i}: {count} 个样本")
    output.append("\n前10个样本的聚类结果:")
    for i in range(10):
        output.append(f"  样本{i}: 簇{clusters[i]}")

    result_str = '\n'.join(output)
    print(result_str)
    with open('../results/spark_result.txt', 'w') as f:
        f.write(result_str)
    sc.stop()

