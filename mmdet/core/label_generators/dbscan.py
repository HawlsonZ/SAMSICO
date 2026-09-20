# Written by Yixiao Ge

import collections

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from .compute_dist import build_dist

from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator  # 需要安装kneed包
import hashlib
__all__ = ["label_generator_dbscan_single", "label_generator_dbscan"]

# 动态参数缓存
_eps_cache = {}

def auto_eps_estimator(features, k=5, plot_knee=False):
    """基于k-距离图的动态EPS估计器,带缓存和采样优化"""
    data_hash = hashlib.md5(features.numpy()).hexdigest()
    if data_hash in _eps_cache:
        return _eps_cache[data_hash]
    if len(features) > 10000:
        indices = np.random.choice(len(features), 5000, replace=False)
        sample_features = features[indices]
    else:
        sample_features = features
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(sample_features)
    distances, _ = nn.kneighbors(sample_features)
    k_distances = np.sort(distances[:, -1])
    kneedle = KneeLocator(
        x=range(len(k_distances)), 
        y=k_distances,
        curve='convex',
        direction='increasing',
        interp_method='interp1d'
    )
    eps_value = kneedle.knee_y if kneedle.knee else np.median(k_distances)
    _eps_cache[data_hash] = eps_value
    return eps_value

def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

@torch.no_grad()
def label_generator_dbscan_single(cfg, features, dist, eps, **kwargs):
    assert isinstance(dist, np.ndarray)

    # clustering
    min_samples = cfg.PSEUDO_LABELS.min_samples
    use_outliers = cfg.PSEUDO_LABELS.use_outliers

    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1,)
    # cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=4, )
    labels = cluster.fit_predict(dist)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # cluster labels -> pseudo labels
    # compute cluster centers
    centers = collections.defaultdict(list)
    outliers = 0
    for i, label in enumerate(labels):
        if label == -1:
            if not use_outliers:
                continue
            labels[i] = num_clusters + outliers
            outliers += 1

        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers, dim=0)
    labels = to_torch(labels).long()
    num_clusters += outliers

    return labels, centers, num_clusters


@torch.no_grad()
def label_generator_dbscan(cfg, features, cuda=True, indep_thres=None, **kwargs):
    assert cfg.PSEUDO_LABELS.cluster == "dbscan"

    if not cuda:
        cfg.PSEUDO_LABELS.search_type = 3

    # compute distance matrix by features
    dist = build_dist(cfg.PSEUDO_LABELS, features, verbose=True)

    features = features.cpu()

    # 动态生成三级EPS参数
    def get_dynamic_eps():
        base_eps = auto_eps_estimator(
            features.numpy(), 
            k=cfg.PSEUDO_LABELS.min_samples
        )
        eps_min = getattr(cfg.PSEUDO_LABELS, 'eps_min', 0.1)
        eps_max = getattr(cfg.PSEUDO_LABELS, 'eps_max', 2.0)
        base_eps = np.clip(base_eps, eps_min, eps_max)
        scales = getattr(cfg.PSEUDO_LABELS, 'eps_scales', [0.7, 1.0, 1.3])
        return sorted([base_eps * s for s in scales])
    
    eps_list = get_dynamic_eps()
    print(f"[Clustering] Dynamic EPS: {eps_list}")
    # 三级聚类流程
    labels_tight, _, _ = label_generator_dbscan_single(cfg, features, dist, eps_list[0])
    labels_normal, _, num_classes = label_generator_dbscan_single(cfg, features, dist, eps_list[1])
    labels_loose, _, _ = label_generator_dbscan_single(cfg, features, dist, eps_list[2])
    # 可靠性验证（带采样优化）
    N = labels_normal.size(0)
    sample_size = 5000 if N > 10000 else N
    def get_sim_matrix(labels, sample_size):
        indices = torch.randperm(N)[:sample_size]
        return (
            labels[indices].unsqueeze(1) == labels.unsqueeze(0)
        ).float()
        sim = get_sim_matrix(labels_normal, sample_size)
    sim_tight = get_sim_matrix(labels_tight, sample_size)
    sim_loose = get_sim_matrix(labels_loose, sample_size)

    # 计算可靠性指标
    R_comp = 1 - torch.min(sim, sim_tight).sum(-1) / torch.max(sim, sim_tight).sum(-1)
    R_indep = 1 - torch.min(sim, sim_loose).sum(-1) / torch.max(sim, sim_loose).sum(-1)

    cluster_metrics = collections.defaultdict(lambda: {'comp': [], 'indep': [], 'count': 0})
    for comp, indep, label in zip(R_comp, R_indep, labels_normal):
        cluster_metrics[label.item()]['comp'].append(comp.item())
        cluster_metrics[label.item()]['indep'].append(indep.item())
        cluster_metrics[label.item()]['count'] += 1
    valid_clusters = [k for k, v in cluster_metrics.items() if v['count'] > 1]
    cluster_R_indep = [min(cluster_metrics[k]['indep']) for k in valid_clusters]
    if indep_thres is None and len(cluster_R_indep) > 0:
        indep_thres = np.percentile(cluster_R_indep, 90)
    # 异常簇重建
    centers = collections.defaultdict(list)
    new_labels = labels_normal.clone()
    outlier_idx = 0
    
    for idx, (label, comp, indep) in enumerate(zip(labels_normal, R_comp, R_indep)):
        label = label.item()
        if label == -1:
            continue
            
        cluster_comp = min(cluster_metrics[label]['comp'])
        if (indep > indep_thres) or (comp > cluster_comp):
            if cluster_metrics[label]['count'] > 1:
                new_labels[idx] = num_classes + outlier_idx
                outlier_idx += 1
        centers[new_labels[idx].item()].append(features[idx])

    # 构建最终中心
    final_centers = [
        torch.stack(centers[idx], dim=0).mean(0) 
        for idx in sorted(centers.keys())
    ]
    return (
        new_labels,
        torch.stack(final_centers, dim=0),
        num_classes + outlier_idx,
        indep_thres
    )