import cv2
import os
from scipy.io import loadmat
import os.path as osp
import numpy as np
import json
from PIL import Image
import pickle
import re

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
import sys
from iou_utils import get_max_iou
import torch

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def set_box_pid(boxes, box, pids, pid):
    for i in range(boxes.shape[0]):
        if np.all(boxes[i] == box):
            pids[i] = pid
            return
    print("Person: %s, box: %s cannot find in images." % (pid, box))

def image_path_at(data_path, image_index, i):
    image_path = osp.join(data_path, image_index[i])
    assert osp.isfile(image_path), "Path does not exist: %s" % image_path
    return image_path

def load_image_index(root_dir, db_name):
    """Load the image indexes for training / testing."""
    # Test images
    test = loadmat(osp.join(root_dir, "annotation", "pool.mat"))
    test = test["pool"].squeeze()
    test = [str(a[0]) for a in test]
    if db_name == "psdb_test":
        return test

    # All images
    all_imgs = loadmat(osp.join(root_dir, "annotation", "Images.mat"))
    all_imgs = all_imgs["Img"].squeeze()
    all_imgs = [str(a[0][0]) for a in all_imgs]

    # Training images = all images - test images
    train = list(set(all_imgs) - set(test))
    train.sort()
    return train

def _get_cam_id(im_name):
        match = re.search('c\d', im_name).group().replace('c', '')
        return int(match)

def load_probes(root):
    query_info = osp.join(root, 'query_info.txt')
    with open(query_info, 'r') as f:
        raw = f.readlines()

    probes = []
    for line in raw:
        linelist = line.split(' ')
        pid = int(linelist[0])
        x, y, w, h = float(linelist[1]), float(
            linelist[2]), float(linelist[3]), float(linelist[4])
        roi = np.array([x, y, x + w, y + h]).astype(np.int32)
        roi = np.clip(roi, 0, None)  # several coordinates are negative
        im_name = linelist[5][:-1] + '.jpg'
        probes.append({'im_name': im_name,
                        'boxes': roi[np.newaxis, :],
                        # Useless. Can be set to any value.
                        'gt_pids': np.array([pid]),
                        'flipped': False,
                        'cam_id': _get_cam_id(im_name)
                        })

    return probes

def gt_roidbs(root):
    imgs = loadmat(
                osp.join(root, 'frame_test.mat'))['img_index_test']
    imgs = [img[0][0] + '.jpg' for img in imgs]

    gt_roidb = []
    for im_name in imgs:
        anno_path = osp.join(root, 'annotations', im_name)
        anno = loadmat(anno_path)
        box_key = 'box_new'
        if box_key not in anno.keys():
            box_key = 'anno_file'
        if box_key not in anno.keys():
            box_key = 'anno_previous'

        rois = anno[box_key][:, 1:]
        ids = anno[box_key][:, 0]
        rois = np.clip(rois, 0, None)  # several coordinates are negative

        assert len(rois) == len(ids)

        rois[:, 2:] += rois[:, :2]
        # num_objs = len(rois)
        # overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # overlaps[:, 1] = 1.0
        # overlaps = csr_matrix(overlaps)
        gt_roidb.append({
            'im_name': im_name,
            'boxes': rois.astype(np.int32),
            'gt_pids': ids.astype(np.int32),
            'flipped': False,
            'cam_id': _get_cam_id(im_name)
            # 'gt_overlaps': overlaps
        })
    return gt_roidb

# @jit(forceobj=True)
if __name__ == "__main__":
    results_path = './work_dirs/prw_dicl/'

    data_root = '/media/yangxilab/DiskB/tmh/datasets/PRW-v16.04.20/'
    gallery_set = gt_roidbs(data_root)

    with open(os.path.join(results_path, 'results_1000.pkl'), 'rb') as fid:
        all_dets = pickle.load(fid)
    
    gallery_det, gallery_feat = [], []
    for det in all_dets:
        gallery_det.append(det[0][:, :5])
        # feat = normalize(det[0][:, 5:], axis=1)
        # gallery_feat.append(feat)
        if det[0].shape[0] > 0:
            feat = normalize(det[0][:, 5:], axis=1)
        else:
            feat = det[0][:, 5:]
        # feat = normalize(det[0][:, 5:], axis=1)
        gallery_feat.append(feat)
   
    
    gallery_det, gallery_feat = [], []
    for det in all_dets:
        # det[0] = det[0][det[0][:, 4]>thresh]
        gallery_det.append(det[0][:, :5])
        if det[0].shape[0] > 0:
            feat = normalize(det[0][:, 5:], axis=1)
        else:
            feat = det[0][:, 5:]
        # feat = normalize(det[0][:, 5:], axis=1)
        gallery_feat.append(feat)

    gallery_feat_gt = []
    gallery_feat_label = []
    for gt, det, feat in zip(gallery_set, gallery_det, gallery_feat):
        for i in range(len(gt['gt_pids'])):
            if gt['gt_pids'][i] < 0:
                continue
            else:
                iou, iou_max, nmax = get_max_iou(det, gt['boxes'][i])
                if iou_max < 0.5:
                    print("not detected", gt['im_name'], iou_max) ###
                gallery_feat_gt.append(feat[nmax])    
                gallery_feat_label.append(gt['gt_pids'][i])

    sampled_list = [882, 554, 739, 523, 635, 691, 643, 157, 773, 645]
    color=['r','orange','gray','yellow','g','cyan','b','violet','pink','k']

    sampled_list_to_color = dict(zip(sampled_list, color))
    sampled_list_label = []
    sampled_list_feat  = []
    sampled_list_color = []
    for i in range(len(gallery_feat_label)):
        if gallery_feat_label[i] in sampled_list:
            sampled_list_label.append(gallery_feat_label[i])
            sampled_list_feat.append(gallery_feat_gt[i][np.newaxis,:])
            sampled_list_color.append(sampled_list_to_color[gallery_feat_label[i]])
    print(len(sampled_list_label))
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(np.concatenate(sampled_list_feat, axis=0))
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color=sampled_list_color)
    plt.savefig("vis.png")


def plot_embedding(resultq, resultg, query_id, gallery_id, title):
    #print('resultq',resultq.type)
    #print('resultg',resultg.shape)
    data = np.concatenate((resultq, resultg),axis=0)
    print('data',data.shape)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    x_min = x_min - 10
    x_max = x_max + 10 
    resultq = (resultq - x_min) / (x_max - x_min)
    resultg = (resultg - x_min) / (x_max - x_min)
    #print('resultq',resultq.shape)
    #print('resultg',resultg.shape)
    # plt.cm.Set1(i)
    fig = plt.figure()
    ax = plt.subplot(111)
    qid=list(set(query_id))
    print('qid',qid)

    gid=list(set(gallery_id))
    print('gid',gid)
    #resultq = resultq[:20]
    #print('###########')
    #print('resultq',resultq.shape)
    #resultg = resultg[:20]
	# print('gid',gid)
	
	# cmap = get_cmap(len(qid))
	# cmap1 = get_cmap(len(gid))
    color=['r','orange','gray','yellow','g','cyan','b','violet','pink','k']
    for i in range(resultq.shape[0]):
	    # print('query_id[i]',query_id[i])
		# print('resultq[i, 0]', resultq[i, 0])
		# print('resultq[i, 1]', resultq[i,1])
        print(qid.index(query_id[i]))
        if(qid.index(query_id[i]) < 10):
            plt.text(resultq[i, 0], resultq[i, 1], '*',
                color=color[qid.index(query_id[i])],
                fontdict={'weight': 'bold', 'size':10})
        print('query_id[i]',query_id[i])
    for i in range(resultg.shape[0]):
		# print('gallery_id',gallery_id[i] )
        if(gid.index(gallery_id[i]) < 10):
            plt.text(resultg[i, 0],resultg[i, 1], '.',
                color=color[gid.index(gallery_id[i])],
                fontdict={'weight': 'bold', 'size':20})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig   

