from Network.models import *
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import numpy as np
from Network.utils import read_mat

'''

 计算acc和kappa
 绘制classification map
 
 输入：
 pred  ： 被传入参数，来源于卷积神经网络的输出
 show  ： 是否可视化
 
 输出  ：  
 
 可视化结果，
 网络评价指标 OA ,AA ，kappa
 
'''
def cvt_map(pred, show=False, mapName='pred',gate=0):

    gth = read_mat(PATH, gth_test,'test')
    matrix = pred
    pred = np.argmax(pred, axis=1)
    pred = np.asarray(pred, dtype=np.int8) + 1
    index = np.load(os.path.join(SAVA_PATH, 'index.npy'))
    # index = index[..., 2832:15029]
    pred_map = np.zeros_like(gth)
    cls = []
    count = 0
    for i in range(index.shape[1]):
        if np.max(matrix[i,...]) > gate:
            pred_map[index[0, i], index[1, i]] = pred[i]
        cls.append(gth[index[0, i], index[1, i]])
    cls = np.asarray(cls, dtype=np.int8)
    if show:
        plt.imshow(pred_map)
        plt.figure()
        plt.imshow(gth)
        plt.show()
    name = mapName
    np.save(os.path.join(SAVA_PATH, name + '.npy'), pred_map)
    for i in range(index.shape[1]):
        if pred[i] == cls[i]:
            count = count + 1
    mx = confusion(pred - 1, cls - 1)
    ua = np.diag(mx) / np.sum(mx, axis=0)

    OA = 100.0 * count / np.sum(gth != 0)
    AA = 100.0 * np.sum(ua) / mx.shape[0]
    kappa = compute_Kappa(mx)
    return OA, AA, kappa


# def cvt_map2(pred, show=False):
#
#     gth = tiff.imread(os.path.join(PATH, gth_train))
#     pred = np.argmax(pred, axis=1)
#     pred = np.asarray(pred, dtype=np.int8) + 1
#     #print(pred)
#     # if gcn:
#     #     index = np.load(os.path.join(SAVA_PATH, 'index2.npy'))
#     #     index = index[...,2832:15028]
#     #     # index = np.load(os.path.join(SAVA_PATH, 'index.npy'))
#     # else:
#     #     index = np.load(os.path.join(SAVA_PATH, 'index2.npy'))
#     #     index = index[...,2832:15028]
#     index = np.load(os.path.join(SAVA_PATH, 'index2.npy'))
#     index = index[...,0:2832]
#     pred_map = np.zeros_like(gth)
#     cls = []
#     count = 0
#     for i in range(index.shape[1]):
#         pred_map[index[0, i], index[1, i]] = pred[i]
#         cls.append(gth[index[0, i], index[1, i]])
#     cls = np.asarray(cls, dtype=np.int8)
#     if show:
#         plt.imshow(pred_map)
#         plt.figure()
#         plt.imshow(gth)
#         plt.show()
#     for i in range(index.shape[1]):
#         if pred[i] == cls[i]:
#             count = count + 1
#     mx = confusion(pred - 1, cls - 1)
#     ua = np.diag(mx) / np.sum(mx, axis=0)
#
#     OA = 100.0 * count / np.sum(gth != 0)
#     AA = 100.0 * np.sum(ua) / mx.shape[0]
#     kappa = compute_Kappa(mx)
#     return OA, AA, kappa
'''

 产生并保存融合分类矩阵

'''
def confusion(pred, labels):

    mx = np.zeros((NUM_CLASS, NUM_CLASS))
    if len(pred.shape) == 2:
        pred = np.asarray(np.argmax(pred, axis=1))

    for i in range(labels.shape[0]):
        mx[pred[i], labels[i]] += 1
    mx = np.asarray(mx, dtype=np.int16)
    np.savetxt('confusion.txt', mx, delimiter=" ", fmt="%s")

    # print(mx)
    return mx

'''

 计算融合分类矩阵的kappa系数
 
'''
def compute_Kappa(confusion_matrix):

    N = np.sum(confusion_matrix)
    N_observed = np.trace(confusion_matrix)
    Po = 1.0 * N_observed / N
    h_sum = np.sum(confusion_matrix, axis=0)
    v_sum = np.sum(confusion_matrix, axis=1)
    Pe = np.sum(np.multiply(1.0 * h_sum / N, 1.0 * v_sum / N))
    kappa = (Po - Pe) / (1.0 - Pe)
    return kappa
