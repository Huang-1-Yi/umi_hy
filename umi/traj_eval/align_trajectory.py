#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import umi.traj_eval.transformations as tfs

# 用于对齐两个点云，使其在旋转和缩放后尽可能匹配。如果需要，可以单独对齐旋转（仅 yaw），或者包括缩放
# 具体步骤如下：
    # 计算两个点云的均值，并将其从每个点云中减去，以获得零中心化的点云。
    # 计算点云之间的相关性矩阵C。
    # 对相关性矩阵C进行奇异值分解（SVD），得到矩阵U_svd、D_svd和V_svd。
    # 计算C的奇异值D_svd，并构建一个3x3的旋转矩阵R，该矩阵基于SVD的奇异值分解。
    # 如果yaw_only参数为True，则计算最佳的yaw角度并应用R。
    # 计算缩放因子s。如果known_scale参数为True，则使用预设的缩放因子；否则，根据奇异值分解计算。
    # 计算旋转后的均值差t。
    # 返回缩放因子s、旋转矩阵R和位移向量t。
def get_best_yaw(C):
    '''
    maximize trace(Rz(theta) * C)
    '''
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta


def rot_z(theta):
    R = tfs.rotation_matrix(theta, [0, 0, 1])
    R = R[0:3, 0:3]

    return R

# Umeyama方法是一种用于刚性变换参数估计的优化算法，它可以找到一个旋转矩阵R和一个缩放因子s，使得两个点云之间的误差最小化
def align_umeyama(model, data, known_scale=False, yaw_only=False):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)

    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
        S[2, 2] = -1

    if yaw_only:
        rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
        theta = get_best_yaw(rot_C)
        R = rot_z(theta)
    else:
        R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0/sigma2*np.trace(np.dot(D_svd, S))

    t = mu_M-s*np.dot(R, mu_D)

    return s, R, t
