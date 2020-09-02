import torch
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import itertools as it

def get_device(gpu=0):
    if torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device='cpu'
    return device

def sample_dat(a, start, end, dest):
    d = {}
    if type(a) == str:
        a = np.load(a, allow_pickle=True)
    for key in a.files:
        if key in ['transformation', 'info', 'data_file']:
            d[key] = a[key]
        if key in ['scans', 'gt', 'ps', 'frame', 'bevs']:
            d[key] = a[key][start : end]

    np.savez_compressed(dest, **d)

def sample_key(a, key, dest):
    d = {}
    d[key] = a[key][a['frame']]

    for k in a.files:
        if k not in ['info', key]:
            d[k] = a[k]
    np.savez_compressed(dest, **d)

def wombo_combo(config):
    values = config.values()
    combination = list(it.product(*values))
    combo_list = []
    for combo in combination:
        d = {key: combo[i] for i, key in enumerate(config.keys())}
        combo_list.append(d)

    return combo_list

def class_weights(gt):
    cls, w = np.unique(gt, return_counts=True)
    inverse = 1/w
    return cls, inverse / sum(inverse)

def intermediates(p1, p2, nb_points=8):
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[p1[0] + i * x_spacing, p1[1] +  i * y_spacing]
            for i in range(1, nb_points+1)]

def line_traj(odo, hack=True):
    odo = np.array(odo, dtype=np.int)
    traj = np.array([(odo[0])], dtype=np.int)

    for i in range(1, len(odo)):

        num_of_point = np.max(( abs(odo[i][0] - odo[i - 1][0]), abs(odo[i][1] - odo[i - 1][1]) ))

        if num_of_point == 0:
            num_of_point = 1

        points = intermediates(odo[i-1], odo[i], num_of_point)
        points = np.array(points)
        traj = np.concatenate((traj, np.array([odo[i-1]])), axis=0)
        traj = np.concatenate((traj, points), axis=0)   # intermediates only

    if len(odo) != 1:
        traj = np.concatenate((traj, np.array([odo[i]])), axis=0)
    else:
        traj = odo

    # specific hack
    if hack:
        traj = np.array(np.round(traj), dtype=int)
        u, ind = np.unique(traj[:,1], return_index=True, axis=0)
        traj = traj[ind]

    traj = traj[traj[:, 1].argsort()]

    return traj


def pick_trajectory(transes, frame, ego, cell_size, number_of_frames):
    positions = transes[..., -1]
    trans = transes[frame]
    pts = (np.linalg.inv(trans) @ positions.T).T
    odo = np.round(pts[frame: frame + number_of_frames, :2] / cell_size + ego).astype('i4')
    traj = line_traj(odo)

    return traj, odo

def exclude_trajectory_points(best, exp):
    """ Works only for same lenght and sorted trajectories in one direction"""
    exp = torch.tensor(exp, dtype=torch.long).to(best.device)
    mask = (best != exp).any(1)
    return best[mask], exp[mask]

def lidar_map(scan, bev, ego, cell_size):

    xy = np.round(scan['x'] / cell_size + ego).astype('i4')
    bev[xy[:, 0], xy[:, 1], 0] = scan['z'] / 2
    bev[xy[:, 0], xy[:, 1], 1] = scan['i']
    bev[xy[:, 0], xy[:, 1], 2] = 1  # scan points

    return bev

def raycasted_lidar_map(scan, scan2, bev, ego, trans, f1, f2, cell_size):
    s1 = np.concatenate((scan['x'], np.expand_dims(scan['z'], axis=1)), axis=1)
    s2 = np.concatenate((scan2['x'], np.expand_dims(scan2['z'], axis=1)), axis=1)

    s1 = np.insert(s1, 3, 1, axis=1)
    s2 = np.insert(s2, 3, 1, axis=1)

    t1 = trans[f1]
    t2 = trans[f2]

    Transform = (t2.T @ np.linalg.inv(t1).T)
    s2 = s2 @ Transform

    positions = trans[..., -1]
    tran = trans[f1]
    pts = (np.linalg.inv(tran) @ positions.T).T


    odo = np.round(pts[f1: f2, :2] / cell_size + ego).astype('i4')

    ego2 = odo[-1]

    # filter
    s1 = s1[abs(s1[:,0]) < 120]
    s2 = s2[abs(s2[:, 0]) < 120]

    xy = np.round(s1[:, :2] / cell_size + ego).astype('i4')
    bev[xy[:, 0], xy[:, 1], 0] = s1[:,2] / 2
    bev[xy[:, 0], xy[:, 1], 1] = 0
    bev[xy[:, 0], xy[:, 1], 2] = 1
    # raycasting
    points = np.unique(xy, axis=0)  # new points

    points = points[(points[:, 0] > ego[0])]
    rays1 = []
    for point in points:
        traj = line_traj(np.stack((ego, point)))
        traj = traj[traj[:, 0].argsort()]
        # if (bev[traj[:-1, 0], traj[:-1, 1], -1] == 1).any():
        #     mask = bev[traj[:, 0], traj[:, 1], -1]
        #     ind = np.argwhere(mask == 1)
        #     traj = traj[:ind[0][0]]
        rays1.append(traj)




    xy2 = np.round(s2[:, :2] / cell_size + ego2).astype('i4')
    # bev[xy2[:, 0], xy2[:, 1], 0] = s2[:, 2] / 2
    # bev[xy2[:, 0], xy2[:, 1], 1] = 0
    # bev[xy2[:, 0], xy2[:, 1], 2] = 1
    #raycasting
    points2 = np.unique(xy2, axis=0)  # new points
    points2 = points2[(points2[:, 0] > ego2[0])]
    rays2 = []
    for point in points2:
        traj = line_traj(np.stack((ego2, point)))
        traj = traj[traj[:,0].argsort()]
        # if (bev[traj[:-1, 0], traj[:-1, 1], -1] == 1).any():
        #     mask = bev[traj[:, 0], traj[:, 1], -1]
        #     ind = np.argwhere(mask == 1)
        #     traj = traj[:ind[0][0]]
        rays2.append(traj)
        #     rays2[traj[:, 0], traj[:, 1]] = 1
        # else:
        #     rays2[traj[:, 0], traj[:, 1]] = 1

    rays1 = np.concatenate([*rays1], axis=0)
    rays2 = np.concatenate([*rays2], axis=0)

    rays1 = np.unique(rays1, axis=0)
    rays2 = np.unique(rays2, axis=0)

    bev[rays1[:,0], rays1[:,1], -2] = 1
    bev[rays2[:,0], rays2[:,1], -2] += 1
    # bev[...,-2] = rays1 + rays2

    return bev


def bbox_pts(position, bbox):
    """
    :param position: x, y, angle
    :param bbox: four-coordinated
    :return: corner points
    """
    pts = bbox.reshape(2, 2)
    pts = np.array(list(it.product(*pts.T)))
    pts_t = pts[2].copy()
    pts[2] = pts[3]
    pts[3] = pts_t
    ang = position[2]
    mat = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = (mat @ pts.T).T
    pts += position[:2][None, ...]
    return pts

def show_coordinates(traj):
    plt.plot(traj[:,0], traj[:,1], '*')
    plt.show()
