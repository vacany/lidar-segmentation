import torch
import ctypes
import numpy as np
from time import time
import inspect
from os.path import abspath, dirname, join

def best_trajectory(cm, tr, move_cost):
    opt_traj = Astar(cm.detach().cpu().numpy(), tr[0], tr[-1], move_cost)

    return opt_traj

def compute_cost(cost_map, trajectory, move_cost):
    if type(trajectory) == np.ndarray:
        trajectory = torch.tensor(trajectory, dtype=torch.float)

    cost_from_move = (((trajectory[1:] - trajectory[:-1]).to(torch.float32) ** 2).sum(1) ** 0.5).mean() * move_cost
    cost_from_map = cost_map[trajectory[1:, 0], trajectory[1:, 1]].mean()

    return cost_from_move + cost_from_map

def filter_band(tr, exp):
    if type(tr) == np.ndarray or type(exp) == np.ndarray:
        tr = torch.tensor(tr, dtype=torch.long)
        exp = torch.tensor(exp, dtype=torch.long)

    exp_one = torch.cat((exp, exp + torch.tensor((1, 0)), exp + torch.tensor((-1, 0))), dim=0)
    allec = torch.cat((tr, exp_one, exp_one), dim=0)
    u, c = torch.unique(allec, return_counts=True, dim=0)
    rest_traj = u[c == 1]

    return rest_traj

def read_cpp():
    fname = abspath(inspect.getfile(inspect.currentframe()))
    lib = ctypes.cdll.LoadLibrary(join(dirname(fname), 'astar.so'))
    astar = lib.astar
    ndmat_f_type = np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
    ndmat_i_type = np.ctypeslib.ndpointer(
        dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
    move_f_type = np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=0, flags='C_CONTIGUOUS')
    astar.restype = ctypes.c_bool
    astar.argtypes = [ndmat_f_type, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, move_f_type,
                      ndmat_i_type]
    return astar

def Astar(weights, start, goal, move_cost):

    astar = read_cpp()

    if type(weights) == torch.Tensor:
        weights = weights.detach().cpu().numpy()
    if type(start) == torch.Tensor:
        start = start.detach().cpu().numpy()
    if type(goal) == torch.Tensor:
        goal = goal.detach().cpu().numpy()
    if type(move_cost) == torch.Tensor:
        move_cost = move_cost.detach().cpu().numpy()
    move_cost = np.array(move_cost).astype('f4')

    height, width = weights.shape
    start_idx = np.ravel_multi_index(start, (height, width))
    goal_idx = np.ravel_multi_index(goal, (height, width))



    # The C++ code writes the solution to the paths array
    paths = np.full(height * width, -1, dtype=np.int32)
    success = astar(weights.flatten(), height, width, start_idx, goal_idx, move_cost,
        paths  # output parameter
    )


    if not success:
        return np.array([])


    coordinates = []
    path_idx = goal_idx
    while path_idx != start_idx:
        pi, pj = np.unravel_index(path_idx, (height, width))
        coordinates.append((pi, pj))

        path_idx = paths[path_idx]

    if coordinates:
        coordinates.append(np.unravel_index(start_idx, (height, width)))
        return np.vstack(coordinates[::-1])
    else:
        print('no path found')
        return np.array([])
