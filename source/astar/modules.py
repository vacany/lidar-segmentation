import torch
from planning import algo
import numpy as np
import datatools


def get_device(gpu=0):
    if torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = 'cpu'
    return device

class Scalar(torch.nn.Module):
    def __init__(self, thet):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(thet, dtype=torch.float32))#.uniform_())

    def forward(self, x):
        return self.theta * x

class TrajectoryLoss(torch.nn.Module):
    def __init__(self, max_margin=0.2):
        super().__init__()
        self.max_margin = torch.nn.Parameter(torch.tensor(max_margin), requires_grad=True)

    def forward(self, cost_map, trajectory, pred, seg):
        trajectory = trajectory.cpu()
        cost_map = cost_map.cpu()

        best_traj = algo.astar_torch(trajectory[0], trajectory[-1], cost_map)
        expert_cost = algo.compute_cost(trajectory, cost_map)
        best_cost = algo.compute_cost(best_traj, cost_map)

        return expert_cost - best_cost , best_traj, expert_cost, best_cost

class Cost_map_transformer(torch.nn.Module):
    def __init__(self, dt_mode='conv', MAX_DIST=20, dt_adress='NN/dt_kernel.npy'):
        super().__init__()
        self.MAX_DIST = MAX_DIST
        self.distance_tranform = torch.nn.Conv2d(1, 1, MAX_DIST * 2 + 1, padding=MAX_DIST, bias=False)
        self.distance_tranform.weight = torch.nn.Parameter((torch.from_numpy(np.load(dt_adress))), requires_grad=False)
        self.softplus = torch.nn.functional.softplus
        self.dt_mode = dt_mode

    def forward(self, cost_map_batch, TRANS_MAT):

        ### COST MAP PART
        if len(cost_map_batch.shape) == 3:
            cost_map_batch = cost_map_batch.unsqueeze(1)

        # Neural network model or ...
        if self.dt_mode == 'conv':
            cost_map_batch = self.distance_tranform(cost_map_batch) # distance transform / cost map model
            cost_map_batch = self.softplus(cost_map_batch)

        # opencv distance transform
        elif self.dt_mode == 'cv':
            new = torch.zeros(cost_map_batch.shape)
            for i in range(len(cost_map_batch)):
                cm = datatools.distance(np.array(cost_map_batch.detach().cpu().numpy()[i][0], dtype=np.uint8), self.MAX_DIST)
                new[i] = torch.tensor(cm, dtype=torch.float32).unsqueeze(0)
            cost_map_batch = new.clone()

        return cost_map_batch

class Odo_matrix_Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, cost_map_batch, TRANS_MAT):
        ### TRANSFORMATION PART
        h = cost_map_batch.shape[-2]
        w = cost_map_batch.shape[-1]

        # Reference frame
        ref_rot_mat = TRANS_MAT[:1] # reference system - first frame in batch
        #rot_mat = TRANS_MAT @ np.linalg.inv(ref_rot_mat)     # all in ref frame
        #rot_mat = ref_rot_mat @ np.linalg.inv(TRANS_MAT)


        rot_mat = np.linalg.inv(ref_rot_mat) @ TRANS_MAT


        rot_mat = torch.tensor(rot_mat[:, :2, [0, 1, -1]], dtype=torch.float32)

        rot_mat[:, [0,1], 2] = rot_mat[:, [1,0], 2]
        rot_mat[:, 1, 2] = -rot_mat[:,1,2]
        rot_mat[:, 0, 2] = (rot_mat[:, 0, 2] * 2 / w + rot_mat[:, 0, 0] + rot_mat[:, 0, 1] - 1) / 0.3
        rot_mat[:, 0, 1] = rot_mat[:, 0, 1] * h / w

        rot_mat[:, 1, 2] = (rot_mat[:, 1, 2] * 2 / h + rot_mat[:, 1, 0] + rot_mat[:, 1, 1] - 1) / 0.3
        rot_mat[:, 1, 0] = rot_mat[:, 1, 0] * w / h


        return rot_mat


class Grid_Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cost_map_batch, rot_mat):
        grid = torch.nn.functional.affine_grid(rot_mat, cost_map_batch.shape).to(cost_map_batch.device) # around middle?
        output = torch.nn.functional.grid_sample(cost_map_batch, grid, mode='nearest', padding_mode='border')

        return output

class Odo_position_Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cost_map_batch, rot_mat):

        bev = torch.zeros(cost_map_batch.shape, dtype=torch.float32)
        bev[..., 500, 200] = 30

        ### TRANSFORMATION OF ODOMETRY AND EXPERT TAJECTORY
        grid = torch.nn.functional.affine_grid(rot_mat, cost_map_batch.shape).to(cost_map_batch.device)
        odo_grid = torch.nn.functional.grid_sample(bev, grid, mode='nearest', padding_mode='border')

        odo_traj = torch.where(odo_grid != 0)
        odo_traj = torch.stack((odo_traj[-2], odo_traj[-1]), dim=1)
        odo = odo_traj.clone()  # odometry points only

        odo_traj = datatools.line_traj(odo_traj)
        odo_traj = torch.tensor(odo_traj)

        return odo, odo_traj

