import torch
import numpy as np
from torch.utils.data import Dataset
from tools.datafuns import line_traj

class NIPS_Dataset(Dataset):
    def __init__(self, data_file, device='cpu', NUM=50, every_th=1, task='upgrade', cost_input='ps', target='ps', source='valeo'):
        self.data = np.load(data_file, allow_pickle=True)
        self.source = source
        self.device = device
        self.every_th = every_th
        self.task = task
        self.EGO = self.data['info'].item()['ego']
        self.CELL_SIZE = self.data['info'].item()['CELL_SIZE']
        self.NUM = NUM
        self.orig_frames = self.data['frame']
        self.trans = self.data['transformation']
        self.data_file = data_file
        self.c_in = cost_input
        self.target = target

        # B, C, H, W
        self.input_data = self.data['scans']
        self.mask = self.data['mask']
        self.label_data = (self.data['gt'] + 1) * self.input_data[:, -1]
        self.gt_data = self.split_label(self.label_data)
        self.ps_data = (self.data['ps'] + 1) * self.input_data[:, -1]


        self.input_data = np.concatenate((self.input_data, np.expand_dims(self.ps_data, 1)),axis=1)

        # to torch
        self.input_data = torch.tensor(self.input_data, dtype=torch.float32)
        self.label_data = torch.tensor(self.label_data, dtype=torch.long)
        self.gt_data = torch.tensor(self.gt_data, dtype=torch.float32)
        self.ps_data = torch.tensor(self.ps_data, dtype=torch.float32)

    def __getitem__(self, index):
        # One, B, C, H, W
        data = {'in' : self.input_data[self.choose_frames(index).T].requires_grad_(True).to(self.device),
                'gt' : self.label_data[self.choose_frames(index).T].to(self.device),
                'orig_gt' : self.gt_data[self.choose_frames(index).T].to(self.device),
                'ps' : self.ps_data[self.choose_frames(index).T].to(self.device).requires_grad_(True),
                'frame' : int(index),
                'orig_frame' : self.orig_frames[index],
                'orig_future_frame' : self.orig_frames[self.choose_frames(index).T],
                'mask' : self.mask[self.choose_frames(index).T],
                }

        return data

    def __len__(self):

        return len(self.input_data) - self.NUM - self.every_th -1

    def choose_frames(self, index):
        return np.linspace(index, index + self.NUM - 1, int(self.NUM / self.every_th), dtype=np.int)

    def get_trajectories(self, index):
        #for frame in index:
        start = self.orig_frames[index]

        trans = self.trans[start]
        pts = (np.linalg.inv(trans) @ self.trans[..., -1].T).T
        odo = np.round(pts[start: start + self.NUM, :2] / self.CELL_SIZE + self.EGO).astype('i4')[:, [1, 0]]

        # clip and correction
        odo = odo[self.choose_frames(0)]
        indices = self.choose_frames(0)
        # mask
        mask = (odo[:, 0] < self.input_data.shape[-2]) & (odo[:, 1] < self.input_data.shape[-1])

        odo = odo[mask]

        #if self.source == 'cadc':
            #odo[:, 0] = odo[:, 0] - 2 * (odo[:, 0] - self.EGO[0])
            #traj[:, 0] = traj[:, 0] - 2 * (traj[:, 0] - self.EGO[0])
            # odo switch!

        odo = odo[(odo > 0).all(1)]

        traj = line_traj(odo)


        return traj, odo

    def split_gt(self, gt, scans):
        li = []
        li.append(scans[:, -1] == 0)
        li.append(((gt == 0) & (scans[:, -1] == 1)))
        li.append((gt == 1) & (scans[:, -1] == 1))
        if self.source == 'valeo':
            li.append((gt == 2) & (scans[:, -1] == 1))

        return np.stack(li, 1)

    def split_orig_gt(self, gt, scans):
        li = []
        li.append((scans == 0) & (gt == 0))
        li.append((gt == 0) & (scans == 1))
        li.append((gt == 1) & (scans == 1))
        li.append((gt == 2) & (scans == 1))


        return np.stack(li, 1)

    def split_label(self, gt):
        li = []
        li.append(gt==0)
        li.append(gt==1)
        li.append(gt==2)
        li.append(gt==3)
        return np.stack(li, 1)


class Seg_Dataset(Dataset):
    def __init__(self, data_file, device='cpu', source='valeo'):
        self.data = np.load(data_file, allow_pickle=True)
        self.source = source
        self.device = device
        self.orig_frames = self.data['frame']
        self.trans = self.data['transformation']
        self.data_file = data_file

        # B, C, H, W
        self.input_data = self.data['scans']
        self.ps = self.data['ps'] + 1
        self.ps[self.ps == 1] = self.ps[self.ps == 1] * self.input_data[:,-1][self.ps == 1]
        self.ps = self.ps * self.input_data[:,-1]
        self.input_data = np.concatenate((self.input_data, np.expand_dims(self.ps, 1)), axis=1)

        #self.label_data = (self.data['gt'] + 1) * self.input_data[:, -2]
        self.label_data = self.ps
        self.gt_data = (self.data['gt'] + 1) - np.array((self.data['gt'] == 0) & (self.input_data[:,-2] != 1))
        self.gt_data = self.split_ps(self.gt_data, self.input_data[:,-2:-1])

        # to torch
        self.input_data = torch.tensor(self.input_data, dtype=torch.float32)
        self.label_data = torch.tensor(self.label_data, dtype=torch.long)
        self.gt_data = torch.tensor(self.gt_data, dtype=torch.long)
        self.ps = torch.tensor(self.ps, dtype=torch.long)

    def __getitem__(self, index):

        data = {'in' : self.input_data[index].requires_grad_(True).to(self.device),
                'gt' : self.label_data[index].to(self.device),
                'orig_gt' : self.gt_data[index],
                'ps' : self.ps[index].to(self.device),
                'frame' : int(index),
                'orig_frame' : self.orig_frames[index],

                }


        return data

    def __len__(self):

        return len(self.input_data)

    def split_ps(self, ps, scans):

        li = []
        li.append(scans[:, -1] == 0)
        li.append(((ps == 0) & (scans[:, -1] == 1)))
        li.append((ps == 1) & (scans[:, -1] == 1))
        if self.source == 'valeo':
            li.append((ps == 2) & (scans[:, -1] == 1))

        return np.stack(li, axis=1).astype('u1')
