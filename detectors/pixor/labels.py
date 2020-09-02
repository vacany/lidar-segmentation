import numpy as np
import torch
from utils import get_points_in_a_rotated_box, trasform_label2metric
import os
import glob
import model
from tools.visuals import Bev_drawer


class Valeo_pixor():
    def __init__(self, seq_path):
        self.seq_path = seq_path
        self.dest = os.path.expanduser("~") + '/datasets/trajectories/'
        self.geometry = {
            'L1': -40.0,
            'L2': 40.0,
            'W1': 0.0,
            'W2': 70.0,
            'H1': -2.5,
            'H2': 1.0,
            'input_shape': (800, 700, 36),
            'label_shape': (200, 175, 7)
        }

        self.target_mean = np.array([0.008, 0.001, 0.202, 0.2, 0.43, 1.368])
        self.target_std_dev = np.array([0.866, 0.5, 0.954, 0.668, 0.09, 0.111])

        self.gt = np.load(glob.glob(f'{part_folder}/20*')[0], allow_pickle=True)
        self.ps = np.load(part_folder + '/psvh.npy')
        self.clca = np.load(part_folder + '/clca.npz', allow_pickle=True)

        self.v_l = self.gt['vehicle_list']
        self.trans = self.gt['odom_list']['transformation']

    def create_label_data(self):

        for frame in range(1, len(self.trans)):
            label_map = np.zeros(self.geometry['label_shape'], dtype=np.float32)
            bbox = []
            for car in self.v_l:
                if frame in car['frame']:
                    bbox.append(self.bounding_box_format(car[car['frame'] == frame]))

            self.__update_labels(label_map, bbox)

            # Normalin, change values on different dataset, maybe dont do at all
            cls_map = label_map[..., 0]
            reg_map = label_map[..., 1:]
            index = np.nonzero(cls_map)
            reg_map[index] = (reg_map[index] - self.target_mean) / self.target_std_dev

            label_map = np.concatenate((cls_map[..., None], reg_map), axis=2)
            label_map = torch.tensor(label_map).permute(2, 0, 1).unsqueeze(0).detach().numpy()

            np.save(f'{self.dest}/{frame:04d}.npy', label_map)
            print(frame)


    def bounding_box_format(self, car):
        car_bbox = np.array((car['x'][0][0], car['x'][0][1], 0, car['bb'][0][3] - car['bb'][0][1], car['bb'][0][2] - car['bb'][0][0], 3, car['x'][0][2]))
        return car_bbox

    def __update_labels(self, label_map, bounding_box):
        for bbox in bounding_box:
            corners, reg_target = self.get_corners(bbox)
            self.update_label_map(label_map, corners, reg_target)

    def update_label_map(self, label_map, bev_corners, reg_target):
        label_corners = (bev_corners / 4 ) / 0.1
        label_corners[:, 1] += self.geometry['label_shape'][0] / 2

        points = get_points_in_a_rotated_box(label_corners)

        for p in points:
            label_x = p[0]
            label_y = p[1]
            metric_x, metric_y = trasform_label2metric(np.array(p))
            actual_reg_target = np.copy(reg_target)
            actual_reg_target[2] = reg_target[2] - metric_x
            actual_reg_target[3] = reg_target[3] - metric_y
            actual_reg_target[4] = np.log(reg_target[4])
            actual_reg_target[5] = np.log(reg_target[5])

            label_map[label_y, label_x, 0] = 1.0
            label_map[label_y, label_x, 1:7] = actual_reg_target

    def get_corners(self, bbox):
        x, y, z, w, l, h, yaw = bbox
        y = -y
        # manually take a negative s. t. it's a right-hand system, with
        # x facing in the front windshield of the car
        # z facing up
        # y facing to the left of driver

        yaw = -(yaw + np.pi / 2)
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]

        return bev_corners, reg_target

if __name__ == "__main__":

    part_folder = '../sample/valeo/part2/'
    dataset = Valeo_pixor(part_folder)
    dataset.create_label_data()
    # dec = model.Decoder()
    # out, centre_x, centre_y = dec(reg)
    #
    # drawer = Bev_drawer()
    #
    # x_coors = centre_x[0,0][mask[0] != 0] # x coordinate for car object in each corresponding cell with confidence above treshold
    # y_coors = centre_y[0, 0][mask[0] != 0]
    #
    # x = x_coors.mean() # need to find way to extract the most probable bounding box, maybe window or approximation
    # y = y_coors.mean()
    #
    # unique_x = torch.unique(x_coors)
    # index = torch.where(centre_x == unique_x[0])
    #
    # x = centre_x[0,0,78,118]



    # print(f'{x}, {y}, \t done')
