import numpy as np
import yaml
import os
import glob
from datatools import datafuns
import cv2

def create_data(data, clca, psvh, destination, config, chosen_frames='all', every_th=5):

    # Preload
    scans = clca['scan_list']
    vh = data['vehicle_list']
    if len(vh) > 0:
        vh = np.concatenate(vh)
    so = data['static_obj_list']
    if so.size:
        so = np.concatenate(so)
    else:
        so = np.zeros((0,), dtype=vh.dtype)

    rb_1 = data['rb_list'][0]['x']
    rb_2 = data['rb_list'][1]['x']
    rb_1 = np.concatenate((rb_1, np.ones(rb_1.shape)), 1)
    rb_2 = np.concatenate((rb_2, np.ones(rb_2.shape)), 1)
    rb_1[:, 2] = 0
    rb_2[:, 2] = 0

    # Trans
    transes = data['odom_list']['transformation']

    scan_list = []
    gt_list = []
    ps_list = []
    frame_id_list = []

    for frame_id in range(len(scans)):
        if chosen_frames != 'all':
            if frame_id not in chosen_frames:
                continue

        frame_id_list.append(frame_id)
        scan = scans[frame_id]
        trans = transes[frame_id]

        # Start looping
        psvh_s = psvh[psvh['frame'] == (frame_id + 1)]
        vh_s = vh[vh['frame'] == (frame_id + 1)]


        min_dims = np.concatenate(scans)['x'].min(0)
        max_dims = np.concatenate(scans)['x'].max(0)
        big_shape = np.array((np.floor(max_dims[0] - min_dims[0]) / config['DATA']['CELL_SIZE'][0] + 10000,
                     np.floor(max_dims[1] - min_dims[1]) / config['DATA']['CELL_SIZE'][1] + 10000), dtype=np.int)
        map_array = np.zeros((big_shape[0], big_shape[1], len(config['INFO']['MAP_LAYER'])), dtype=np.uint8)


        ego = ((big_shape[0]/2).astype('i4'), (big_shape[1]/2).astype('i4'))
        scan_map = datafuns.raycasted_lidar_map(scan, scans[frame_id+every_th], np.zeros(map_array[..., :3].shape), ego, transes, frame_id, frame_id+every_th, config['DATA']['CELL_SIZE'])

        #return scan_map
        gt_map = np.zeros((map_array[..., 0].shape), dtype=np.uint8)
        ps_map = np.zeros((map_array[..., 0].shape), dtype=np.uint8)


        for v in vh_s:
            pts = datafuns.bbox_pts(v['x'], v['bb'])
            pts = (np.round(pts / config['DATA']['CELL_SIZE']) + ego).astype('i4')


            gt_map = cv2.fillPoly(gt_map, pts = np.array([pts[..., [1,0]]]), color=config['DATA']['GT'][v['cls']]) # annotation fill

        # road boundaries
        for num, rb in enumerate([rb_1, rb_2]):
            nrb = (np.linalg.inv(trans) @ rb.T).T
            nrb /= nrb[:, 3, None]
            rb_coors = np.round((nrb[:,:2]) / config['DATA']['CELL_SIZE']).astype('i4')
            rb_coors += ego
            mask = (rb_coors[:, 0] >= 0) & (rb_coors[:, 0] < map_array.shape[0])
            rb_coors = rb_coors[mask]
            rb_line = datafuns.line_traj(rb_coors, hack=False).astype('i4')

            gt_map[rb_line[:,0], rb_line[:,1]] =  config['INFO']['ANNO']['Road boundary'] # One line of rb

            if num == 0:
                for point in rb_line:
                    gt_map[point[0], point[1]:] = config['INFO']['ANNO']['Road boundary']

            if num == 1:
                for point in rb_line:
                    gt_map[point[0], :point[1]] = config['INFO']['ANNO']['Road boundary']

        # primary system
        for p in psvh_s:
            pts = datafuns.bbox_pts(p['x'], p['bb'])
            pts = (np.round(pts / config['DATA']['CELL_SIZE']) + ego).astype('i4')

            ps_map = cv2.fillPoly(ps_map, pts=np.array([pts[..., [1, 0]]]), color=config['DATA']['PS'][p['classification']])  # annotation fill

        if (ps_map == 0).all():
            print(frame_id, 'skipped!')
            continue


        x_min = (ego[0] + config['DATA']['AREA_SIZE'][0][0] / config['DATA']['CELL_SIZE'][0]).astype('i4')
        x_max = (ego[0] + config['DATA']['AREA_SIZE'][0][1] / config['DATA']['CELL_SIZE'][0]).astype('i4')
        y_min = (ego[1] + config['DATA']['AREA_SIZE'][1][0] / config['DATA']['CELL_SIZE'][1]).astype('i4')
        y_max = (ego[1] + config['DATA']['AREA_SIZE'][1][1] / config['DATA']['CELL_SIZE'][1]).astype('i4')


        new_ego = ego - np.array((x_min, y_min))

        scan_map = scan_map[x_min : x_max, y_min : y_max]
        gt_map = gt_map[x_min : x_max, y_min : y_max]
        ps_map = ps_map[x_min : x_max, y_min : y_max]

        scan_list.append(scan_map.swapaxes(0, 2))   # swap to CH, H, W
        gt_list.append(gt_map.swapaxes(0, 1))
        ps_list.append(ps_map.swapaxes(0, 1))

        print(frame_id)

    # Saving
    np.savez_compressed(destination, **{'scans' : np.stack(scan_list),
                         'gt' : np.stack(gt_list),
                         'ps' : np.stack(ps_list),
                         'transformation' : transes,
                         'frame' : frame_id_list,
                        'info' : {**{'ego' : new_ego}, **config['INFO'], **config['DATA']},
                        'data_file' : destination}
                        )


if __name__ == '__main__':
    con = 'valeo.yml'
    data_folder = 'data'
    dest_root = 'dataset/processed/'
    os.makedirs(dest_root, exist_ok=True)

    with open('configs/' + con, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Looping in data folder to processed multiple sequences
    dat_fi = glob.glob(f'{data_folder}/2*.npz')[0]
    data = np.load(dat_fi, allow_pickle=True)
    clca_fi = glob.glob(f'{data_folder}/CLCA*.npz')[0]
    clca = np.load(clca_fi, allow_pickle=True)
    psvh_fi = glob.glob(f'{data_folder}/psvh*.npy')[0]
    psvh = np.load(psvh_fi)

    create_data(data, clca, psvh, dest_root + 'example.npz', config, chosen_frames=[i for i in range(5)])
