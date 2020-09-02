import torch
import numpy as np
import os
import glob
import argparse
from learning import Learning
from dataloaders import NIPS_Dataset, Seg_Dataset
from tools.visuals import Bev_drawer
from tools import stats, datafuns
from astar import trajectories
import modules
from make import make
import sys
from tools import utils

parser = argparse.ArgumentParser(description="set hyperparameters for learning")
parser.add_argument('--extra', "-extra", default='', type=str, help='name')
parser.add_argument('--task', "-t", default='upgrade', type=str, help='set main procedure')
parser.add_argument('--source', '-s', default='valeo', type=str, help='choose source data for training')
parser.add_argument('--mode', '-m', default='trn', type=str, help='describe procedure and save to mode dir')
parser.add_argument('--ground_truth', '-gt', default='gt', type=str, help='choose ground truth for learning')
parser.add_argument('--cost_input', '-c_in', default='gt', type=str, help='choose input type for learning cost map')
parser.add_argument('--validate', '-val', default=False, type=bool, help='choose input type for learning cost map')
# params
parser.add_argument('--epochs', '-e', default=80, type=int, help='number of epochs')
parser.add_argument('--batch_size', '-bs', default=1, type=int, help='batch_size value')
parser.add_argument('--learning_rate', "-lr", default=0.001, type=float, help='set learning rate')
parser.add_argument('--weights', '-w', nargs="+", default=(1, 1, 1, 1), type=float, help='choose weights for training loss')
parser.add_argument('--loss_mode', '-lm', default='cost', type=str, help='describe loss tree logic')
parser.add_argument('--consecutive_frames', '-cf', default=80, type=int, help='specify number of frames to the future')
parser.add_argument('--every_th', '-et', default=5, type=int, help='specify number of grabbed frames')
# models
parser.add_argument('--model_seg', '-ms', default='Lil_UNet', type=str, help='Choose model for segmentation')
parser.add_argument('--model_cost', '-mc', default='Lil_UNet', type=str, help='Choose model for cost map generation')
parser.add_argument('--load_seg', '-ls', default='002_02.pt', type=str, help='adress to model state dict')
parser.add_argument('--load_cost', '-lc', default=False, type=str, help='adress to model state dict')
parser.add_argument('--bernoulli_prob', '-bp', default=1, type=float, help='Bernoulli noise prob')
parser.add_argument('--target', '-tar', default='gt', type=str, help='learn by PS or GT')
# loading
parser.add_argument('--gpu', default=0, type=int, help='choose GPU on nvidia schedule')
parser.add_argument('--trn_file', '-tf', default='sample', type=str, help='choose trn data for training')
parser.add_argument('--val_file', '-vf', default='sample', type=str, help='choose val data for training')
parser.add_argument('--output_log', '-ol', action='store_false', help='Logging outside shell y/n')

# Learning init
args = parser.parse_args()
if torch.cuda.is_available():
    args.gpu = int(utils.get_free_gpu()[0])
learner = Learning(args)

if 'concat' not in args.extra:
    learner.cm_model.load_state_dict(torch.load(f'{os.path.expanduser("~")}/datasets/nets/cmaps/000_140.pt', map_location='cpu')['cost'])
folder = f'{os.path.expanduser("~")}/datasets/{args.source}/experiments/{args.mode}_{args.target}_{learner.cm_model.name}' + f'_{args.extra}'
make(folder)
# Init layers
drawer = Bev_drawer(img_resize=(1,1))


cat_layer = modules.Cat_outputs()
argmax = modules.Softargmax(4)
noise_layer = modules.Noise_Predictions(prob=args.bernoulli_prob)
move = torch.tensor(0.0, dtype=torch.float).to(learner.device)
t_loss = modules.Temp_rb()

learner.cm_model.train()
learner.seg_model.train()
if args.validate:
    learner.seg_model.train()
    learner.cm_model.train()

trn_files = glob.glob(f'{os.path.expanduser("~")}/datasets/{learner.source}/data/trn/*.npz') # finetunning on trajs only
seg_trn_files = glob.glob(f'{os.path.expanduser("~")}/datasets/{learner.source}/data/trn/*.npz')[0] # use for comparsion loss
val_files = glob.glob(f'{os.path.expanduser("~")}/datasets/{learner.source}/data/val/*.npz') # validate models w/ & w/o upgrade
tst_files = glob.glob(f'{os.path.expanduser("~")}/datasets/{learner.source}/data/val/*.npz') # test models w/ & w/o upgrade (to paper)

if args.output_log:
    sys.stdout = open(f'{folder}/out.log', 'w')
    sys.stderr = sys.stdout

metric_seg = stats.Valeo_IoU(4)
metric_ps = stats.Valeo_IoU(4)

#TODO: learn cost map on GT?
# seg model purely on GT? keep PS as experiment for paper
# planning in 3D cancels car label dynamics
# learn cm on GT - from whole labels, not just scans



for e in range(args.epochs):

    #for i, (trn_file, gt_trn, val_file, tst_file) in enumerate(zip(trn_files, seg_trn_files, val_files, tst_files)):
    for i, trn_file in enumerate(trn_files):
        if args.mode == 'trn':
            trn_dataset = NIPS_Dataset(trn_file, NUM=learner.cf, task=args.task, every_th=learner.every_th, device=learner.device)
            data_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=1, shuffle=True)

        # if args.mode == 'val':
        #     val_dataset = NIPS_Dataset(val_file, NUM=learner.cf, task=args.task, every_th=learner.every_th, device=learner.device)
        #     data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
        #
        # if args.mode == 'tst':
        #     tst_dataset = NIPS_Dataset(tst_file, NUM=learner.cf, task=args.task, every_th=learner.every_th, device=learner.device)
        #     data_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=1, shuffle=False)


        gt_dataset = NIPS_Dataset(seg_trn_files, NUM=learner.cf, task='consistency', every_th=learner.every_th, device=learner.device)
        gt_loader = torch.utils.data.DataLoader(gt_dataset, batch_size=1, shuffle=True)

        new_weights = torch.tensor(datafuns.class_weights(trn_dataset.label_data)[1], dtype=torch.float).to(gt_dataset.device)

        if len(new_weights) == 4:
            weights = new_weights


        Cross_ent = torch.nn.CrossEntropyLoss(weight=weights)

        temp_layer = modules.ray_temp().to(learner.device)
        # GT learning cost map

        for it, batch in enumerate(data_loader):
            comp_cm_loss = torch.tensor(0)
            comp_seg_loss = torch.tensor(0)
            temp_loss = torch.tensor(0)
            if 'comp' not in args.extra:
                for inner, inner_batch in enumerate(gt_loader):
                #     ### Segment

                    occ_mask = inner_batch['in'][0][:,-2:-1].repeat(1,4,1,1)

                    pred = learner.seg_model(inner_batch['in'][0])
                    tr, odo = gt_loader.dataset.get_trajectories(inner_batch['frame'])
                    t_lay = modules.Temp_rb()  # check only alone in segmentation

                    pred = argmax(pred)
                    pred = pred * occ_mask
                    # Loss
                    comp_seg_loss = Cross_ent(pred, inner_batch['gt'][0])
                    temp_loss = temp_layer(pred, inner_batch['in'][0], batch['mask'][0].to(learner.device))

                    seg_loss = comp_seg_loss + temp_loss
                    seg_loss.backward()

                    learner.optimizer_seg.step()
                    learner.optimizer_seg.zero_grad()



                    ### Cost

                    if 'cm' not in args.extra:
                        inner_tr, inner_odo = gt_loader.dataset.get_trajectories(inner_batch['frame'])
                        pred = batch['orig_gt'].squeeze(0)
                        cm = learner.cm_model(pred)
                        cm = torch.sigmoid(cm)
                        one_cm = cat_layer(cm.squeeze(1), inner_odo)
                        one_pred = torch.argmax(pred, dim=1)
                        pr = cat_layer(one_pred, inner_odo)
                        occ = cat_layer(batch['in'][0, :, -2], inner_odo)

                        best = trajectories.best_trajectory(one_cm, inner_tr, move)

                        gen = [trajectories.best_trajectory(one_cm + torch.rand(one_cm.shape).to(learner.device) * 0.2, inner_tr, move) for i in range(1)]
                        gen = np.concatenate([*gen], axis=0)

                        comp_cm_loss = learner.criterion(one_cm, pr, occ, inner_tr, best, gen)

                        comp_cm_loss.backward()
                        learner.optimizer_cost.step()
                        learner.optimizer_cost.zero_grad()

                    del inner_batch
                    if 'only_inner' not in args.extra:
                        break
                    print(seg_loss, comp_cm_loss)


            if 'only_inner' in args.extra:
                continue

            ### Consistency
            tr, odo = data_loader.dataset.get_trajectories(batch['frame'])
            occ_mask = batch['in'][0][:, -2:-1].repeat(1, 4, 1, 1)
            pred = learner.seg_model(batch['in'][0])

            pred = argmax(pred)
            pred = pred * occ_mask

            cm = learner.cm_model(pred)
            cm = torch.sigmoid(cm)


            one_cm = cat_layer(cm.squeeze(1), odo)

            one_pred = torch.argmax(pred, dim=1)
            pr = cat_layer(one_pred, odo)
            occ = cat_layer(batch['in'][0,:,-2], odo)


            best = trajectories.best_trajectory(one_cm, tr, move)

            gen = [trajectories.best_trajectory(one_cm + torch.rand(one_cm.shape).to(learner.device) * 0.4, tr, move) for i in range(1)]
            gen = np.concatenate([*gen], axis=0)

            loss = learner.criterion(one_cm, pr, occ, tr, best, gen)

            # weight_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(learner.device)
            # for W in learner.cm_model.parameters():
            #     weight_loss = weight_loss + W.norm(2)
            #
            # loss = loss + weight_loss
            #
            # print(weight_loss)
            if args.mode in ['trn', 'toy']:
                loss.backward()
                learner.optimizer_seg.step()
                learner.optimizer_seg.zero_grad()
                if 'cm' not in args.extra:
                    learner.optimizer_cost.step()
                    learner.optimizer_cost.zero_grad()

            det_rb, det_car = metric_seg.build(torch.argmax(pred, dim=1), batch['gt'].squeeze(0), batch['in'][0, :, -2])
            ps_rb, ps_car = metric_ps.build(batch['ps'].squeeze(0), batch['gt'].squeeze(0), batch['in'][0, :, -2])

            print(f'Seg: {det_rb:.3f}, {det_car:.3f}, \t Ps: {ps_rb:.3f}, {ps_car:.3f}')

            loss_dict = {'exp': np.round(learner.criterion.exp_loss.item(), decimals=4),
                         'best': np.round(learner.criterion.best_loss.item(), decimals=4),
                         'gen': np.round(learner.criterion.gen_loss.item(), decimals=4),
                         'sim': np.round(learner.criterion.sim_loss.item(), decimals=4),
                         'comp_seg': np.round(comp_seg_loss.item(), decimals=4), 'comp_cost' : np.round(comp_cm_loss.item(), decimals=4),
                         'temp': np.round(temp_loss.item(), decimals=4), 'scan': np.round(learner.criterion.scan_loss.item(), decimals=4),
                         'obj': np.round(learner.criterion.obj_loss.item(), decimals=4),
                         'all': np.round(loss.item(), decimals=4)}

            print(loss_dict)

            if it % 20 == 0:
                if args.mode in ['trn', 'toy', 'val', 'tst']:
                    drawer.update(one_cm, expert=tr, best=best, generated=gen)
                    drawer.add_label(pr.detach().cpu().numpy())
                    drawer.add_label(torch.argmax(pred, dim=1)[0].detach().cpu().numpy())
                    drawer.add_label(batch['gt'][0][0].detach().cpu().numpy())
                    drawer.out(f'{folder}/img/{e:03d}_{it:03d}_{batch["frame"][0]}_{i}.png')


                if args.mode in ['trn', 'toy', 'val']:

                    data = {'epoch' : e,
                            'iter' : it,
                            'seg' : learner.seg_model.state_dict(),
                            'cost' : learner.cm_model.state_dict(),
                            'loss' : loss_dict,
                            'metric_seg' : metric_seg,
                            'metric_ps' : metric_ps,
                            'args' : args,
                            'exp_traj' : tr,
                            'best_traj' : best,
                            'cmap' : one_cm,
                            'frame' : batch['frame'],
                            'data_file' : data_loader.dataset.data_file}

                    torch.save(data, f'{folder}/data/{e:03d}_{it:03d}_{i}.pt')

        print(f'epoch {e:03d} Ended')

        metric_seg.calculate()
        metric_ps.calculate()

        stats.results(metric_seg, metric_ps)
        metric_seg.reset()
        metric_ps.reset()

    if args.mode in ['val','tst']:
        break



