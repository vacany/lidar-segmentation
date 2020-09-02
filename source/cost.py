import torch
import torch.nn as nn
import os
import numpy as np
from astar import trajectories
from tools.visuals import Bev_drawer

class Loss(torch.nn.Module):

    def __init__(self, task='cost', move_cost=0.01, num_traj=6, num_sim=4, device='cpu', weights=(1,1,1,1), extra=''):
        super().__init__()
        self.task = task
        self.device = device
        self.move_cost = torch.tensor(move_cost, dtype=torch.float).to(self.device)
        self.w = torch.tensor(weights).to(self.device)
        self.num_traj = num_traj
        self.num_sim = num_sim
        self.exp_loss = torch.tensor(0)
        self.gen_loss = torch.tensor(0)
        self.best_loss = torch.tensor(0)
        self.sim_loss = torch.tensor(0)
        self.obj_loss = torch.tensor(0)
        self.scan_loss = torch.tensor(0)
        self.extra = extra

    def forward(self, cm, pred_map, occ, exp_traj, best_traj, gen):
        # clip best, gen trajs out of band width
        #best_traj = [self.filter_band(best[i], batch['traj'][i]) for i in range(len(best))]
        #gen_traj = [self.filter_band(gen[i], batch['traj'][i]) for i in range(len(gen))]
        #self.L2_loss = self.L2(batch['pred_map'], cm)

        best = torch.tensor(best_traj, dtype=torch.long).to(self.device)
        exp = torch.tensor(exp_traj, dtype=torch.long).to(self.device)
        gen = torch.tensor(gen, dtype=torch.long).to(self.device)

        self.exp_loss = self.calculate_loss(cm, exp)

        self.best_loss = self.calculate_loss(cm, best)

        self.gen_loss = self.calculate_loss(cm, gen)


        self.sim_loss = self.similarity_loss(cm, exp, best) + self.similarity_loss(cm, exp, gen)

        #loss = self.w[0] * self.L2_loss + self.w[1] * self.exp_loss - self.w[2] * self.gen_loss - self.w[3] * self.sim_loss
        car_rb_mask = (pred_map >= 2).to(torch.float).to(self.device)  # 1 on car, rb
        scan_seg_mask = ((occ == 1) * (pred_map == 1)).to(torch.float).to(self.device)  # 0 on scans, does not change on no-vision
        scan_cm_mask = ((occ == 1) * (pred_map == 1)).to(torch.float).to(self.device)  # 0 on scans, does not change on no-vision
        self.obj_loss = self.calculate_loss(cm * car_rb_mask, best)
        self.scan_loss = self.calculate_loss(cm * scan_seg_mask, gen) + self.calculate_loss(cm * scan_seg_mask, best)

        self.scan_cm = self.calculate_loss(cm * scan_cm_mask, gen) # minimaze

        if 'sim' in self.extra:
            loss = self.exp_loss - self.scan_loss - self.gen_loss #- self.sim_loss

        else:
            loss = 3 * self.exp_loss - self.scan_loss - self.gen_loss - self.sim_loss





        return loss


    def L2(self, future, cm):

        L2 = []

        for b in range(len(future)):
            car_rb_mask = future[b] >= 2    # 1 on car, rb
            scan_mask = future[b] == 1     # 0 on scans, does not change on no-vision
            car_rb_pred = (car_rb_mask[car_rb_mask]).to(torch.float)

            l2_loss = ((cm[b][car_rb_mask] - car_rb_pred) ** 2).mean() + ((cm[b][scan_mask] ** 2).mean())
            L2.append(l2_loss)

        return torch.stack(L2, dim=0).mean().to(self.device)

    def calculate_loss(self, cm, traj):
        if type(traj) == torch.Tensor:
            traj = traj.to(torch.long).to(self.device)
        else:
            traj = torch.tensor(traj, dtype=torch.long).to(self.device)
        loss = trajectories.compute_cost(cm, traj, self.move_cost)

        return loss

    def similarity_loss(self, cm, exp, best):   # cuts output trajs

        best_one, exp_one = best, exp

        diff = best_one - exp_one
        cond = (torch.sign(diff[:,0]) != 0) & (abs(diff[:,0]) > 1)
        best_one = best_one[cond]
        sign = torch.sign(diff[cond])
        li = []

        for j in range(0, self.num_sim):
            li.append(best_one + j * sign)

        mask = torch.cat((best_one, *li), dim=0).to(self.device)

#        if len(mask) > 0:
        sim_loss = cm[mask[:,0], mask[:,1]].mean()

#        else:
#            sim_loss = torch.tensor(0.0).to(self.device)

        return sim_loss

class Motion_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'MotionModel'
        self.rb = nn.Conv2d(1, 1, 3, padding=1)
        #self.rb_pad = nn.ConstantPad2d((0,0,9,10), 0)
        self.obj_conv = nn.Conv2d(2, 1, 1) # object cost
        self.obj_conv.weight = torch.nn.Parameter(torch.ones(self.obj_conv.weight.shape), requires_grad=False)
        self.car = nn.Conv2d(1, 1, [1,30], padding=0)

        self.pad = nn.ConstantPad2d((14,15,0,0), 0)

    def forward(self, x):
        # with torch.no_grad():
        obj_cost = self.obj_conv(x[:,2:])

        rb_cost = self.rb(x[:,2:3])

        car_cost = self.car(x[:,3:4])
        car_cost = self.pad(car_cost)

        cost_map = rb_cost + car_cost + obj_cost
        cost_map = torch.sigmoid(cost_map)

        return cost_map

class Simple_cost(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'SimpleCost'
        self.rb1 = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.rbmid1 = nn.Conv2d(16, 16, 3, padding=1,bias=False)
        self.rbmid2 = nn.Conv2d(16, 16, 3, padding=1,bias=False)

        self.rb3 = nn.Conv2d(32, 16, 3, padding=1,bias=False)
        self.rbout = nn.Conv2d(16,4,1,bias=False)

        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.Upsample((160,320),mode='nearest')

        self.car1 = nn.Conv2d(1, 16, 7, padding=3,bias=False)
        self.carmid1 = nn.Conv2d(16, 16, 3, padding=1,bias=False)
        self.carmid2 = nn.Conv2d(16, 16, 3, padding=1,bias=False)

        self.car3 = nn.Conv2d(32, 16, 11, padding=5,bias=False)
        self.carout = nn.Conv2d(16, 4, 1,bias=False)

        self.both1 = nn.Conv2d(8, 4, 3, padding=1,bias=False)
        self.bothout = nn.Conv2d(4, 1, 1,bias=False)

        self.activ = torch.sigmoid

    def forward(self, x):
        # RB
        rb1 = self.activ(self.rb1(x[:, 2:3]))
        rb_pool = self.activ(self.rbmid1(self.pool(rb1)))
        rb_pool = self.activ(self.rbmid2(rb_pool))
        rb_pool = self.upsample(rb_pool)

        rb = self.activ(self.rb3(torch.cat((rb1, rb_pool), dim=1)))
        rb = self.activ(self.rbout(rb))

        # Car
        car1 = self.activ(self.car1(x[:, 3:4]))
        car_pool = self.activ(self.carmid1(self.pool(car1)))
        car_pool = self.activ(self.carmid2(car_pool))
        car_pool = self.upsample(car_pool)

        car = self.activ(self.car3(torch.cat((car1, car_pool), dim=1)))
        car = self.activ(self.carout(car))

        both = self.activ(self.both1(torch.cat((rb,car), dim=1)))
        both = self.activ(self.bothout(both))

        both = both + x[:,2:3] + x[:,3:4]
        both = torch.sigmoid(both)

        return both

    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.xavier_uniform_(lay.weight)


