import torch
import numpy as np
from tools import stats, datafuns


class Concat_prediction(torch.nn.Module):
    def __init__(self, seg_model=None, cost_model=None, task='upgrade'):
        super().__init__()
        self.seg = seg_model
        self.cost = cost_model
        self.task = task
        self.concat = Cat_outputs()

    def forward(self, x, odo):
        #if self.task == 'cost':
            #x = self.cost(x.squeeze)

        x = self.concat(odo, x.squeeze(1))

        return x

class Cat_outputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Concatenate along x-coordinate, last dim


    def forward(self, out, odo):

        cat_list = []
        st = out[0][:, : odo[1][1]].clone()  # till first odo
        diff = 0

        for i in range(len(odo) - 1):
            # move also y-coor
            y_coor = odo[i+1][0] - odo[i][0]
            diff += y_coor
            # iter odometry
            start = odo[0][1]
            end = start + odo[i+1][1] - odo[i][1]

            if i == len(out) - 2:
                # add rest of last cm
                st = torch.cat((st, out[i+1].roll(int(diff), dims=0)[:, start: out[i+1].shape[-1] - st.shape[-1] + start]), dim=1)

            else:
                # join next
                st = torch.cat((st, out[i+1].roll(int(diff), dims=0)[:, start : end]), dim=1)

        return st

class Noise_Predictions(torch.nn.Module):
    def __init__(self, prob=0.7):
        super().__init__()
        self.prob = prob
        # Concatenate along x-coordinate, last dim


    def forward(self, batch_pred):
        mask = torch.bernoulli(torch.ones(batch_pred.shape) * self.prob).to(batch_pred.device)
        noised_pred = batch_pred.clone()
        noised_pred[(mask == 1) & (batch_pred != 0)] = 1

        #diff = batch_pred - batch_noised

        return noised_pred #- diff

class ray_temp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, scan, mask):
        masked = ((scan[:,1] == 2) * (mask[:,0] == 1) * (mask[:,1] == 0))
        temp_loss = pred[:,2][masked]

        return temp_loss.mean()



class Temp_rb(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, odo):
        self.temp_loss = torch.tensor(0)
        comp_list = []
        rb = pred[:, -2]
        diff_y = 0
        diff_x = 0
        ### if there is scanpoint both
        for i in range(len(odo) -1):
            diff_x = odo[i+1][1] - odo[i][1]
            diff_y = odo[i+1][0] - odo[i][0]

            a = rb[i]  # roll forward
            b = rb[i+1].roll((+diff_y, +diff_x), dims=(0,1))

            comp = (a[:,:-diff_x] - b[:,:-diff_x]) ** 2
            comp_list.append(comp.mean())

        self.temp_loss = torch.stack(comp_list).mean()

        return self.temp_loss, a,b

class Consistency_cls(torch.nn.Module):
    def __init__(self, model, trn_dataset, batch_size=4, max_iter=0, target='gt'):
        super().__init__()
        self.dataset = trn_dataset
        self.bs = batch_size
        self.max_iter = max_iter
        self.model = model
        self.con_loss = 0
        self.weights = torch.tensor(datafuns.class_weights(trn_dataset.label_data)[1], dtype=torch.float).to(self.dataset.device)
        self.weights[[0,1]] = 0 # dont care about no vision and scans, try also cars
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weights)
        self.metric = stats.Valeo_IoU(4)
        self.metric_ps = stats.Valeo_IoU(4)
        self.target = target

    def forward(self):
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.bs, shuffle=True)

        for it, batch in enumerate(loader):
            self.comp_loss = torch.tensor(0)

            pred = self.model(batch['in'])

            if self.target == 'ps':
                batch['gt'] = batch['in'][:,-1].to(torch.long)

            self.comp_loss = self.criterion(pred, batch['gt'].squeeze(1))

            if it == self.max_iter:
                break

        return self.comp_loss

class Softargmax(torch.nn.Module):

    def __init__(self, num_cls):
        super().__init__()
        self.num_cls = num_cls

    def forward(self, logits, dim=1):
        y_soft = logits.softmax(dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft

        if self.num_cls == 2:
            gum = ret[:,0] * 0 + ret[:,1] * 1

        if self.num_cls == 3:
            gum = ret[:,0] * 0 + ret[:,1] * 1 + ret[:,2] * 2

        if self.num_cls == 4:
            gum = ret[:,0] * 0 + ret[:,1] * 1 + ret[:,2] * 2 + ret[:, 3] * 3

        return ret#gum.unsqueeze(1)

class Rotate_grid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bev, trans):
        w, h = bev.shape[-2:]

        theta = torch.stack([torch.tensor(self.param2theta(trans[i], w, h)) for i in range(len(trans))])

        print(bev.shape)
        print(theta)

        grid = torch.nn.functional.affine_grid(theta, bev.size(), align_corners=True)
        rotated_grid = torch.nn.functional.grid_sample(bev, grid.to(torch.float), mode='nearest', align_corners=True)

        return rotated_grid


    def param2theta(self, param, w, h):
        param = param.detach().cpu().numpy()

        param = np.linalg.inv(param)
        theta = np.zeros([2, 3])
        theta[0, 0] = param[0, 0]
        theta[0, 1] = param[0, 1] * h / w
        theta[0, 2] = param[0, 2] * 2 / w + param[0, 0] + param[0, 1] - 1
        theta[1, 0] = param[1, 0] * w / h
        theta[1, 1] = param[1, 1]
        theta[1, 2] = param[1, 2] * 2 / h + param[1, 0] + param[1, 1] - 1
        return theta

# class Metric(torch.nn.Module):
#     def __init__(self, num_cls):
#         super().__init__()
#         self.num_cls = num_cls
#
#     def forward(self, pred, gt):
#
#
#
#     def keep_best(self):
#         return 0
