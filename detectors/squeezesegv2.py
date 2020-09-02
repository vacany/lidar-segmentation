import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeSegV2(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        squeeze_kwargs = {'input_channels' : input_channel,
        'squeeze_depth' : 3,
        'cam_depth' : 1,
        'conv_starts' : 64,
        'squeeze_start' : 16,
        'ef_start' : 64}

        head_kwargs = {'in_channels' : 64,
        'mid_channels' : 32,
        'num_classes' : output_channel,
        'crf_iters' : 3,
        'crf_dims' : 3,
        'crf_start_dim' : 0}

        self.squeeze = SqueezeSegBone(**squeeze_kwargs)
        self.head = SegmentHead(**head_kwargs)

    def forward(self, x):
        features = self.squeeze(x)
        return self.head(x, features)

    @classmethod
    def load_from_kwargs(cls, data):
        if isinstance(data['head_cls'], str):
            head_cls = SegmentHead()
            data['head_cls'] = head_cls
        return cls(**data)

class SqueezeSegBone(nn.Module):
    def __init__(self, input_channels, squeeze_depth=2, cam_depth=1, conv_starts=64, squeeze_start=16, ef_start=64):
        super().__init__()
        self.reduce = 1
        self.start = nn.Sequential(
            Conv(input_channels, conv_starts, 3, 1, 2, top_parent=self),
            ContextAggregation(conv_starts, top_parent=self),
            Conv(conv_starts, conv_starts, 1, top_parent=self),
        )
        self.rest = nn.Sequential(
            Pool(3, 2, 1, top_parent=self),
            SqueezePart(conv_starts, squeeze_start, ef_start, squeeze_depth, cam_depth, top_parent=self),
            DeFire(2 * ef_start, squeeze_start, int(conv_starts / 2), top_parent=self),
            nn.Dropout2d(),
        )

    def forward(self, x):
        shape = x.shape
        over = shape[-1] % self.reduce
        if over:
            over = self.reduce - over
            x = F.pad(x, (int(over / 2), int(over / 2), 0, 0), 'replicate')
        pre_add = self.start(x)
        insides = self.rest(pre_add)
        result = pre_add + insides
        return result

class SegmentHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes, crf_iters, crf_start_dim, crf_dims, **crf_kwargs):
        super().__init__()
        self.net = nn.Sequential(
            DeFire(in_channels, mid_channels // 16, mid_channels // 2),
            Fire(mid_channels, mid_channels // 16, mid_channels // 2),
            Conv(mid_channels, num_classes, 1, relu=False, norm=False),
        )
        self.crf = CRF(crf_iters, crf_start_dim, crf_dims, **crf_kwargs)

    def forward(self, data_input, features):
        result = self.net(features)
        if result.shape[-1] != data_input.shape[-1]:
            diff = result.shape[-1] - data_input.shape[-1]
            result = result[..., (diff // 2) : -(diff // 2)]
        result = self.crf(data_input, result)
        return result

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze, expand, cam=False, top_parent=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = expand * 2
        self.squeeze = Conv(in_channels, squeeze, 1, top_parent=top_parent)
        self.expand1x1 = Conv(squeeze, expand, 1, top_parent=top_parent)
        self.expand3x3 = Conv(squeeze, expand, 3, 1, top_parent=top_parent)
        if cam:
            self.cam = ContextAggregation(self.out_channels, top_parent=top_parent)
        else:
            self.cam = None

    def forward(self, x):
        sq = self.squeeze(x)
        e1 = self.expand1x1(sq)
        e3 = self.expand3x3(sq)
        c = torch.cat([e1, e3], 1)
        if self.cam is not None:
            return self.cam(c)
        return c


class DeFire(nn.Module):
    def __init__(self, in_channels, squeeze, expand, cam=False, top_parent=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = expand * 2
        self.squeeze = Conv(in_channels, squeeze, 1, top_parent=top_parent)
        self.deconv = DeConv(squeeze)
        self.expand1x1 = Conv(squeeze, expand, 1, top_parent=top_parent)
        self.expand3x3 = Conv(squeeze, expand, 3, 1, top_parent=top_parent)
        if cam:
            self.cam = ContextAggregation(self.out_channels, top_parent=top_parent)
        else:
            self.cam = None

    def forward(self, x):
        sqd = self.deconv(self.squeeze(x))
        e1 = self.expand1x1(sqd)
        e3 = self.expand3x3(sqd)
        c = torch.cat([e1, e3], 1)
        if self.cam is not None:
            return self.cam(c)
        return c


class Pool(nn.Module):
    def __init__(self, size, stride, pad=0, top_parent=None):
        super().__init__()
        if top_parent is not None:
            top_parent.reduce *= stride
        self.pool = nn.MaxPool2d(size, (1, stride), padding=pad)

    def forward(self, x):
        return self.pool(x)


class ContextAggregation(nn.Module):
    def __init__(self, channels, reduction=16, top_parent=None):
        super().__init__()
        mid = channels // reduction
        self.in_channels = channels
        self.out_channels = channels
        nets = [
            Pool(7, 1, 3, top_parent=top_parent),
            Conv(channels, mid, 1, relu=True, norm=False, top_parent=top_parent),
            Conv(mid, channels, 1, relu=False, norm=False, top_parent=top_parent),
            torch.nn.Sigmoid(),
        ]
        self.nets = nn.Sequential(*nets)

    def forward(self, x):
        return x * self.nets(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad=0, stride=1, relu=True, norm=True, top_parent=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if top_parent is not None:
            top_parent.reduce *= stride
        nets = []
        nets.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, stride=(1, stride)))
        if relu:
            nets.append(nn.ReLU(inplace=True))
        if norm:
            nets.append(nn.BatchNorm2d(out_channels))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class DeConv(nn.Module):
    def __init__(self, channels, relu=True, norm=True):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        nets = []
        nets.append(nn.ConvTranspose2d(channels, channels, (1, 4), (1, 2), padding=(0, 1)))
        if relu:
            nets.append(nn.ReLU(inplace=True))
        if norm:
            nets.append(nn.BatchNorm2d(channels))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class SqueezePart(nn.Module):
    SQ_ADD = 16
    EF_ADD = 64

    def __init__(self, input_channels, sq, ef, depth, cam_depth=0, top_parent=None):
        super().__init__()
        cam = cam_depth > 0
        if depth == 0:
            self.net = nn.Sequential(
                Fire(input_channels, sq, ef, cam, top_parent=top_parent),
                Fire(2 * ef, sq, ef, cam, top_parent=top_parent),
                Fire(2 * ef, sq + self.SQ_ADD, ef + self.EF_ADD, cam, top_parent=top_parent),
                Fire(2 * (ef + self.EF_ADD), sq + self.SQ_ADD, ef + self.EF_ADD, cam, top_parent=top_parent),
            )
        else:
            self.beg = nn.Sequential(
                Fire(input_channels, sq, ef, cam, top_parent=top_parent), Fire(2 * ef, sq, ef, cam, top_parent=top_parent)
            )
            self.rest = nn.Sequential(
                Pool(3, 2, 1, top_parent=top_parent),
                SqueezePart(2 * ef, sq + self.SQ_ADD, ef + self.EF_ADD, depth - 1, cam_depth - 1, top_parent=top_parent),
                DeFire(2 * (ef + self.EF_ADD * (2 if depth == 1 else 1)), 2 * sq, ef, top_parent=top_parent),
            )
        self.depth = depth

    def forward(self, x):
        if self.depth:
            pre_add = self.beg(x)
            insides = self.rest(pre_add)
            return pre_add + insides
        else:
            return self.net(x)


class CRF(nn.Module):
    SQ_VAR_BI = np.array([0.015, 0.015, 0.01]) ** 2
    SQ_VAR_ANG = np.array([0.9, 0.9, 0.6]) ** 2

    def __init__(
        self,
        num_iterations,
        bf_start_dim,
        bf_dims,
        mask_dim=-1,
        size_a=3,
        size_b=5,
        sq_var_bi=None,
        sq_var_ang=None,
        sq_var_bi_ang=None,
        ang_coef=0.02,
        bi_coef=0.1,
    ):
        super().__init__()
        if sq_var_ang is None:
            sq_var_ang = self.SQ_VAR_ANG
        if sq_var_bi is None:
            sq_var_bi = self.SQ_VAR_BI
        num_classes = len(sq_var_ang)
        init = (np.ones((num_classes, num_classes)) - np.eye(num_classes))[..., None, None].astype(np.float32)
        self.mask_dim = mask_dim
        self.bilateral = _BilateralWeights(size_a, size_b, bf_dims, sq_var_bi)
        self.local = _LocalPassing(size_a, size_b, num_classes, sq_var_ang, sq_var_bi_ang)
        self.ang_compat = nn.Conv2d(num_classes, num_classes, 1, bias=False)
        self.bi_ang_compat = nn.Conv2d(num_classes, num_classes, 1, bias=False)
        self.iterations = num_iterations
        self.bf_start_dim = bf_start_dim
        self.bf_dims = bf_dims
        self.ang_compat.weight = nn.Parameter(torch.from_numpy(init * ang_coef))
        self.bi_ang_compat.weight = nn.Parameter(torch.from_numpy(init * bi_coef))

    def forward(self, lidar_input, data):
        bf_weights = self.bilateral(lidar_input[:, self.bf_start_dim : self.bf_start_dim + self.bf_dims])
        mask = (lidar_input[:, self.mask_dim, None, ...] >= 0.5).float()
        for _ in range(self.iterations):
            unary = F.softmax(data, 1)
            ang, bi_ang = self.local(unary, mask, bf_weights)
            ang = self.ang_compat(ang)
            bi_ang = self.bi_ang_compat(bi_ang)
            outputs = unary + ang + bi_ang
            data = outputs
        return outputs


class DropoutNoise(nn.Module):
    def __init__(self, np_file=osp.join(osp.dirname(osp.abspath(__file__)), 'mask.npy')):
        super().__init__()
        self.mask = torch.from_numpy(np.load(np_file)).clamp(0, 1)[None, ...]

    def forward(self, data):
        bsize = data.shape[0]
        for i in range(bsize):
            mask = torch.bernoulli(self.mask).float()
            data[i] *= mask
        return data



class _LocalPassing(nn.Module):
    def __init__(self, size_a, size_b, in_channels, sq_var_ang, sq_var_bi=None):
        if sq_var_bi is None:
            sq_var_bi = sq_var_ang
        pad = (size_a // 2, size_b // 2)
        super().__init__()
        self.ang_conv = nn.Conv2d(in_channels, in_channels, (size_a, size_b), padding=pad, bias=False)
        self.bi_ang_conv = nn.Conv2d(in_channels, in_channels, (size_a, size_b), padding=pad, bias=False)
        self.condense_conv = nn.Conv2d(in_channels, (size_a * size_b - 1) * in_channels, (size_a, size_b), padding=pad, bias=False)

        self.ang_conv.weight = nn.Parameter(torch.from_numpy(_gauss_weights(size_a, size_b, in_channels, sq_var_ang)), requires_grad=False)
        self.bi_ang_conv.weight = nn.Parameter(
            torch.from_numpy(_gauss_weights(size_a, size_b, in_channels, sq_var_bi)), requires_grad=False
        )
        self.condense_conv.weight = nn.Parameter(torch.from_numpy(_condensing_weights(size_a, size_b, in_channels)), requires_grad=False)

    def forward(self, data, mask, bilateral):
        b, c, h, w = data.shape
        ang = self.ang_conv(data)
        bi_ang = self.bi_ang_conv(data)
        condense = self.condense_conv(data * mask).view(b, c, -1, h, w)
        bi_out = (condense * bilateral).sum(2) * mask * bi_ang
        return ang, bi_out


class _BilateralWeights(nn.Module):
    def __init__(self, size_a, size_b, in_channels, sq_var):
        super().__init__()
        pad = (size_a // 2, size_b // 2)
        self.in_channels = in_channels
        self.sq_var = sq_var
        self.condense_conv = nn.Conv2d(in_channels, (size_a * size_b - 1) * in_channels, (size_a, size_b), padding=pad, bias=False)
        self.condense_conv.weight = nn.Parameter(torch.from_numpy(_condensing_weights(size_a, size_b, in_channels)), requires_grad=False)

    def forward(self, data):
        condensed = self.condense_conv(data)
        diffs = [data[:, i, None, ...] - condensed[:, i :: self.in_channels, ...] for i in range(self.in_channels)]
        return torch.stack([torch.exp_(-sum([diff ** 2 for diff in diffs]) / (2 * self.sq_var[i])) for i in range(len(self.sq_var))], 1)


def _gauss_weights(size_a, size_b, num_classes, sq_var):
    kernel = np.zeros((num_classes, num_classes, size_a, size_b), dtype=np.float32)
    for k in range(num_classes):
        kernel_2d = np.zeros((size_a, size_b), dtype=np.float32)
        for i in range(size_a):
            for j in range(size_b):
                diff = np.sum((np.array([i - size_a // 2, j - size_b // 2])) ** 2)
                kernel_2d[i, j] = np.exp(-diff / 2 / sq_var[k])
        kernel_2d[size_a // 2, size_b // 2] = 0
        kernel[k, k] = kernel_2d
    return kernel


def _condensing_weights(size_a, size_b, in_channels):
    half_filter_dim = (size_a * size_b) // 2
    kernel = np.zeros((size_a * size_b * in_channels, in_channels, size_a, size_b), dtype=np.float32)
    for i in range(size_a):
        for j in range(size_b):
            for k in range(in_channels):
                kernel[i * (size_b * in_channels) + j * in_channels + k, k, i, j] = 1
    kernel = np.concatenate([kernel[: in_channels * half_filter_dim], kernel[in_channels * (half_filter_dim + 1) :]], axis=0)
    return kernel

