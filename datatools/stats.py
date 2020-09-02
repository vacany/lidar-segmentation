
import numpy as np
import torch
import scipy.sparse
import cv2

# option for calculating withou GT FN, one that would be fair to segmentor

class IOU_Metric():
    def __init__(self, num_classes, dtype = 'torch'):
        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes), 'u8')  # confusion matrix
        self.dtype = dtype
        self.IOU_trn = np.zeros(self.num_classes)
        self.best = np.zeros(self.num_classes)
        np.set_printoptions(precision=4)

    def build(self, prediction, label, filter_class=None):
        if self.dtype == 'torch':
            if prediction.shape[1] == 1:
                predictions = torch.flatten(prediction).detach().cpu().numpy()
            else:
                predictions = torch.flatten(torch.argmax(prediction, 1)).detach().cpu().numpy()
            labels = torch.flatten(label).detach().cpu().numpy()
        if self.dtype == 'numpy':
            predictions = prediction.flatten()
            labels = label.flatten()

        if filter_class is not None:
            predictions[labels==filter_class] = filter_class

        self.tmp_cm = scipy.sparse.coo_matrix((np.ones(np.prod(labels.shape), 'u8'), (labels, predictions)), shape=(self.num_classes, self.num_classes)
        ).toarray()

        self.cm += self.tmp_cm

    def calculate(self, validation=False, overall=False):
        IOU = np.zeros(len(self.cm))
        with np.errstate(all='ignore'):
            if overall:
                for x in range(len(IOU)):
                    IOU[x] = self.cm[x, x] / (sum(self.cm[:, x]) + sum(self.cm[x, :]) - self.cm[x, x]) # IOU = TP / (FP + FN + TP)
                self.IOU = IOU

            else:
                for x in range(len(IOU)):
                    IOU[x] = self.tmp_cm[x, x] / (sum(self.tmp_cm[:, x]) + sum(self.tmp_cm[x, :]) - self.tmp_cm[x, x])
                self.IOU = IOU

        if validation:
            if sum(self.IOU[1:]) >= sum(self.best[1:]):
                self.best_val = self.IOU_val
        else:
            if sum(self.IOU[1:]) >= sum(self.best[1:]):
                self.best_trn = self.IOU_trn

        return IOU

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes))
        self.tmp_cm = np.zeros((self.num_classes, self.num_classes))
        self.IOU = np.zeros(self.num_classes)


class Valeo_IoU():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.rb_TP = 0
        self.rb_FP = 0
        self.rb_FN = 0
        self.rb_unTP = 0
        self.rb_unFP = 0
        self.rb_unFN = 0

        self.car_TP = 0
        self.car_FP = 0
        self.car_FN = 0
        self.car_unTP = 0
        self.car_unFP = 0
        self.car_unFN = 0

        self.IOU_rb = 0
        self.IOU_car = 0
        np.set_printoptions(precision=2)

    def build(self, pred, gt, occ):
        if pred.shape[1] == 4:
            pred = torch.argmax(pred, dim=1).detach().cpu().numpy()
        else:
            pred = pred.detach().cpu().numpy()

        if gt.shape[1] == 4:
            gt = torch.argmax(gt, dim=1).detach().cpu().numpy()
        else:
            gt = gt.detach().cpu().numpy()

        occ = occ.detach().cpu().numpy()

        max_rb = 4


        # rb_only = (gt == 2).astype('u1')
        # rb_only = np.stack([cv2.distanceTransform(rb_only[i], cv2.DIST_L1, 3) for i in range(len(rb_only))])
        # rb_only = ((rb_only > 0) & (rb_only <= max_rb)).astype('u1')
        # if not enough ---> dilate


        ### RB
        # True positive
        rb_TP = (gt == 2) * (pred == 2) * (occ == 1)
        rb_TP_num = rb_TP.sum()
        self.rb_TP += rb_TP_num
        # False positive - complete
        rb_FP = (gt != 2) * (pred == 2) * (occ == 1)
        rb_FP_num = rb_FP.sum()
        self.rb_FP += rb_FP_num
        # False Negative
        rb_FN = (gt == 2) * (occ == 1) * (pred != 2)
        rb_FN_num = rb_FN.sum()
        self.rb_FN += rb_FN_num

        ### Unseen RB
        # Unseen True positive
        rb_unTP= (gt == 2) * (occ != 1) * (pred == 2)
        rb_unTP_num = rb_unTP.sum()
        self.rb_unTP += rb_unTP_num
        # Unseen False positive - doest not matter
        rb_unFP = (gt != 2) * (pred == 2) * (occ != 1)
        rb_unFP_num = rb_unFP.sum()
        self.rb_unFP += rb_unFP_num
        # Unseen False negative
        rb_unFN = (gt == 2) * (occ != 1) * (pred != 2)
        rb_unFN_num = rb_unFN.sum()
        self.rb_unFN += rb_unFN_num

        ### CAR
        # True positive
        car_TP = ((gt == 3) * (occ == 1) * (pred == 3))
        car_TP_num = car_TP.sum()
        self.car_TP += car_TP_num
        # False positive
        car_FP = ((gt != 3) * (occ == 1) * (pred == 3))
        car_FP_num = car_FP.sum()
        self.car_FP += car_FP_num
        # False negative
        car_FN = ((gt == 3) * (occ == 1) * (pred != 3))
        car_FN_num = car_FN.sum()
        self.car_FN += car_FN_num

        ### Unseen CAR
        # Unseen True positive
        car_unTP_num = (gt == 3) * (occ != 1) * (pred == 3)
        car_unTP_num = car_unTP_num.sum()
        self.car_unTP += car_unTP_num
        # Unseen False positive
        car_unFP_num = (gt != 3) * (occ != 1) * (pred == 3)
        car_unFP_num = car_unFP_num.sum()
        self.car_unFP += car_unFP_num
        # Unseen False negative
        car_unFN_num = (gt == 3) * (occ != 1) * (pred != 3)
        car_unFN_num = car_unFN_num.sum()
        self.car_unFN += car_unFN_num

        with np.errstate(all='ignore'):

            IOU_rb = rb_TP_num  / (rb_TP_num + rb_FP_num + rb_FN_num)
            IOU_car = car_TP_num  / (car_TP_num + car_FP_num + car_FN_num)

        return IOU_rb, IOU_car

    def calculate(self):
        with np.errstate(all='ignore'):
            self.IOU_rb = self.rb_TP / (self.rb_TP + self.rb_FP + self.rb_FN)
            self.IOU_car = self.car_TP / (self.car_TP + self.car_FP + self.car_FN)

        return self.IOU_rb, self.IOU_car

    def reset(self):
        self.rb_TP = 0
        self.rb_FP = 0
        self.rb_FN = 0
        self.rb_unTP = 0
        self.rb_unFP = 0
        self.rb_unFN = 0

        self.car_TP = 0
        self.car_FP = 0
        self.car_FN = 0
        self.car_unTP = 0
        self.car_unFP = 0
        self.car_unFN = 0

def merge_metrics(main, metric):
    main.rb_TP += metric.rb_TP
    main.rb_FP += metric.rb_FP
    main.rb_FN += metric.rb_FN
    main.rb_unTP += metric.rb_unTP
    main.rb_unFP += metric.rb_unFP
    main.rb_unFN += metric.rb_unFN

    main.car_TP += metric.car_TP
    main.car_FP += metric.car_FP
    main.car_FN += metric.car_FN
    main.car_unTP += metric.car_unTP
    main.car_unFP += metric.car_unFP
    main.car_unFN += metric.car_unFN

    main.IOU_rb += metric.IOU_rb
    main.IOU_car += metric.IOU_car

    return main

def results(seg, ps):
    print(f'IOU_seg: {seg.IOU_rb:.3f}, {seg.IOU_car:.3f} \t IOU_ps: {ps.IOU_rb:.3f}, {ps.IOU_car:.3f} \n'
          f'rb_TP: {seg.rb_TP} \t {ps.rb_TP} \n'
          f'rb_FP: {seg.rb_FP} \t {ps.rb_FP} \n'
          f'rb_FN: {seg.rb_FN} \t {ps.rb_FN} \n'
          
          f'car_TP: {seg.car_TP} \t {ps.car_TP} \n'
          f'car_FP: {seg.car_FP} \t {ps.car_FP} \n'
          f'car_FN: {seg.car_FN} \t {ps.car_FN} \n'
          
          f'Unseen \n'
          f'rb_unTP: {seg.rb_unTP} \t {ps.rb_unTP} \n'
          f'rb_unFP: {seg.rb_unFP} \t {ps.rb_unFP} \n'
          f'rb_unFN: {seg.rb_unFN} \t {ps.rb_unFN} \n'
          
          f'car_unTP: {seg.car_unTP} \t {ps.car_unTP} \n'
          f'car_unFP: {seg.car_unFP} \t {ps.car_unFP} \n'
          f'car_unFN: {seg.car_unFN} \t {ps.car_unFN} \n'

          )
