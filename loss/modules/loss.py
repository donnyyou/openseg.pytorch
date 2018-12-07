import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.6, min_kept=0, use_weight=True):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[ min(len(index), self.min_kept) - 1 ]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            # print('hard ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class CenterLoss(nn.Module):
    def __init__(self, ignore_label=255):
        super(CenterLoss, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, x, centers, labels):
        n, c, h, w = x.size()
        num_classes,_ = centers.size()
        batch_size = n*h*w
        x = F.normalize(x, p=2, dim=1)
        centers = F.normalize(centers, p=2, dim=1)

        x = x.permute(0,2,3,1)
        x = x.contiguous().view(-1, c)
        labels = labels.contiguous().view(-1, 1)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
                torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, centers.t())

        classes = torch.arange(num_classes).long().cuda()
        labels = labels.expand(batch_size, num_classes) #
        mask = labels.data.eq(classes.expand(batch_size, num_classes))

        _indexes1 = range(batch_size)
        _indexes2 = mask[range(batch_size)]
        _distmat = distmat[_indexes1][_indexes2]

        loss = _distmat.mean()
        return loss


class BinaryOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.6, min_kept=0, use_weight=True):
        super(BinaryOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        # self.criterion = torch.nn.BCELoss()
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = x

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[ min(len(index), self.min_kept) - 1 ]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())
        return self.criterion(predict, target)


class OhemMse2d(nn.Module):
    def __init__(self, thres=0.2):
        super(OhemMse2d, self).__init__()
        self.thres = float(thres)
        self.criterion = torch.nn.MSELoss()

    def forward(self, predict, target):
        assert not target.requires_grad
        error = torch.abs(predict - target)
        valid_inds = error >= self.thres
        valid_predict = predict[valid_inds]
        valid_target = target[valid_inds]
        return self.criterion(valid_predict, valid_target)