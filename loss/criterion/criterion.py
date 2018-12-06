import pdb
import torch.nn as nn
import math
import os
import sys
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import scipy.ndimage as nd
from torch.nn import functional as F
from torch.autograd import Variable

from .loss import OhemCrossEntropy2d, CrossEntropy2d, CenterLoss, BinaryOhemCrossEntropy2d, OhemMse2d
from .extend_utils import down_sample_target
from .edge_criterion import Edge_mIOU_Criterion, Edge_F1_Criterion

torch_ver = torch.__version__[:3]

class Ce(nn.Module):
    '''
    Compute cross-entropy loss on the last output.
    '''
    def __init__(self, ignore_index=255):
        super(Ce, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear')
        loss = self.criterion(scale_pred, target)
        return loss
    

class Ce_Dsn(nn.Module):
    '''
    Compute cross-entropy loss on the last output and the intermediate output.
    '''
    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4):
        super(Ce_Dsn, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight*loss1 + loss2


class Edge_mIOU_Ce_Dsn(nn.Module):
    '''
    Compute cross-entropy loss on the last output and the intermediate output.
    '''
    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4, edge_weight=0.1):
        super(Edge_mIOU_Ce_Dsn, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.edge_weight = edge_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.edge_criterion = Edge_mIOU_Criterion()

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        loss3 = self.edge_criterion(scale_pred, target)
        # print('seg loss = {:.3f}, edge loss = {:.3f}'.format(loss2, loss3))
        return self.dsn_weight*loss1 + loss2 + self.edge_weight*loss3


class Edge_F1_Ce_Dsn(nn.Module):
    '''
    Compute cross-entropy loss on the last output and the intermediate output.
    '''
    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4, edge_weight=0.1):
        super(Edge_F1_Ce_Dsn, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.edge_weight = edge_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.edge_criterion = Edge_F1_Criterion()

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        loss3 = self.edge_criterion(scale_pred, target)
        # print('seg loss = {:.3f}, edge loss = {:.3f}'.format(loss2, loss3))
        return self.dsn_weight*loss1 + loss2 + self.edge_weight*loss3


class Ce_Quad(nn.Module):
    '''
    Compute cross-entropy loss on the last output and the intermediate output.
    '''
    def __init__(self, ignore_index=255, use_weight=True):
        super(Ce_Quad, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if torch_ver == '0.4':
            scale_pred_0 = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred_1 = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred_2 = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred_3 = F.upsample(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred_0 = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
            scale_pred_1 = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
            scale_pred_2 = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
            scale_pred_3 = F.upsample(input=preds[0], size=(h, w), mode='bilinear')

        loss0 = self.criterion(scale_pred_0, target)
        loss1 = self.criterion(scale_pred_1, target)
        loss2 = self.criterion(scale_pred_2, target)
        loss3 = self.criterion(scale_pred_3, target)

        return loss0 + loss1 + loss2 + loss3

class Ce_Triple(nn.Module):
    '''
    Compute cross-entropy loss on the last output and the intermediate output.
    '''
    def __init__(self, ignore_index=255, use_weight=True):
        super(Ce_Triple, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if torch_ver == '0.4':
            scale_pred_0 = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred_1 = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred_2 = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred_0 = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
            scale_pred_1 = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
            scale_pred_2 = F.upsample(input=preds[0], size=(h, w), mode='bilinear')

        loss0 = self.criterion(scale_pred_0, target)
        loss1 = self.criterion(scale_pred_1, target)
        loss2 = self.criterion(scale_pred_2, target)

        return 0.4*loss0 + loss1 + loss2

class Ce_Sphere_Dsn(nn.Module):
    '''
    Compute cross-entropy loss on the last output and the intermediate output.
    '''
    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4):
        super(Ce_Sphere_Dsn, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss3 = self.criterion(scale_pred, target)

        return self.dsn_weight*loss1 + loss2 + loss3

class Ce_Dsn_Ohem(nn.Module):
    '''
    Compute cross-entropy loss with hard-sampling mining on both two branches.
    '''
    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000, dsn_weight=0.4, use_weight=True):
        super(Ce_Dsn_Ohem, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=use_weight)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight*loss1 + loss2

class Ce_Dsn_Ohem_Single(nn.Module):
    '''
    Compute cross-entropy loss with hard-sampling mining on the main branch.
    '''
    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000, dsn_weight=0.4):
        super(Ce_Dsn_Ohem_Single, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.criterion_ohem = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=True)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion_ohem(scale_pred, target)
        return self.dsn_weight*loss1 + loss2


class Ce_Proxy(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4, center_weight=0.1):
        super(Ce_Proxy, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.center_weight = center_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.criterion_center = CenterLoss(ignore_label=ignore_index)


    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[3], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)

        scale_target = down_sample_target(target, 8)
        center_loss = self.criterion_center(preds[0], preds[1], scale_target)
        print('center loss: {}'.format(self.center_weight*center_loss.data.cpu().numpy()))
        return self.dsn_weight*loss1 + loss2 + self.center_weight*center_loss 


class Mse_Aff(nn.Module):
    '''
    Compute cross-entropy loss + mse based affinity loss.
    '''
    def __init__(self, ignore_index=255, dsn_weight=0.4, pair_weight=1):
        super(Mse_Aff, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.affinity_weight = pair_weight
        self.affinity_criterion = torch.nn.MSELoss()

        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        batch_size, h, w = target.size(0), target.size(1), target.size(2)
        # affinity loss
        preds_h, preds_w = preds[1].size(2), preds[1].size(3)
        label = F.upsample(input=target.unsqueeze(1).type(torch.cuda.FloatTensor), size=(preds_h, preds_w), mode='nearest')
        label_row_vec = label.view(batch_size, 1, -1).expand(batch_size, preds_h * preds_w, preds_h * preds_w)
        label_col_vec = label_row_vec.permute(0, 2, 1)
        pair_label = label_col_vec.eq(label_row_vec)
        affinity_gt = pair_label.type(torch.cuda.FloatTensor)
        affinity_pred = preds[0].type(torch.cuda.FloatTensor)
        affinity_loss = self.affinity_criterion(affinity_pred, affinity_gt)

        # segmentation loss
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)

        print('affinity loss: {}'.format(self.affinity_weight*affinity_loss.data.cpu().numpy()))
        return self.dsn_weight*loss1 + loss2 + self.affinity_weight*affinity_loss


class Mse_Aff_01(nn.Module):
    '''
    Compute cross-entropy loss + mse based affinity loss.
    '''
    def __init__(self, ignore_index=255, dsn_weight=0.4, pair_weight=0.1):
        super(Mse_Aff_01, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.affinity_weight = pair_weight
        self.affinity_criterion = torch.nn.MSELoss()

        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        batch_size, h, w = target.size(0), target.size(1), target.size(2)
        # affinity loss
        preds_h, preds_w = preds[1].size(2), preds[1].size(3)
        label = F.upsample(input=target.unsqueeze(1).type(torch.cuda.FloatTensor), size=(preds_h, preds_w), mode='nearest')
        label_row_vec = label.view(batch_size, 1, -1).expand(batch_size, preds_h * preds_w, preds_h * preds_w)
        label_col_vec = label_row_vec.permute(0, 2, 1)
        pair_label = label_col_vec.eq(label_row_vec)
        affinity_gt = pair_label.type(torch.cuda.FloatTensor)
        affinity_pred = preds[0].type(torch.cuda.FloatTensor)
        affinity_loss = self.affinity_criterion(affinity_pred, affinity_gt)

        # segmentation loss
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)

        print('affinity loss: {}'.format(self.affinity_weight*affinity_loss.data.cpu().numpy()))
        return self.dsn_weight*loss1 + loss2 + self.affinity_weight*affinity_loss


class Mse_Aff_0(nn.Module):
    '''
    Compute cross-entropy loss + mse based affinity loss.
    '''
    def __init__(self, ignore_index=255, dsn_weight=0.4):
        super(Mse_Aff_0, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        batch_size, h, w = target.size(0), target.size(1), target.size(2)
        # segmentation loss
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight*loss1 + loss2 


class Mse_Aff_Ohem(nn.Module):
    '''
    Compute cross-entropy loss + mse based affinity loss.
    '''
    def __init__(self, ignore_index=255, dsn_weight=0.4, pair_weight=1):
        super(Mse_Aff_Ohem, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.affinity_weight = pair_weight
        self.affinity_criterion = OhemMse2d()

        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        batch_size, h, w = target.size(0), target.size(1), target.size(2)
        # affinity loss
        pdb.set_trace()
        preds_h, preds_w = preds[1].size(2), preds[1].size(3)
        label = F.upsample(input=target.unsqueeze(1).type(torch.cuda.FloatTensor), size=(preds_h, preds_w), mode='nearest')
        label_row_vec = label.view(batch_size, 1, -1).expand(batch_size, preds_h * preds_w, preds_h * preds_w)
        label_col_vec = label_row_vec.permute(0, 2, 1)
        pair_label = label_col_vec.eq(label_row_vec)
        affinity_gt = pair_label.type(torch.cuda.FloatTensor)
        affinity_pred = preds[0].type(torch.cuda.FloatTensor)
        affinity_loss = self.affinity_criterion(affinity_pred, affinity_gt)

        # segmentation loss
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)

        print('affinity loss: {}'.format(self.affinity_weight*affinity_loss.data.cpu().numpy()))
        return self.dsn_weight*loss1 + loss2 + self.affinity_weight*affinity_loss


class Ce_Aff(nn.Module):
    '''
    Compute cross-entropy loss + cross-entropy based transposed affinity loss.
    '''
    def __init__(self, ignore_index=255, dsn_weight=0.4, pair_weight=1):
        super(Ce_Aff, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.affinity_weight = pair_weight
        self.affinity_criterion = torch.nn.CrossEntropyLoss()

        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        batch_size, h, w = target.size(0), target.size(1), target.size(2)
        # affinity loss
        preds_h, preds_w = 48, 48
        label = F.upsample(input=target.unsqueeze(1).type(torch.cuda.FloatTensor), size=(preds_h, preds_w), mode='nearest')
        label_row_vec = label.view(batch_size, 1, -1).expand(batch_size, preds_h * preds_w, preds_h * preds_w)
        label_col_vec = label_row_vec.permute(0, 2, 1)
        pair_label = label_col_vec.eq(label_row_vec)
        pair_label = pair_label.permute(0, 2, 1)
        affinity_gt = pair_label.type(torch.cuda.LongTensor)
        affinity_pred = preds[0].type(torch.cuda.FloatTensor)
        affinity_pred = affinity_pred.unsqueeze(1)
        # 
        ext_affinity_pred = torch.cat([affinity_pred, 1-affinity_pred], 1)
        affinity_loss = self.affinity_criterion(ext_affinity_pred, 1-affinity_gt)

        # segmentation loss
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)

        print('affinity loss: {}'.format(self.affinity_weight*affinity_loss.data.cpu().numpy()))
        return self.dsn_weight*loss1 + loss2 + self.affinity_weight*affinity_loss


class Mse_Aff_T(nn.Module):
    '''
    Train the model to predict the pair-wise affinity.
    '''
    def __init__(self, ignore_index=255, dsn_weight=0.4, pair_weight=1):
        super(Mse_Aff_T, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.affinity_weight = pair_weight
        self.affinity_criterion = torch.nn.MSELoss() 
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        batch_size, h, w = target.size(0), target.size(1), target.size(2)
        # affinity loss
        preds_h, preds_w = 48, 48
        label = F.upsample(input=target.unsqueeze(1).type(torch.cuda.FloatTensor), size=(preds_h, preds_w), mode='nearest')
        label_row_vec = label.view(batch_size, 1, -1).expand(batch_size, preds_h * preds_w, preds_h * preds_w)
        label_col_vec = label_row_vec.permute(0, 2, 1)
        pair_label = label_col_vec.eq(label_row_vec)
        pair_label = pair_label.permute(0, 2, 1)
        affinity_gt = pair_label.type(torch.cuda.FloatTensor)
        affinity_pred = preds[0].type(torch.cuda.FloatTensor)
        affinity_loss = self.affinity_criterion(affinity_pred, affinity_gt)

        # segmentation loss
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        print('affinity loss: {}'.format(self.affinity_weight*affinity_loss.data.cpu().numpy()))
        return self.dsn_weight*loss1 + loss2 + self.affinity_weight*affinity_loss


# class CriterionAffinityOhem(nn.Module):
#     '''
#     Train the model to predict the pair-wise affinity.
#     '''
#     def __init__(self, ignore_index=255, dsn_weight=0.4, pair_weight=1):
#         super(CriterionAffinityOhem, self).__init__()
#         self.ignore_index = ignore_index
#         self.dsn_weight = dsn_weight
#         self.affinity_weight = pair_weight
#         self.affinity_criterion = BinaryOhemCrossEntropy2d(thresh=0.7)
#         weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
#         self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
#         self.relu = nn.ReLU()

#     def forward(self, preds, target):
#         batch_size, h, w = target.size(0), target.size(1), target.size(2)
#         # preds_h, preds_w = preds[1].size(2), preds[1].size(3)
#         preds_h, preds_w = 48, 48
#         label = F.upsample(input=target.unsqueeze(1).type(torch.cuda.FloatTensor), size=(preds_h, preds_w), mode='nearest')
#         label_row_vec = label.view(batch_size, 1, -1).expand(batch_size, preds_h * preds_w, preds_h * preds_w)
#         label_col_vec = label_row_vec.permute(0, 2, 1)
#         pair_label = label_col_vec.eq(label_row_vec)

#         affinity_gt = pair_label.type(torch.cuda.LongTensor)
#         affinity_pred = preds[0].type(torch.cuda.FloatTensor)
#         affinity_pred = affinity_pred.unsqueeze(1)
#         # affinity_pred = torch.clamp(affinity_pred, min=0, max=1)
#         ext_affinity_pred = torch.cat([affinity_pred, 1-affinity_pred], 1)
#         affinity_loss = self.affinity_criterion(ext_affinity_pred, 1-affinity_gt)

#         # segmentation loss
#         if torch_ver == '0.4':
#             scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
#         else:
#             scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
#         loss1 = self.criterion(scale_pred, target)

#         if torch_ver == '0.4':
#             scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
#         else:
#             scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
#         loss2 = self.criterion(scale_pred, target)
#         print('affinity loss: {} seg loss: {}'.format(affinity_loss.data.cpu().numpy()), loss2.data.cpu().numpy()[0])
#         return self.dsn_weight*loss1 + loss2 + self.affinity_weight*affinity_loss


class Ce_Edge(nn.Module):
    '''
        Reference:  CE2P
                    https://github.com/liutinglt/CE2P/blob/master/train.py
    '''
    def __init__(self, ignore_index=255):
        super(Ce_Edge, self).__init__()
        self.ignore_index = ignore_index
          
    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        
        input_labels = target.data.cpu().numpy().astype(np.int64)
        pos_num = np.sum(input_labels==1).astype(np.float)
        neg_num = np.sum(input_labels==0).astype(np.float)
        
        weight_pos = neg_num/(pos_num+neg_num)
        weight_neg = pos_num/(pos_num+neg_num)
        weights = (weight_neg, weight_pos)  
        weights = Variable(torch.from_numpy(np.array(weights)).float().cuda())
        
        scale_pred1 = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = F.cross_entropy(scale_pred1, target, weights )
        scale_pred2 = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = F.cross_entropy(scale_pred2, target, weights )
        scale_pred3 = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss3 = F.cross_entropy(scale_pred3, target, weights )
        scale_pred4 = F.upsample(input=preds[3], size=(h, w), mode='bilinear')
        loss4 = F.cross_entropy(scale_pred4, target, weights )
        scale_pred5 = F.upsample(input=preds[4], size=(h, w), mode='bilinear')
        loss5 = F.cross_entropy(scale_pred5, target, weights ) 
        scale_pred6 = F.upsample(input=preds[5], size=(h, w), mode='bilinear')
        loss6 = F.cross_entropy(scale_pred6, target, weights ) 

        return loss1 + loss2 + loss3 + loss4 + loss5 + loss6


class Ce_Edge_Parse(nn.Module):
    '''
        Reference:  CE2P
                    https://github.com/liutinglt/CE2P/blob/master/train.py
    '''
    def __init__(self, ignore_index=255):
        super(Ce_Edge_Parse, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) 
          
    def forward(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)
        
        input_labels = target[1].data.cpu().numpy().astype(np.int64)
        pos_num = np.sum(input_labels==1).astype(np.float)
        neg_num = np.sum(input_labels==0).astype(np.float)
        
        weight_pos = neg_num/(pos_num+neg_num)
        weight_neg = pos_num/(pos_num+neg_num)
        weights = (weight_neg, weight_pos)  
        weights = Variable(torch.from_numpy(np.array(weights)).float().cuda())
        
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss = self.criterion(scale_pred, target[0])
        
        scale_pred1 = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred1, target[0])
     
        scale_pred2 = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss2 = F.cross_entropy(scale_pred2, target[1], weights )
        return loss+loss1+loss2
