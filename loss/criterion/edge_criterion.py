import pdb
import torch.nn as nn
import math
import os
import sys
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
import cv2
import scipy.ndimage as nd



class Edge_mIOU_Criterion(nn.Module):
    '''
    Edge-Mean-IOU-Loss: We compute the mean-IOU loss over the edge regions.
    '''
    def __init__(self, ignore_index=255):
        super(Edge_mIOU_Criterion, self).__init__()
        self.ignore = ignore_index
        
        disk = np.array([[0,   0,   0,   1,   0,   0,   0],
                         [0,   1,   1,   1,   1,   1,   0],
                         [0,   1,   1,   1,   1,   1,   0],
                         [1,   1,   1,   1,   1,   1,   1], 
                         [0,   1,   1,   1,   1,   1,   0],
                         [0,   1,   1,   1,   1,   1,   0],
                         [0,   0,   0,   1,   0,   0,   0]])
        disk = disk.reshape([1, 1, 7, 7])
        disk = torch.from_numpy(disk.copy()).float()
        self.disk = disk

        Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        Gx = Gx.reshape([1,1,3,3])
        Gx = torch.from_numpy(Gx.copy()).float()

        Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        Gy = Gy.reshape([1,1,3,3])
        Gy = torch.from_numpy(Gy.copy()).float()

        self.Gx = Gx
        self.Gy = Gy
        print("use IoU for Segmentation and IoU for Boundary\n")

    def forward(self, pred_seg, target):
        n, h, w = target.size(0), target.size(1), target.size(2)
        
        # compute the miou loss
        mask = np.array(target)
        mask = mask.reshape([target.size(0),1,h,w])
        mask = torch.from_numpy(mask.copy()).float()
        Ed_x = F.conv2d(mask, self.Gx, padding=1).abs()
        Ed_y = F.conv2d(mask, self.Gy, padding=1).abs()
        Ed_target = F.conv2d(Ed_x + Ed_y, self.disk, padding=1).abs()
        
        scale_pred = F.upsample(input=pred_seg, size=(h, w), mode='bilinear', align_corners=True)
        mask = scale_pred.argmax(dim=1, keepdim=True)
        mask = mask.type('torch.FloatTensor')
        Ed_x = F.conv2d(mask, self.Gx, padding=1).abs()
        Ed_y = F.conv2d(mask, self.Gy, padding=1).abs()
        Ed_pred = F.conv2d(Ed_x + Ed_y, self.disk, padding=1).abs()

        intersection = ((Ed_pred > 0) & (Ed_target > 0)).sum()
        union = ((Ed_pred > 0) | ((Ed_target > 0) & (Ed_target != self.ignore))).sum()

        if union > 0:
           loss_bnd = 1 - intersection.type('torch.cuda.FloatTensor') / union.type('torch.cuda.FloatTensor')
        else:
           loss_bnd = torch.tensor(0).type('torch.cuda.FloatTensor')

        return loss_bnd.type('torch.cuda.FloatTensor')



class Edge_F1_Criterion(nn.Module):
    '''
    Edge-F1-Loss: We compute the F1 score based loss over the edge regions.
    '''

    def __init__(self, ignore_index=255, d_disk=7):
        super(Edge_F1_Criterion, self).__init__()
        self.ignore = ignore_index

        if d_disk == 1:
           disk = np.array([[1]])
           disk = disk.reshape([1, 1, 1, 1])
        elif d_disk == 3:
           disk = np.array([[0,   1,  0],
                            [1,   1,  1],
                            [0,   1,  0]])
           disk = disk.reshape([1, 1, 3, 3])        
        elif d_disk == 7:
           disk = np.array([[0,   0,   0,   1,   0,   0,   0],
                            [0,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   0],
                            [1,   1,   1,   1,   1,   1,   1], 
                            [0,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   0],
                            [0,   0,   0,   1,   0,   0,   0]])
           disk = disk.reshape([1, 1, 7, 7])
        elif d_disk == 5:
           disk = np.array([[0,   0,   1,   0,   0],
                            [0,   1,   1,   1,   0],
                            [1,   1,   1,   1,   1],
                            [0,   1,   1,   1,   0],
                            [0,   0,   1,   0,   0]])
           disk = disk.reshape([1, 1, 5, 5])
        elif d_disk == 9:
           disk = np.array([[0,   0,   0,   0,   1,   0,   0,   0,   0],
                            [0,   0,   1,   1,   1,   1,   1,   0,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   0],
                            [1,   1,   1,   1,   1,   1,   1,   1,   1],
                            [0,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   0,   1,   1,   1,   1,   1,   0,   0],
                            [0,   0,   0,   0,   1,   0,   0,   0,   0]])
           disk = disk.reshape([1, 1, 9, 9])
        elif d_disk == 11:
           disk = np.array([[0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0],
                            [0,   0,   1,   1,   1,   1,   1,   1,   1,   0,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   0,   1,   1,   1,   1,   1,   1,   1,   0,   0],
                            [0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0]])                       
           disk = disk.reshape([1, 1, 11, 11])
        elif d_disk == 13:
           disk = np.array([[0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0],
                            [0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0],
                            [0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
                            [0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],   
                            [0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0],     
                            [0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0],
                            [0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0]])                    
           disk = disk.reshape([1, 1, 13, 13])

        disk = torch.from_numpy(disk.copy()).float()
        self.disk = disk
        self.padding = np.int((d_disk-1)/2)

        Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        Gx = Gx.reshape([1,1,3,3])
        Gx = torch.from_numpy(Gx.copy()).float()

        Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        Gy = Gy.reshape([1,1,3,3])
        Gy = torch.from_numpy(Gy.copy()).float()

        self.Gx = Gx
        self.Gy = Gy
        print("use IoU for Segmentation and Fscore for Boundary\n")

    def forward(self, pred_seg, target):
        n, h, w = target.size(0), target.size(1), target.size(2)

        # compute the miou-based loss
        mask = np.array(target)
        mask = mask.reshape([target.size(0),1,h,w])
        mask = torch.from_numpy(mask.copy()).float()
        Ed_x = F.conv2d(mask, self.Gx, padding=1).abs()
        Ed_y = F.conv2d(mask, self.Gy, padding=1).abs()
        Ed_lbl    = Ed_x + Ed_y
        Ed_target = F.conv2d(Ed_x + Ed_y, self.disk, padding=self.padding).abs()

        scale_pred = F.upsample(input=pred_seg, size=(h, w), mode='bilinear', align_corners=True)
        mask = scale_pred.argmax(dim=1, keepdim=True)
        mask = mask.type('torch.FloatTensor')
        Ed_x = F.conv2d(mask, self.Gx, padding=1).abs()
        Ed_y = F.conv2d(mask, self.Gy, padding=1).abs()
        Ed_res  = Ed_x + Ed_y
        Ed_pred = F.conv2d(Ed_x + Ed_y, self.disk, padding=self.padding).abs()

        precision = ((Ed_res > 0) & (Ed_target > 0)).sum().type('torch.cuda.FloatTensor')/(Ed_res > 0).sum().type('torch.cuda.FloatTensor')
        recall    = ((Ed_lbl > 0) & (Ed_pred > 0)).sum().type('torch.cuda.FloatTensor')/(Ed_lbl > 0).sum().type('torch.cuda.FloatTensor')


        if precision+recall > 0:
           loss_bnd = 1 - 2*precision*recall/(precision+recall)
        else:
           loss_bnd = torch.tensor(0).type('torch.cuda.FloatTensor')
 
        return loss_bnd.type('torch.cuda.FloatTensor')
