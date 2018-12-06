##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: https://github.com/AlexHex7/Non-local_pytorch
## updated by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import os
import sys
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import functools

sys.path.append('/inplace_abn')
from modules import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


def _POSITION_ENCODEING_INIT_SINCOS(batch, x_position, y_position, d_pos_vec):
    ''' 
    Instead of the original position encoding, we scale the position range to [0-10] and [0-20], we just
    use the original size to controll the stride of our position encoding.

       Refer to "Attention Is All You Need (NIPS-2017)"
       Init the positional encoding with sin/cos functions, the encodings are fixed during both
       training and testing.
       Return Feature Shape:  N X H X W X D
       Maybe it is better to implement the position encoding module as a class.(To do)
    '''
    position_enc = np.array(
        [
        [
            [
             [pos_y / np.power(10000, 2 * (i // 2) / d_pos_vec) for i in range(d_pos_vec)]+
             [pos_x / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
              for pos_y in np.linspace(0, 20, y_position)
            ] for pos_x in np.linspace(0, 10, x_position)
        ] for k in range(batch)
        ]
    )
    position_enc[:,:,:,0::2] = np.sin(position_enc[:,:,:,0::2])  # dim 2i
    position_enc[:,:,:,1::2] = np.cos(position_enc[:,:,:,1::2])  # dim 2i+1
    return nn.Parameter(torch.from_numpy(np.transpose(position_enc, (0, 3, 1, 2))).type(torch.FloatTensor).cuda(), requires_grad=False)


def _POSITION_ENCODEING(batch, x_position, y_position, d_pos_vec):
    '''
        Randomly init the positional encodings, the encodings will be learned during training
        and fixed during testing.
        Return Feature Shape:  N X H X W X D
    '''
    # stdv = 1. / math.sqrt(d_pos_vec)
    position_enc = nn.Parameter(torch.randn(batch, d_pos_vec, x_position, y_position).type(torch.FloatTensor).cuda(), requires_grad=True)
    init.xavier_uniform(position_enc)
    return position_enc


class _NonLocalBlockND_POS(nn.Module):
    '''
    Warning!
        Due to the huge memory cost related to the point-wise attention maps, we add or concate the position features 
        when we compute the attention maps, we neglect the position features along the branch for Value transform.

    Input:
        N X C X H X W
    Parameters:
        in_channels: the dimension of the input feature map
        c1         : the dimension of W_theta and W_phi
        c2         : the dimension of W_g and W_rho
        bn_layer   : whether use BN within W_rho
        use_g / use_w: whether use the W_g transform and W_rho transform
        scale      : choose the scale to downsample the input feature maps
        fuse_method: choose the fusion method to fuse the two affinity matrix, such as, product, add, mix
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, c1, c2, out_channels=None, mode='embedded_gaussian', sub_sample=False,
                 bn_layer=False, scale=1, value_w_pos=True, fuse_method="concate", init_pos=True, choice=0):
        super(_NonLocalBlockND_POS, self).__init__()

        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']
        self.choice = choice
        self.value_w_pos = value_w_pos
        self.fuse_method = fuse_method
        self.mode = mode
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = c1
        self.context_channels = in_channels
        self.init_pos = init_pos
        self.context_channels = c2
        if out_channels == None:
            self.out_channels = in_channels

        self.pool = nn.AvgPool2d(kernel_size=(scale, scale))
        self.pos_encode = _POSITION_ENCODEING_INIT_SINCOS(1, 97, 97, self.in_channels//2)
        self.relu = nn.ReLU()

        # Value function: apply only on the input features.
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.context_channels,
            kernel_size=1, stride=1, padding=0)
        self.g_add = nn.Conv2d(in_channels=self.in_channels, out_channels=self.context_channels,
            kernel_size=1, stride=1, padding=0)
        self.g_concate = nn.Conv2d(in_channels=2*self.in_channels, out_channels=self.context_channels,
            kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.context_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

        # Key / Query function for adding [Semantic feature, Position feature]
        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=False)
        )
        self.phi = self.theta

        # Key / Query function for concatenating [Semantic feature, Position feature]
        self.theta_concate = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels*2, out_channels=self.inter_channels,
                kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=False)
        )
        self.phi_concate = self.theta_concate

        # The transform functions on the Position feature.
        if self.choice == 0:
            #Apply no transform
            pass
        elif self.choice == 1:
            self.pos_f = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels,
                    kernel_size=1, stride=1, padding=0),
                BatchNorm2d(self.in_channels),
                nn.ReLU()
            )
        elif self.choice == 2:
            self.pos_f = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels,
                    kernel_size=1, stride=1, padding=0),
                BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels,
                    kernel_size=1, stride=1, padding=0),
                BatchNorm2d(self.in_channels),
                nn.ReLU(),                    
            )


    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # Transform the Position Encoding features.
        pos_embedding = self.pos_encode.repeat(batch_size,1,1,1)
        if h > pos_embedding.size(2):
            pos_embedding = F.upsample(input=pos_embedding, size=(h, w), mode='bilinear')
        if self.choice > 0:
            pos_embedding = self.pos_f(pos_embedding)

        # Value transform
        if not self.value_w_pos:
            g_x = self.g(x).view(batch_size, self.in_channels, -1)
            g_x = g_x.permute(0, 2, 1)

        # Key/ Query transform
        if self.fuse_method == "add":
            x_mix = torch.add(x, pos_embedding)
            theta_x = self.theta(x_mix).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x_mix).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_fuse = (self.inter_channels**-.5) * f
            G = F.softmax(f_fuse, dim=-1)
            if self.value_w_pos:
                g_x = self.g_add(x_mix).view(batch_size, self.in_channels, -1)
                g_x = g_x.permute(0, 2, 1)

        elif self.fuse_method == "concate":
            x_mix = torch.cat([x, pos_embedding], 1)
            theta_x = self.theta_concate(x_mix).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi_concate(x_mix).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_fuse = (self.inter_channels**-.5) * f
            G = F.softmax(f_fuse, dim=-1)
            if self.value_w_pos:
                g_x = self.g_concate(x_mix).view(batch_size, self.in_channels, -1)
                g_x = g_x.permute(0, 2, 1)

        y = torch.matmul(G, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.context_channels, *x.size()[2:])
        W_y = self.W(y)

        if self.scale > 1:
            W_y = F.upsample(input=W_y, size=(h, w), mode='bilinear')

        return W_y


class NONLocalBlock2D(_NonLocalBlockND_POS):
    def __init__(self, in_channels, c1=None, c2=None, out_channels=None, mode='embedded_gaussian', 
        bn_layer=False, scale=1, value_w_pos=True, fuse_method="product", init_pos=True, choice=0):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              c1=c1,
                                              c2=c2,
                                              out_channels=out_channels,     
                                              mode=mode,
                                              bn_layer=bn_layer,
                                              scale=scale, 
                                              value_w_pos=value_w_pos,
                                              fuse_method=fuse_method,
                                              init_pos=init_pos,
                                              choice=choice)


class BilateralAttentionContextModule(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
    Return:
        features after "concat" or "add"
    """
    def __init__(self, in_channels, out_channels, c1, c2, dropout, fusion="concat", sizes=([1]), init_pos=True, choice=1):
        super(BilateralAttentionContextModule, self).__init__()
        self.fusion = fusion
        self.choice = choice
        self.init_pos = init_pos
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, c1, c2, size) for size in sizes])
        self.bottleneck_add = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(dropout)
            )
        self.bottleneck_concat = nn.Sequential(
            nn.Conv2d(2*in_channels, out_channels, kernel_size=1, padding=0),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(dropout)
            )

    def _make_stage(self, in_channels, c1, c2, size):
        if self.choice == 1:
            return NONLocalBlock2D(in_channels=in_channels, c1=c1, c2=c2, mode='dot_product', 
                scale=size, value_w_pos=False, fuse_method="concate", init_pos=self.init_pos, choice=0)
        elif self.choice == 2: #793373901750813
            return NONLocalBlock2D(in_channels=in_channels, c1=c1, c2=c2, mode='dot_product', 
                scale=size, value_w_pos=False, fuse_method="concate", init_pos=self.init_pos, choice=1)
        elif self.choice == 3:
            return NONLocalBlock2D(in_channels=in_channels, c1=c1, c2=c2, mode='dot_product', 
                scale=size, value_w_pos=False, fuse_method="concate", init_pos=self.init_pos, choice=2)
        elif self.choice == 4:
            return NONLocalBlock2D(in_channels=in_channels, c1=c1, c2=c2, mode='dot_product', 
                scale=size, value_w_pos=True, fuse_method="concate", init_pos=self.init_pos, choice=0)
        elif self.choice == 5:
            return NONLocalBlock2D(in_channels=in_channels, c1=c1, c2=c2, mode='dot_product', 
                scale=size, value_w_pos=False, fuse_method="add", init_pos=self.init_pos, choice=0)
        elif self.choice == 6:
            return NONLocalBlock2D(in_channels=in_channels, c1=c1, c2=c2, mode='dot_product', 
                scale=size, value_w_pos=False, fuse_method="add", init_pos=self.init_pos, choice=1)
        elif self.choice == 7:
            return NONLocalBlock2D(in_channels=in_channels, c1=c1, c2=c2, mode='dot_product', 
                scale=size, value_w_pos=False, fuse_method="add", init_pos=self.init_pos, choice=2) 
        elif self.choice == 8:
            return NONLocalBlock2D(in_channels=in_channels, c1=c1, c2=c2, mode='dot_product', 
                scale=size, value_w_pos=True, fuse_method="add", init_pos=self.init_pos, choice=0)            

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        if self.fusion == "concat":
            bottle = self.bottleneck_concat(torch.cat([context, feats], 1))
        else:
            bottle = self.bottleneck_add(context + feats)
        return bottle
