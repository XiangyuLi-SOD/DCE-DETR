'''   
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/自研模块-FDPN.png   
自研模块：FocusingDiffusionPyramidNetwork 
'''

import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
 
import warnings
warnings.filterwarnings('ignore')     
from calflops import calculate_flops

import copy
from collections import OrderedDict   

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.core import register
from engine.extre_module.ultralytics_nn.conv import Conv, autopad  
from engine.extre_module.ultralytics_nn.block import C2f
from engine.extre_module.custom_nn.attention.CASAB import CASAB
from engine.extre_module.custom_nn.attention.simam import SimAM
from engine.extre_module.custom_nn.attention.CBAM import CBAM
from engine.extre_module.custom_nn.attention.ema import EMA
from engine.deim.hybrid_encoder import RepNCSPELAN4

   
class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand    
        super().__init__()   
        self.c = c2 // 2  
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
 
    def forward(self, x):  
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True) 
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)     
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1) 


class MSGA(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9), e=0.5):
        super().__init__()
        hidc = int(inc[1] * e)
        num_scales = len(kernel_sizes)
        out_c = int(hidc / e)

        self.hidc = hidc
        self.num_scales = num_scales

        # 空间对齐
        self.conv1 = nn.Sequential(nn.Upsample(scale_factor=2), Conv(inc[0], hidc, 1))
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)

        # 语义门控：每一路（x1/x2/x3）每个尺度一个权重
        # 输出通道 = 3 * num_scales，对应 [scale0_x1, scale0_x2, scale0_x3, scale1_x1, ...]
        self.semantic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidc, 3 * num_scales, 1, bias=False),
            nn.BatchNorm2d(3 * num_scales),
            nn.Sigmoid()
        )

        # 多尺度 DW 卷积
        self.dw_conv = nn.ModuleList([
            nn.Conv2d(hidc * 3, hidc * 3, k, padding=autopad(k), groups=hidc * 3, bias=False)
            for k in kernel_sizes
        ])

        # 跨层注意力
        self.q_proj = nn.Conv2d(hidc * 3, hidc, 1, bias=False)
        self.k_proj = nn.Conv2d(hidc * 3, hidc, 1, bias=False)
        self.v_proj = nn.Conv2d(hidc * 3, hidc * 3, 1, bias=False)
        self.scale = hidc ** -0.5

        self.pw_conv = Conv(hidc * 3, hidc * 3)
        self.conv_1x1 = Conv(hidc * 3, out_c)

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x = torch.cat([x1, x2, x3], dim=1)  # (B, 3*hidc, H, W)


        # 语义门控：(B, 3*num_scales, 1, 1)
        sem = self.semantic_gate(x3)

        dw_outs = []
        for i in range(self.num_scales):
            # 取出当前尺度的3个调制参数（对应x1/x2/x3三路）
            w = sem[:, i * 3:(i + 1) * 3, :, :]  # (B, 3, 1, 1)
            # 扩展到 hidc 维度：每个调制参数复制 hidc 次
            w = w.repeat_interleave(self.hidc, dim=1)  # (B, 3*hidc, 1, 1)

            dw_out = self.dw_conv[i](x) * w
            dw_outs.append(dw_out)

        multi = torch.stack(dw_outs, dim=0).sum(dim=0)  # (B, 3*hidc, H, W)

        # 跨层注意力
        B, C, H, W = x.shape
        q = self.q_proj(x).view(B, self.hidc, H * W).permute(0, 2, 1)
        k = self.k_proj(multi).view(B, self.hidc, H * W)
        v = self.v_proj(multi).view(B, self.hidc * 3, H * W).permute(0, 2, 1)

        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, self.hidc * 3, H, W)

        out = self.pw_conv(out)
        return self.conv_1x1(x + out)











