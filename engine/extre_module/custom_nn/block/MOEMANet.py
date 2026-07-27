import os, sys

from numpy.ma.core import identity

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings

warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from engine.extre_module.ultralytics_nn.conv import Conv, DWConv
from engine.extre_module.ultralytics_nn.block import Bottleneck



# =========================================================================
# 2. 路由模块 (Routing Mechanism)
# =========================================================================

class EfficientSpatialRouter(nn.Module):
    """
    轻量级空间路由器，生成 Top-K 索引和权重。
    """

    def __init__(self, in_channels, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 降维后进行分类，减少参数量
        self.reduction = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.classifier = nn.Conv2d(in_channels // 2, num_experts, 1)

        # 初始化：确保初始状态下各个专家负载相对均衡
        nn.init.normal_(self.classifier.weight, mean=0, std=0.01)

    def forward(self, x):
        # x: [B, C, H, W]
        # Generate logits: [B, num_experts, H, W]
        # 这里为了简化计算，我们做全局路由 (Sample-level) 或 空间路由 (Pixel-level)
        # 本实现采用 Sample-level (对整个特征图平均池化后决策) 以节省FLOPs
        # 若需 Pixel-level，去掉 AdaptiveAvgPool 即可

        pooled = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        logits = self.classifier(self.reduction(pooled)).view(x.shape[0], -1)  # [B, num_experts]

        # Gumbel-Softmax trick or Standard Softmax usually
        # 这里使用 Standard TopK
        probs = F.softmax(logits, dim=-1)

        # 选出 Top-K
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        # 归一化权重
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 返回 loss_dict 供训练使用
        loss_dict = {
            'router_probs': probs,
            'router_logits': logits
        }

        return topk_weights, topk_indices, loss_dict


# =========================================================================
# 3. 核心创新：尺度特异性专家混合 (Scale-Specific MoE)
# =========================================================================

class ScaleSpecificMoE(nn.Module):
    """
    SS-MoE: 包含不同感受野专家的混合模块
    """

    def __init__(self, c_in, c_out, num_experts=4, top_k=2):  #修改top_k
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.c_in = c_in
        self.c_out = c_out

        # 1) Router
        self.router = EfficientSpatialRouter(c_in, num_experts, top_k)

        # 2) Heterogeneous Experts (异构专家池)
        # 创新点：专家不再是同构的，而是分尺度的
        self.experts = nn.ModuleList()

        # Expert 0: 3x3 Conv (Standard Details)
        self.experts.append(Conv(c_in, c_out, k=3))

        # Expert 1: 3x3 Conv (Standard Details)
        self.experts.append(Conv(c_in, c_out, k=3))

        # Expert 2: 5x5 DWConv (Medium Scale Context)
        self.experts.append(DWConv(c_in, c_out, k=5))

        # Expert 3: 7x7 DWConv (Large Scale Context)
        # 注意：使用 DWConv 避免大核导致参数量爆炸
        self.experts.append(DWConv(c_in, c_out, k=7))

        # 3) Shared Expert (Shared Path)
        # 总是激活，保证基础特征流，防止 Mode Collapse
        self.shared_expert = Conv(c_in, c_out, k=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 计算 Shared Path
        shared_out = self.shared_expert(x)

        # 2. 路由
        # weights: [B, top_k], indices: [B, top_k]
        weights, indices, _ = self.router(x)

        # 3. 稀疏专家计算 (Sparse Computation Loop)
        expert_output = torch.zeros(B, self.c_out, H, W, device=x.device, dtype=x.dtype)

        # 展平以便索引
        indices_flat = indices.view(-1)  # [B * top_k]
        weights_flat = weights.view(-1)  # [B * top_k]

        # 遍历所有专家
        for i in range(self.num_experts):
            # 找出分配给专家 i 的样本索引 (Batch维度)
            # 这是一个 mask: [B, top_k]
            mask = (indices == i)

            if mask.any():
                # 获取 Batch 索引
                # batch_indices 代表第几个样本需要用到该专家
                batch_indices, k_idx = torch.where(mask)

                # 提取对应的输入样本
                inp = x[batch_indices]  # [Sub_Batch, C, H, W]

                # 专家计算
                out = self.experts[i](inp)

                # 获取对应的权重
                w = weights[batch_indices, k_idx].view(-1, 1, 1, 1)  # [Sub_Batch, 1, 1, 1]

                # 加权并累加回输出 tensor
                # index_add_ 是稀疏计算的关键
                expert_output.index_add_(0, batch_indices, out * w)

        # 4. 融合结果
        return shared_out + expert_output


# =========================================================================
# 4. 整体网络：MANet + SS-MoE
# =========================================================================

class MoEScaleBranch(nn.Module):


    def __init__(self, c_in, n=1):
        super().__init__()
        # c_in 对应原代码中的 2 * self.c
        self.mid_c = c_in // 2

        # 这里的 n 决定了 MoE 的深度
        if n == 1:
            self.moe_layer = ScaleSpecificMoE(self.mid_c, self.mid_c)
        else:
            self.moe_layer = nn.Sequential(
                *[ScaleSpecificMoE(self.mid_c, self.mid_c) for _ in range(n)]
            )

    def forward(self, x):
        # 对应 y2, y3 = y.chunk(2, 1)
        # y2 是恒等分支，y3 进入专家系统
        y2, y3 = x.chunk(2, 1)

        y3_out = self.moe_layer(y3)

        # 返回拼接结果，保持通道数与输入一致
        return torch.cat([y2, y3_out], dim=1)

# class MoEScaleBranch(nn.Module):    #no chunk
#     def __init__(self, c_in, n=1):
#         super().__init__()
#         # 不 split，MoE 处理全部通道
#         if n == 1:
#             self.moe_layer = ScaleSpecificMoE(c_in, c_in)  # 输入输出都是 2*self.c
#         else:
#             self.moe_layer = nn.Sequential(
#                 *[ScaleSpecificMoE(c_in, c_in) for _ in range(n)]
#             )
#
#     def forward(self, x):
#         identity = x
#         y3_out = self.moe_layer(x)       # 输入输出都是 2*self.c
#         return identity + y3_out         # 残差相加，通道匹配
# DMS-MoE

class DMSMoE(nn.Module):
    def __init__(self, c1, c2, n=1, p=1, kernel_size=3, e=0.5):
        super().__init__()
        self.c = int(c2 * e)

        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)

        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            DWConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1)
        )

        self.dynamic_branch = MoEScaleBranch(2 * self.c, n=n)

        self.cv_final = Conv(4 * self.c, c2, 1)

    def forward(self, x):
        y = self.cv_first(x)

        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)

        # 这里的 y_dyn 包含了原来的 y2 和 y3_out
        y_dyn = self.dynamic_branch(y)

        # 最终拼接：y0(c) + y1(c) + y_dyn(2c) = 4c
        out = torch.cat([y0, y1, y_dyn], 1)
        return self.cv_final(out)



