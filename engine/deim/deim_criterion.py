"""     
DEIM: DETR with Improved Matching for Fast Convergence 
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------   
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.    
"""

import torch
import torch.nn as nn  
import torch.distributed
import torch.nn.functional as F    
import torchvision    
   
import copy
  
from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou    
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized     
from ..core import register

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    

@register()
class DEIMCriterion(nn.Module):   
    """ This class computes the loss for DEIM.   
    """   
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,    
        weight_dict,
        losses,
        alpha=0.2,    
        gamma=2.0,  
        num_classes=80,     
        reg_max=32, 
        boxes_weight_format=None,   
        share_matched_indices=False,   
        mal_alpha=None,
        use_uni_set=True,
        no_weight_vfl_epoch=-1
        ):   
        """Create the criterion.
        Parameters: 
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses. 
            num_classes: number of object categories, omitting the special no-object category.   
            reg_max (int): Max number of the discrete bins in D-FINE.    
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()  
        self.num_classes = num_classes    
        self.matcher = matcher     
        self.weight_dict = weight_dict   
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha   
        self.use_uni_set = use_uni_set
        self.no_weight_vfl_epoch = no_weight_vfl_epoch     
        self.epoch = 0    
    
        if self.no_weight_vfl_epoch != -1: 
            print(RED + f"no_weight_vfl_epoch set {self.no_weight_vfl_epoch}" + RESET)

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs    
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices) 
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o  
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]     
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes     
 
        return {'loss_focal': loss}  
     
    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):    
        assert 'pred_boxes' in outputs 
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))  
            ious = torch.diag(ious).detach()  
        else:
            ious = values    

        src_logits = outputs['pred_logits'] 
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])     
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,   
                                    dtype=torch.int64, device=src_logits.device)    
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1] 
 
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype) 
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target 

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score  
 
        if self.no_weight_vfl_epoch == -1 or self.epoch >= self.no_weight_vfl_epoch:
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        else:
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, reduction='none')     
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}  

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):    
        assert 'pred_boxes' in outputs   
        idx = self._get_src_permutation_idx(indices)
        if values is None: 
            src_boxes = outputs['pred_boxes'][idx]  
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)     
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()    
        else: 
            ious = values    

        src_logits = outputs['pred_logits']    
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])   
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,   
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]    
 
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)     
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
   
        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)  
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target   
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))  
        if self.no_weight_vfl_epoch == -1 or self.epoch >= self.no_weight_vfl_epoch:  
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        else:
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, reduction='none') 
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes    
        return {'loss_mal': loss}
  
    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    # def get_wiou_v3_loss(pred_boxes, target_boxes, alpha=1.9, delta=3):
    #     """
    #     Wise-IoU v3 Loss Implementation
    #     参数:
    #         pred_boxes: 预测框 [N, 4], 格式 (cx, cy, w, h)
    #         target_boxes: 真实框 [N, 4], 格式 (cx, cy, w, h)
    #         alpha: 超参数，控制梯度的增益 (默认 1.9)
    #         delta: 超参数，控制离群度的阈值 (默认 3)
    #     """
    #     # 1. 基础数据准备
    #     # 将 (cx, cy, w, h) 转换为 (x1, y1, x2, y2) 用于计算 IoU
    #     pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    #     pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    #     pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    #     pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    #
    #     target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    #     target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    #     target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    #     target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
    #
    #     # 2. 计算基本 IoU (Inter / Union)
    #     # 交集
    #     inter_x1 = torch.max(pred_x1, target_x1)
    #     inter_y1 = torch.max(pred_y1, target_y1)
    #     inter_x2 = torch.min(pred_x2, target_x2)
    #     inter_y2 = torch.min(pred_y2, target_y2)
    #
    #     inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    #
    #     # 并集
    #     pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    #     target_area = target_boxes[:, 2] * target_boxes[:, 3]
    #     union_area = pred_area + target_area - inter_area + 1e-7  # 防止除0
    #
    #     iou = inter_area / union_area
    #     loss_iou = 1 - iou
    #
    #     # 3. 计算距离注意力 (Distance Attention) - WIoU v1 部分
    #     # 最小外接矩形的宽和高 (Wg, Hg)
    #     enclose_x1 = torch.min(pred_x1, target_x1)
    #     enclose_y1 = torch.min(pred_y1, target_y1)
    #     enclose_x2 = torch.max(pred_x2, target_x2)
    #     enclose_y2 = torch.max(pred_y2, target_y2)
    #
    #     wg = enclose_x2 - enclose_x1
    #     hg = enclose_y2 - enclose_y1
    #
    #     # 中心点距离平方
    #     # pred_boxes[:, 0] 是 cx, pred_boxes[:, 1] 是 cy
    #     center_dist_sq = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
    #                      (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2
    #
    #     # 最小外接矩形对角线距离平方
    #     diag_dist_sq = wg ** 2 + hg ** 2 + 1e-7
    #
    #     # R_WIoU
    #     # 注意：这里 detach() 很重要，防止距离惩罚过度影响中心点梯度的方向
    #     dist_attn = torch.exp(center_dist_sq / diag_dist_sq)
    #
    #     loss_wiou_v1 = dist_attn * loss_iou
    #
    #     # 4. 计算非单调聚焦系数 (Non-monotonic Focusing) - WIoU v3 部分
    #     # beta: 离群度 (Outlier Degree)
    #     # 使用 .detach() 是因为我们只用它来调整权重，不希望反向传播这个统计量
    #     loss_iou_detach = loss_iou.detach()
    #     beta = loss_iou_detach / loss_iou_detach.mean()
    #
    #     # r: 梯度增益系数
    #     # 小目标/普通样本 r 高，极大离群值 r 低
    #     # 加上 1e-7 防止 log(0) 或其他数值不稳定
    #     r = beta / (delta * torch.pow(alpha, beta - delta))
    #
    #     # 最终 Loss
    #     loss_wiou_v3 = r * loss_wiou_v1
    #
    #     return loss_wiou_v3

    # def get_nwd_wiou_loss(self,pred_boxes, target_boxes, alpha=1.9, delta=3):
    #     # 1. 计算基础 IoU Loss
    #     # ... (省略基础 IoU 计算代码，同前) ...
    #     pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    #     pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    #     pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    #     pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    #
    #     target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    #     target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    #     target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    #     target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
    #
    #     # 2. 计算基本 IoU (Inter / Union)
    #     # 交集
    #     inter_x1 = torch.max(pred_x1, target_x1)
    #     inter_y1 = torch.max(pred_y1, target_y1)
    #     inter_x2 = torch.min(pred_x2, target_x2)
    #     inter_y2 = torch.min(pred_y2, target_y2)
    #
    #     inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    #
    #     # 并集
    #     pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    #     target_area = target_boxes[:, 2] * target_boxes[:, 3]
    #     union_area = pred_area + target_area - inter_area + 1e-7  # 防止除0
    #
    #     iou = inter_area / union_area
    #     loss_iou = 1 - iou
    #
    #     # 2. 计算 NWD (归一化巴氏距离)
    #     # 将框视为高斯分布
    #     b1_x, b1_y, b1_w, b1_h = pred_boxes.unbind(-1)
    #     b2_x, b2_y, b2_w, b2_h = target_boxes.unbind(-1)
    #
    #     # 距离项 (中心点距离 + 宽高差异)
    #     wasserstein_2 = (b1_x - b2_x) ** 2 + (b1_y - b2_y) ** 2 + \
    #                     ((b1_w - b2_w) ** 2 + (b1_h - b2_h) ** 2) / 4
    #
    #     # NWD 值 (0~1之间，越接近1越好)
    #     constant = 12.8
    #     nwd = torch.exp(-torch.sqrt(wasserstein_2.clamp(min=1e-7)) / constant)
    #
    #     # 3. 创新点：用 (1 - NWD) 替代 WIoU 原本的距离注意力 R_WIoU
    #     # 原版 WIoU 是 exp(dist / diag)，这里我们要一个类似的惩罚项
    #     # NWD 越小(不重叠)，Penalty 越大
    #     # dist_penalty = 1 - nwd
    #     # 推荐写法：基于 WIoU 思想的放大惩罚
    #     # 当 NWD=1 时，penalty=1
    #     # 当 NWD=0 时，penalty=e ≈ 2.718
    #     dist_penalty = torch.exp(1 - nwd)
    #
    #     # 4. 计算聚焦系数 beta (离群度)
    #     loss_iou_detach = loss_iou.detach()
    #     beta = loss_iou_detach / loss_iou_detach.mean()
    #
    #     # 5. 计算梯度增益 r
    #     r = beta / (delta * torch.pow(alpha, beta - delta))
    #
    #     # 6. 最终 Loss
    #     # 结合了 WIoU 的聚焦机制 和 NWD 的距离度量
    #     loss = r * dist_penalty * loss_iou
    #
    #     return loss

    # def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
    #     """Compute the losses related to the bounding boxes, the L1 regression loss and the WIoU v3 loss"""
    #     assert 'pred_boxes' in outputs
    #     idx = self._get_src_permutation_idx(indices)
    #
    #     # 获取预测框和真实框
    #     # src_boxes 格式为 (cx, cy, w, h)，且归一化过 (0-1)
    #     src_boxes = outputs['pred_boxes'][idx]
    #     target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    #
    #     losses = {}
    #
    #     # 1. L1 Loss (保持不变，用于基础回归稳定性)
    #     loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
    #     losses['loss_bbox'] = loss_bbox.sum() / num_boxes
    #
    #     # 2. 计算 WIoU v3 Loss 替换原有的 GIoU
    #     # 注意：WIoU 需要真实的像素坐标计算距离注意力可能更准确，
    #     # 但在归一化坐标系下计算也是完全合法的 (相当于 W,H 都除以了图像尺寸)
    #     # 这里直接使用归一化的 src_boxes 和 target_boxes 即可
    #
    #     # 调用上面定义的函数
    #     # alpha=1.9, delta=3 是原论文在 COCO 上的推荐参数
    #     # 针对你的小目标任务，如果发现收敛慢，可以尝试减小 delta (如 2.5)
    #     wiou_loss_tensor = self.get_nwd_wiou_loss(src_boxes, target_boxes, alpha=1.9, delta=3)
    #
    #     if boxes_weight is not None:
    #         wiou_loss_tensor = wiou_loss_tensor * boxes_weight
    #
    #     # 使用 'loss_giou' 作为键名是为了兼容现有的 weight_dict 配置，
    #     # 这样你不需要去改配置文件里的 key
    #     losses['loss_giou'] = wiou_loss_tensor.sum() / num_boxes
    #
    #     return losses


    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss  
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:   
            idx = self._get_src_permutation_idx(indices)     
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)   
 
            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))   
            ref_points = outputs['ref_points'][idx].detach()  
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:    
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes), 
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets
  
            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])   
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)   

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))     
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)  
     
                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach() 

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:    
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU   
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5 
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0   
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0 
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)   

        return losses
     
    def _get_src_permutation_idx(self, indices):     
        # permute predictions following indices 
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
   
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]) 
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])   
        return batch_idx, tgt_idx 
    
    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """    
        results = []    
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))   
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:    
            unique, counts = torch.unique(ind, return_counts=True, dim=0) 
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}   
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()    
                if row_idx not in column_to_row:    
                    column_to_row[row_idx] = col_idx 
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))  
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None    
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None    

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {     
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,    
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,    
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs) 
   
    def forward(self, outputs, targets, **kwargs):    
        """ This performs the loss computation.
        Parameters:   
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """ 
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets   
        indices = self.matcher(outputs_without_aux, targets)['indices']     
        self._clear_cache()
    
        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:  
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']    
            if 'pre_outputs' in outputs:  
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)   
                indices_aux_list.append(indices_aux) 
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):   
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)
     
            num_boxes_go = sum(len(x[0]) for x in indices_go)   
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device) 
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:    
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)   
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes) 
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
    
        # Compute all the requested losses, main loss
        losses = {}  
        for loss in self.losses:  
            # TODO, indices and num_box are different from RT-DETRv2    
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])     
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes  
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)     
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict} 
            losses.update(l_dict)
   
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:   
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']    
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]    
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes  
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)     
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
   
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']  
            for loss in self.losses:   
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)  
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}    
                l_dict = {k + '_pre': v for k, v in l_dict.items()}     
                losses.update(l_dict)    
  
        # In case of encoder auxiliary losses.   
        if 'enc_aux_outputs' in outputs:    
            assert 'enc_meta' in outputs, '' 
            class_agnostic = outputs['enc_meta']['class_agnostic']  
            if class_agnostic:    
                orig_num_classes = self.num_classes    
                self.num_classes = 1  
                enc_targets = copy.deepcopy(targets) 
                for t in enc_targets:     
                    t['labels'] = torch.zeros_like(t["labels"]) 
            else:   
                enc_targets = targets
  
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:    
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes')   
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)    
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)  
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:   
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.     
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)     
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
    
            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss 
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']   
                for loss in self.losses:     
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}  
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
 
            # In case of auxiliary traditional head output at first decoder layer, just for dfine  
            if 'dn_pre_outputs' in outputs:  
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:     
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)    
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}   
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)
   
        # For debugging Objects365 pre-train. 
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}    
        return losses
   
    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None: 
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]   
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)     

        if self.boxes_weight_format == 'iou':    
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)) 
            iou = torch.diag(iou)  
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))  
        else:
            raise AttributeError()

        if loss in ('boxes', ):   
            meta = {'boxes_weight': iou}     
        elif loss in ('vfl', 'mal'):  
            meta = {'values': iou}
        else:
            meta = {}   
     
        return meta
 
    @staticmethod    
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices   
        """  
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]   
        num_gts = [len(t['labels']) for t in targets]    
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device) 
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)    
                dn_match_indices.append((dn_positive_idx[i], gt_idx))    
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))    
     
        return dn_match_indices
    

    def feature_loss_function(self, fea, target_fea):     
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()   
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):     
        dis_left = label.long() 
        dis_right = dis_left + 1
    
        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)
 
        if weight is not None:     
            weight = weight.float() 
            loss = loss * weight
     
        if avg_factor is not None:    
            loss = loss.sum() / avg_factor 
        elif reduction == 'mean':    
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()  
   
        return loss
    
    def get_gradual_steps(self, outputs):   
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1    
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]  
        return opt_list 
  
    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.no_weight_vfl_epoch != -1 and self.epoch == self.no_weight_vfl_epoch:     
            print(RED + f"Epoch:[{self.epoch}]>=[{self.no_weight_vfl_epoch}] using weight in vfl/mal." + RESET)