import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from mmseg.models.losses import DiceLoss, CrossEntropyLoss
from .mmdet_cross_entropy import CrossEntropyLoss as CrossEntropyLoss2
from .mmdet_dice import DiceLoss as DiceLoss2
from mmseg.models.losses import FocalLoss


def new_loss(target_list, pre_list, criterion, sample_wise):
    loss = 0
    for i in range(len(target_list)):
        target = target_list[i]
        target = target.long()
        mask = (target == 0) | (target == 1)
        target[~mask] = 2
        pre = pre_list[i].unsqueeze(0)
        weight = torch.ones_like(target).float().cuda()
        weight_ = torch.tensor([[1, 1]]).float().cuda()
        if sample_wise:
            safe_mask = target == 0
            unsafe_mask = target == 1
            safe_num = torch.sum(safe_mask)
            unsafe_num = torch.sum(unsafe_mask)
            if safe_num != 0 and unsafe_num != 0:
                if safe_num > unsafe_num:
                    weight[unsafe_mask] = safe_num / unsafe_num
                    weight_[0][1] = safe_num / unsafe_num
                else:
                    weight[safe_mask] = unsafe_num / safe_num
                    weight_[0][0] = unsafe_num / safe_num
        if criterion.__class__.__name__ != "CrossEntropyLoss":
            weight = weight_
        loss += criterion(pre, target, ignore_index=2, weight=weight)
    return loss


def process_tensor(input_tensor, pad, scale):
    """
    处理四维张量，对每个样本进行裁剪和插值操作。

    参数:
        input_tensor (torch.Tensor): 输入张量，形状为 (B, C, H, W)
        pad (torch.Tensor): 裁剪参数，形状为 (B, 4)，每行对应一个样本的 [pad_top, pad_bottom, pad_left, pad_right]
        scale (torch.Tensor): 缩放因子，形状为 (B,)，每个元素对应一个样本的缩放比例

    返回:
        torch.Tensor: 处理后的张量，形状为 (B, C, H_scaled, W_scaled)
    """
    B, C, H, W = input_tensor.shape
    processed = []

    for i in range(B):
        # 获取当前样本的裁剪参数
        pad_top, pad_bottom, pad_left, pad_right = pad[i]

        # 计算裁剪范围
        h_start = pad_top
        h_end = H - pad_bottom
        w_start = pad_left
        w_end = W - pad_right

        # 裁剪当前样本
        cropped = input_tensor[i, :, h_start:h_end, w_start:w_end]

        # 计算目标尺寸
        h_crop = cropped.shape[1]
        w_crop = cropped.shape[2]
        target_h = int(h_crop * scale[i].item())
        target_w = int(w_crop * scale[i].item())

        # 使用双线性插值调整尺寸
        scaled = F.interpolate(
            cropped.unsqueeze(0),  # 增加 batch 维度
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # 移除 batch 维度

        processed.append(scaled)

    # 堆叠所有样本
    return processed


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SafeConfLoss(nn.Module):
    def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'denselidar_syn'], loss_gamma=0.9, **kwargs):
        super(SafeConfLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.criterion_conf = nn.BCEWithLogitsLoss()
        self.loss_gamma = loss_gamma
        self.conf_thresh = 0.8

    def forward(self, **kwargs):
        target = kwargs['safe_target'].squeeze(1)
        predictions_list = kwargs['safe_predictions_list']
        mask = (target == 0) | (target == 1)
        mask = mask.squeeze(1)
        confidence_target = target.float()

        n_predictions = len(predictions_list)
        assert n_predictions >= 1
        loss = 0.0

        for i, prediction in enumerate(predictions_list):
            adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

            # probs = torch.softmax(prediction, dim=1)  # [batch_size, 2, height, width]
            # conf = probs[:, 1, :, :]
            # conf = torch.sigmoid(conf)
            # conf_loss = self.criterion_conf(prediction[:, 1, :, :][mask], confidence_target[mask])
            # loss += i_weight * conf_loss
            safe_pred = F.softmax(prediction, dim=1)
            mask1 = safe_pred[:, 1, :, :] < self.conf_thresh
            safe_pred = torch.argmax(safe_pred, dim=1)
            mask2 = safe_pred == 1
            loss_mask = mask & mask1 & mask2
            valid_mask = mask & mask2
            loss += i_weight * torch.sum(loss_mask) / torch.sum(valid_mask)

        # conf_loss = F.binary_cross_entropy_with_logits(confidence_pre[mask], confidence_target[mask])
        return loss * self.loss_weight


class SafeEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'denselidar_syn'], loss_gamma=0.9,
                 use_conf=True, conf_thresh=0.8, class_weight=[1.0, 1.0], **kwargs):
        super(SafeEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.loss_gamma = loss_gamma
        self.use_conf = use_conf
        self.conf_thresh = conf_thresh
        # self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.criterion = CrossEntropyLoss(use_sigmoid=False, class_weight=class_weight)
        # self.criterion2 = CrossEntropyLoss(use_sigmoid=False, class_weight=[1.0,2.0])
        # self.criterion = nn.CrossEntropyLoss()
        self.class_weight = class_weight

    def forward(self, **kwargs):
        predictions_list = kwargs['safe_predictions_list']
        target = kwargs['safe_target'].squeeze(1)
        target = target.long()
        target[target == 255] = 1
        # target = kwargs['safe_target']
        # target = torch.cat([1 - target, target], dim=1)
        # mask = (target == 0) | (target == 1)
        # target[~mask] = 0
        # target = target[mask]

        n_predictions = len(predictions_list)
        assert n_predictions >= 1
        loss = 0.0

        for i, prediction in enumerate(predictions_list):
            adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

            # valid_pre = prediction.permute(0, 2, 3, 1)[mask]
            curr_loss = 0
            if not self.use_conf:
                # prediction = prediction[mask]
                # target = target[mask]
                # target = target.squeeze(1).long()
                # print(torch.unique(target))
                # weight_tensor = torch.ones_like(prediction)
                # weight_tensor[:, 0, :, :] *= self.class_weight[0]  # 通道0
                # weight_tensor[:, 1, :, :] *= self.class_weight[1]  # 通道1
                curr_loss = self.criterion(prediction, target, ignore_index=255)
                # loss2 = self.criterion2(prediction, target, ignore_index=255)
                # a = loss2
            else:
                curr_loss = self.criterion(prediction, target)
                probs = torch.softmax(prediction, dim=1)  # [batch_size, 2, height, width]
                conf = probs[:, 1, :, :]  # 取类别 1 的概率作为置信度

                # 根据置信度调整预测
                adjusted_probs = probs.clone()  # 复制原始概率
                adjusted_probs[:, 1, :, :][conf < self.conf_thresh] = 0  # 将置信度低于阈值的类别 1 预测置为 0
                adjusted_probs[:, 0, :, :][conf < self.conf_thresh] = 1
                curr_loss = self.criterion(torch.log(adjusted_probs + 1e-10), target)
            # if torch.isnan(curr_loss).item() | torch.isinf(curr_loss).item():
            #     print(f'GRUSequenceLoss-depth NAN error, {curr_loss}')
            #     curr_loss = 0 * torch.sum(prediction)

            # confidence L1 loss

            # if confidence_list is not None:
            #     conf_loss = self.conf_loss(confidence_list[i], prediction, target, mask)

            loss += curr_loss * i_weight
        # conf_loss = F.binary_cross_entropy_with_logits(confidence_pre[mask], confidence_target[mask])
        # loss += conf_loss
        return loss * self.loss_weight


class SemiSafeEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'denselidar_syn'], loss_gamma=0.9,
                 use_conf=True, conf_thresh=0.8, class_weight=[1.0, 1.0], punish_factor=1, final_only=False,
                 loss_raw_size=False, sample_wise=False, **kwargs):
        super(SemiSafeEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.loss_gamma = loss_gamma
        self.use_conf = use_conf
        self.conf_thresh = conf_thresh
        # self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.criterion = CrossEntropyLoss2(use_sigmoid=False, class_weight=class_weight)
        # self.criterion2 = CrossEntropyLoss(use_sigmoid=False, class_weight=[1.0,2.0])
        # self.criterion = nn.CrossEntropyLoss()
        self.class_weight = class_weight
        self.punish_factor = punish_factor
        self.final_only = final_only
        self.loss_raw_size = loss_raw_size
        self.sample_wise = sample_wise
        self.pos_class = 0
        self.neg_class = 1

    def forward(self, **kwargs):
        if self.final_only:
            predictions_list = [kwargs['safe_prediction']]
        else:
            predictions_list = kwargs['safe_predictions_list']
        if self.loss_raw_size:
            target_list = process_tensor(kwargs['safe_target'], kwargs['pad'], kwargs['scale'])
            sum_loss = 0
            n_predictions = len(predictions_list)
            for i, pre in enumerate(predictions_list):
                adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))
                i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
                pre_list = process_tensor(pre, kwargs['pad'], kwargs['scale'])
                raw_loss = new_loss(target_list, pre_list, self.criterion, self.sample_wise)
                sum_loss += raw_loss * i_weight
            return sum_loss
            # print(raw_loss)
        target = kwargs['safe_target'].squeeze(1)
        target = target.long()
        # target[target==255] = 1
        # target = kwargs['safe_target']
        # target = torch.cat([1 - target, target], dim=1)
        mask = (target == 0) | (target == 1)
        target[~mask] = 2
        # target.float()
        # target = target[mask]

        n_predictions = len(predictions_list)
        assert n_predictions >= 1
        loss = 0.0
        if n_predictions > 1:
            for i, prediction in enumerate(predictions_list):
                adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))
                i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

                # valid_pre = prediction.permute(0, 2, 3, 1)[mask]
                # curr_loss = 0
                punish_weight = None
                if self.punish_factor > 1:
                    pre_logits = torch.softmax(prediction, dim=1)
                    pred = torch.argmax(pre_logits, dim=1)
                    punish_mask = (pred == 0) & (target == 1)
                    punish_weight = torch.ones_like(target)
                    punish_weight[punish_mask] = self.punish_factor
                weight = torch.ones_like(target).float().cuda()
                if self.sample_wise:
                    safe_mask = target == self.pos_class
                    unsafe_mask = target == self.neg_class
                    safe_num = torch.sum(safe_mask)
                    unsafe_num = torch.sum(unsafe_mask)
                    if safe_num != 0 and unsafe_num != 0:
                        if safe_num > unsafe_num:
                            weight[unsafe_mask] = safe_num / unsafe_num
                        else:
                            weight[safe_mask] = unsafe_num / safe_num
                curr_loss = self.criterion(prediction, target, ignore_index=2, punish_weight=punish_weight,
                                           weight=weight)

                loss += curr_loss * i_weight
        else:
            prediction = predictions_list[0]
            punish_weight = None
            if self.punish_factor > 1:
                # 同样，对于 argmax，不需要先做 softmax，直接 argmax 即可
                pred = torch.argmax(prediction, dim=1)
                punish_mask = (pred == 0) & (target == 1)
                
                # 【修复】：必须生成和 target 形状 [B, H, W] 一样的权重矩阵
                punish_weight = torch.ones_like(target).float().cuda()
                punish_weight[punish_mask] = self.punish_factor
                
            loss = self.criterion(prediction, target, ignore_index=2, punish_weight=punish_weight)
        return loss * self.loss_weight


class SemiSafeEntropyLoss2(nn.Module):
    def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'denselidar_syn'], loss_gamma=0.9,
                 use_conf=True, conf_thresh=0.8, class_weight=[1.0, 1.0], **kwargs):
        super(SemiSafeEntropyLoss2, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.loss_gamma = loss_gamma
        self.use_conf = use_conf
        self.conf_thresh = conf_thresh
        # self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.criterion = CrossEntropyLoss2(use_sigmoid=False, class_weight=class_weight)
        # self.criterion2 = CrossEntropyLoss(use_sigmoid=False, class_weight=[1.0,2.0])
        # self.criterion = nn.CrossEntropyLoss()
        self.class_weight = class_weight

    def forward(self, **kwargs):
        predictions_list = kwargs['safe_predictions_list']
        target = kwargs['safe_target'].squeeze(1)
        target = target.long()
        # target[target==255] = 1
        # target = kwargs['safe_target']
        # target = torch.cat([1 - target, target], dim=1)
        mask = (target == 0) | (target == 1)
        target[~mask] = 2
        # target.float()
        # target = target[mask]
        loss = 0.0

        for i, prediction in enumerate(predictions_list):
            # valid_pre = prediction.permute(0, 2, 3, 1)[mask]
            # curr_loss = 0
            curr_loss = self.criterion(prediction, target, ignore_index=2)

            loss += curr_loss
        return loss * self.loss_weight


class SafeBCELoss(nn.Module):
    def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'denselidar_syn'], loss_gamma=0.9,
                 conf_thresh=0.7, use_conf=False, conf_loss_weight=2, **kwargs):
        super(SafeBCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.loss_gamma = loss_gamma
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_3 = nn.BCELoss()
        self.criterion_2 = F.binary_cross_entropy_with_logits
        self.conf_thresh = conf_thresh
        self.use_conf = use_conf
        self.conf_loss_weight = conf_loss_weight

    def forward(self, **kwargs):
        predictions_list = kwargs['safe_predictions_list']
        target = kwargs['safe_target']
        mask = (target == 0) | (target == 1)

        target = target[mask]

        n_predictions = len(predictions_list)
        assert n_predictions >= 1
        loss = 0.0

        for i, prediction in enumerate(predictions_list):
            adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            prediction = prediction[mask]
            if not self.use_conf:
                curr_loss = self.criterion(prediction, target)
                # probs = torch.sigmoid(prediction)
                # pred = (probs > self.conf_thresh).float()
                # curr_loss = self.criterion_3(pred, target)
            else:
                curr_loss = self.criterion_2(prediction, target, reduction='none')
                prob = torch.sigmoid(prediction)
                weight_mask = (prob < self.conf_thresh) & (target == 1)
                curr_loss[weight_mask] *= self.conf_loss_weight
                curr_loss = torch.mean(curr_loss)
            loss += curr_loss * i_weight
        return loss * self.loss_weight


class SafeFocalLoss(nn.Module):
    def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'denselidar_syn'], loss_gamma=0.9,
                 final_only=False, loss_raw_size=False, sample_wise=False, class_weight=[1.0, 1.0], **kwargs):
        super(SafeFocalLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.loss_gamma = loss_gamma
        self.criterion = FocalLoss(class_weight=class_weight)
        self.final_only = final_only
        self.loss_raw_size = loss_raw_size
        self.sample_wise = sample_wise

    def forward(self, **kwargs):
        if self.final_only:
            predictions_list = [kwargs['safe_prediction']]
        else:
            predictions_list = kwargs['safe_predictions_list']
        if self.loss_raw_size:
            target_list = process_tensor(kwargs['safe_target'], kwargs['pad'], kwargs['scale'])
            sum_loss = 0
            n_predictions = len(predictions_list)
            for i, pre in enumerate(predictions_list):
                adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))
                i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
                pre_list = process_tensor(pre, kwargs['pad'], kwargs['scale'])
                raw_loss = new_loss(target_list, pre_list, self.criterion, self.sample_wise)
                sum_loss += raw_loss * i_weight
            return sum_loss
        target = kwargs['safe_target']  # Ensure target is in the correct shape and type
        mask = (target == 0) | (target == 1)
        target[~mask] = 2
        target = target.squeeze(1).long()
        # mask = (target == 0) | (target == 1)
        # target = target[mask]

        n_predictions = len(predictions_list)
        assert n_predictions >= 1, "At least one prediction is required."
        total_loss = 0.0

        if n_predictions > 1:
            for i, prediction in enumerate(predictions_list):
                adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))
                i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

                # prediction = torch.softmax(prediction, dim=1)
                # prediction = torch.argmax(prediction, dim=1)
                weight = torch.tensor([[1, 1]]).float().cuda()
                if self.sample_wise:
                    safe_mask = target == 0
                    unsafe_mask = target == 1
                    safe_num = torch.sum(safe_mask)
                    unsafe_num = torch.sum(unsafe_mask)
                    if safe_num != 0 and unsafe_num != 0:
                        if safe_num > unsafe_num:
                            weight[0][1] = safe_num / unsafe_num
                        else:
                            weight[0][0] = unsafe_num / safe_num

                loss = self.criterion(prediction, target, ignore_index=2, weight=weight)
                total_loss += loss * i_weight
        else:
            prediction = predictions_list[0]
            # prediction = torch.softmax(prediction, dim=1)
            # prediction = torch.argmax(prediction, dim=1)

            total_loss = self.criterion(prediction, target, ignore_index=2)

        return total_loss * self.loss_weight


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class SafeStandardDiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0, loss_gamma=0.9, data_type=['lidar', 'denselidar', 'denselidar_syn'], eps=1e-6,
                 reduction='mean', ignore_index=2, naive_dice=False, activate=False, class_weight=[1.0, 0.0], **kwargs):
        """
        Standard Dice Loss implementation for Safe Landing Zone.

        Args:
            loss_weight (float): Weight of the loss. Default: 1.0.
            eps (float): A small value to avoid division by zero. Default: 1e-6.
            reduction (str): Reduction method for the loss. Options: 'none', 'mean', 'sum'. Default: 'mean'.
            ignore_index (int): Label index to ignore. Default: 2 (or 255 depending on your pipeline).
            class_weight (list): Weights for each class. [1.0, 0.0] means only optimize Dice for class 0 (safe).
        """
        super(SafeStandardDiceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_gamma = loss_gamma
        self.data_type = data_type
        self.naive_dice = naive_dice
        self.activate = activate
        self.class_weight = class_weight  # <--- [新增] 接收类权重参数
        
        # 将 class_weight 传递给底层的 DiceLoss2
        self.criterion = DiceLoss2(
            activate=self.activate, 
            naive_dice=self.naive_dice, 
            ignore_index=self.ignore_index,
            reduction=self.reduction, 
            eps=self.eps, 
            use_sigmoid=False
        )

    def forward(self, **kwargs):
        """
        Forward function to compute the standard Dice Loss.
        """
        predictions_list = kwargs['safe_predictions_list']
        target = kwargs['safe_target']
        
        # 处理标签，确保非 0/1 的区域都被设为 ignore_index(2)
        mask = (target == 0) | (target == 1)
        target[~mask] = self.ignore_index  # 动态使用 self.ignore_index
        
        # 安全地去掉可能多余的 channel 维度 (B, 1, H, W) -> (B, H, W)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        target = target.long()

        n_predictions = len(predictions_list)
        assert n_predictions >= 1, "At least one prediction is required."

        total_loss = 0.0

        if n_predictions > 1:
            for i, prediction in enumerate(predictions_list):
                # RAFT 架构经典的多级递进权重衰减
                adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))
                i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
                
                # 【重要】：保留 Softmax 得到软概率，绝不能用 argmax
                prediction = torch.softmax(prediction, dim=1) 

                # 计算当前层的 Dice Loss
                loss = self.criterion(prediction, target)
                total_loss += loss * i_weight
        else:
            prediction = predictions_list[0]
            
            # 【重要】：保留 Softmax 得到软概率，绝不能用 argmax
            prediction = torch.softmax(prediction, dim=1)
            
            total_loss = self.criterion(prediction, target)

        return total_loss * self.loss_weight