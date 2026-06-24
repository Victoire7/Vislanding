import copy
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.nn.init as init

# =====================================================================
# 基础工具与辅助函数 (保持原样)
# =====================================================================
def compute_depth_expectation(prob, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(prob * depth_values, 1)
    return depth

def interpolate_float32(x, size=None, scale_factor=None, mode='nearest', align_corners=None):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        return F.interpolate(x.float(), size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

def upflow4(flow, mode='bilinear'):
    new_size = (4 * flow.shape[2], 4 * flow.shape[3])
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        return F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def coords_grid(batch, ht, wd):
    coords = (
        torch.zeros((ht, wd)), torch.zeros((ht, wd)), torch.zeros((ht, wd)), torch.zeros((ht, wd)),
        torch.zeros((ht, wd)),
        torch.zeros((ht, wd)))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def coords_grid_safe(batch, ht, wd):
    coords = (torch.zeros((ht, wd)), torch.zeros((ht, wd)))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def norm_normalize(norm_out):
    min_kappa = 0.01
    norm_x, norm_y, norm_z, kappa = torch.split(norm_out, 1, dim=1)
    norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    kappa = F.elu(kappa) + 1.0 + min_kappa
    final_out = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
    return final_out

@torch.no_grad()
def sample_points(init_normal, gt_norm_mask, sampling_ratio, beta):
    device = init_normal.device
    B, _, H, W = init_normal.shape
    N = int(sampling_ratio * H * W)
    beta = beta

    # uncertainty map
    uncertainty_map = -1 * init_normal[:, -1, :, :]  # B, H, W

    # gt_invalid_mask (B, H, W)
    if gt_norm_mask is not None:
        gt_invalid_mask = F.interpolate(gt_norm_mask.float(), size=[H, W], mode='nearest')
        gt_invalid_mask = gt_invalid_mask[:, 0, :, :] < 0.5
        uncertainty_map[gt_invalid_mask] = -1e4

    # (B, H*W)
    _, idx = uncertainty_map.view(B, -1).sort(1, descending=True)

    # importance sampling
    if int(beta * N) > 0:
        importance = idx[:, :int(beta * N)]  # B, beta*N

        # remaining
        remaining = idx[:, int(beta * N):]  # B, H*W - beta*N

        # coverage
        num_coverage = N - int(beta * N)

        if num_coverage <= 0:
            samples = importance
        else:
            coverage_list = []
            for i in range(B):
                idx_c = torch.randperm(remaining.size()[1])  # shuffles "H*W - beta*N"
                coverage_list.append(remaining[i, :][idx_c[:num_coverage]].view(1, -1))  # 1, N-beta*N
            coverage = torch.cat(coverage_list, dim=0)  # B, N-beta*N
            samples = torch.cat((importance, coverage), dim=1)  # B, N

    else:
        # remaining
        remaining = idx[:, :]  # B, H*W

        # coverage
        num_coverage = N

        coverage_list = []
        for i in range(B):
            idx_c = torch.randperm(remaining.size()[1])  # shuffles "H*W - beta*N"
            coverage_list.append(remaining[i, :][idx_c[:num_coverage]].view(1, -1))  # 1, N-beta*N
        coverage = torch.cat(coverage_list, dim=0)  # B, N-beta*N
        samples = coverage

    # point coordinates
    rows_int = samples // W  # 0 for first row, H-1 for last row
    rows_float = rows_int / float(H - 1)  # 0 to 1.0
    rows_float = (rows_float * 2.0) - 1.0  # -1.0 to 1.0

    cols_int = samples % W  # 0 for first column, W-1 for last column
    cols_float = cols_int / float(W - 1)  # 0 to 1.0
    cols_float = (cols_float * 2.0) - 1.0  # -1.0 to 1.0

    point_coords = torch.zeros(B, 1, N, 2)
    point_coords[:, 0, :, 0] = cols_float  # x coord
    point_coords[:, 0, :, 1] = rows_float  # y coord
    point_coords = point_coords.to(device)
    return point_coords, rows_int, cols_int

# =====================================================================
# 基础网络组件 (保持原样)
# =====================================================================
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim_depth=2, output_dim_norm=4):
        super(FlowHead, self).__init__()
        self.conv1d = nn.Conv2d(input_dim, hidden_dim // 2, 3, padding=1)
        self.conv2d = nn.Conv2d(hidden_dim // 2, output_dim_depth, 3, padding=1)

        self.conv1n = nn.Conv2d(input_dim, hidden_dim // 2, 3, padding=1)
        self.conv2n = nn.Conv2d(hidden_dim // 2, output_dim_norm, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        depth = self.conv2d(self.relu(self.conv1d(x)))
        normal = self.conv2n(self.relu(self.conv1n(x)))
        return torch.cat((depth, normal), dim=1)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid((self.convz(hx) + cz))
        r = torch.sigmoid((self.convr(hx) + cr))
        q = torch.tanh((self.convq(torch.cat([r * h, x], dim=1)) + cq))
        h = (1 - z) * h + z * q
        return h

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return interpolate_float32(x, dest.shape[2:], **interp_args)

class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[], out_dims=2):
        super().__init__()
        self.args = args
        self.n_gru_layers = args.model.decode_head.n_gru_layers  
        self.n_downsample = args.model.decode_head.n_downsample  
        encoder_output_dim = 6  
        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (self.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (self.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=2 * hidden_dims[2])
        factor = 2 ** self.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], hidden_dims[2], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[2], (factor ** 2) * 9, 1, padding=0))

    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):
        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), interp(pool2x(net[0]), net[1]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), interp(pool2x(net[0]), net[1]))
        if iter08:
            if corr is not None:
                motion_features = self.encoder(flow, corr)
            else:
                motion_features = flow
            if self.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])
        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super(LayerNorm2d, self).__init__(dim)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super(LayerNorm2d, self).forward(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'layer':
            self.norm1 = LayerNorm2d(planes)
            self.norm2 = LayerNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = LayerNorm2d(planes)
        else:
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)

class ContextFeatureEncoder(nn.Module):
    def __init__(self, in_dim, output_dim):
        super().__init__()
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(in_dim[0], dim[0], 'layer', stride=1),
                nn.Conv2d(dim[0], dim[0], 3, padding=1))
            output_list.append(conv_out)
        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(in_dim[1], dim[1], 'layer', stride=1),
                nn.Conv2d(dim[1], dim[1], 3, padding=1))
            output_list.append(conv_out)
        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(in_dim[2], dim[2], 'layer', stride=1),
                nn.Conv2d(dim[2], dim[2], 3, padding=1))
            output_list.append(conv_out)
        self.outputs16 = nn.ModuleList(output_list)

    def forward(self, encoder_features):
        x_4, x_8, x_16, x_32 = encoder_features
        outputs04 = [f(x_4) for f in self.outputs04]
        outputs08 = [f(x_8) for f in self.outputs08]
        outputs16 = [f(x_16) for f in self.outputs16]
        return (outputs04, outputs08, outputs16)

class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        out = self.act(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return x + out

class FuseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fuse=True, upsample=True, scale_factor=2):
        super(FuseBlock, self).__init__()
        self.fuse = fuse
        self.scale_factor = scale_factor
        self.way_trunk = ConvBlock(in_channels)
        if self.fuse:
            self.way_branch = ConvBlock(in_channels)
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.upsample = upsample

    def forward(self, x1, x2=None):
        if x2 is not None:
            x2 = self.way_branch(x2)
            x1 = x1 + x2
        out = self.way_trunk(x1)
        if self.upsample:
            out = interpolate_float32(out, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
        out = self.out_conv(out)
        return out

class Readout(nn.Module):
    def __init__(self, in_features, use_cls_token=True, num_register_tokens=0):
        super(Readout, self).__init__()
        self.use_cls_token = use_cls_token
        if self.use_cls_token == True:
            self.project_patch = nn.Linear(in_features, in_features)
            self.project_learn = nn.Linear((1 + num_register_tokens) * in_features, in_features, bias=False)
            self.act = nn.GELU()
        else:
            self.project = nn.Identity()

    def forward(self, x):
        if self.use_cls_token == True:
            x_patch = self.project_patch(x[0])
            x_learn = self.project_learn(x[1])
            x_learn = x_learn.expand_as(x_patch).contiguous()
            features = x_patch + x_learn
            return self.act(features)
        else:
            return self.project(x)

class Token2Feature(nn.Module):
    def __init__(self, vit_channel, feature_channel, scale_factor, use_cls_token=True, num_register_tokens=0):
        super(Token2Feature, self).__init__()
        self.scale_factor = scale_factor
        self.readoper = Readout(in_features=vit_channel, use_cls_token=use_cls_token, num_register_tokens=num_register_tokens)
        if scale_factor > 1 and isinstance(scale_factor, int):
            self.sample = nn.ConvTranspose2d(vit_channel, feature_channel, kernel_size=scale_factor, stride=scale_factor, padding=0)
        elif scale_factor > 1:
            self.sample = nn.Sequential(nn.Conv2d(vit_channel, feature_channel, kernel_size=1, stride=1, padding=0))
        elif scale_factor < 1:
            scale_factor = int(1.0 / scale_factor)
            self.sample = nn.Conv2d(vit_channel, feature_channel, kernel_size=scale_factor + 1, stride=scale_factor, padding=1)
        else:
            self.sample = nn.Identity()

    def forward(self, x):
        x = self.readoper(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        if isinstance(self.scale_factor, float):
            x = interpolate_float32(x.float(), scale_factor=self.scale_factor, mode='nearest')
        x = self.sample(x)
        return x

class EncoderFeature(nn.Module):
    def __init__(self, vit_channel, num_ch_dec=[256, 512, 1024, 1024], use_cls_token=True, num_register_tokens=0):
        super(EncoderFeature, self).__init__()
        self.vit_channel = vit_channel
        self.num_ch_dec = num_ch_dec
        self.read_3 = Token2Feature(self.vit_channel, self.num_ch_dec[3], scale_factor=1, use_cls_token=use_cls_token, num_register_tokens=num_register_tokens)
        self.read_2 = Token2Feature(self.vit_channel, self.num_ch_dec[2], scale_factor=1, use_cls_token=use_cls_token, num_register_tokens=num_register_tokens)
        self.read_1 = Token2Feature(self.vit_channel, self.num_ch_dec[1], scale_factor=2, use_cls_token=use_cls_token, num_register_tokens=num_register_tokens)
        self.read_0 = Token2Feature(self.vit_channel, self.num_ch_dec[0], scale_factor=7 / 2, use_cls_token=use_cls_token, num_register_tokens=num_register_tokens)

    def forward(self, ref_feature):
        x = self.read_3(ref_feature[3])  
        x2 = self.read_2(ref_feature[2])  
        x1 = self.read_1(ref_feature[1])  
        x0 = self.read_0(ref_feature[0])  
        return x, x2, x1, x0

class DecoderFeature(nn.Module):
    def __init__(self, vit_channel, num_ch_dec=[128, 256, 512, 1024, 1024], use_cls_token=True):
        super(DecoderFeature, self).__init__()
        self.vit_channel = vit_channel
        self.num_ch_dec = num_ch_dec
        self.upconv_3 = FuseBlock(self.num_ch_dec[4], self.num_ch_dec[3], fuse=False, upsample=False)
        self.upconv_2 = FuseBlock(self.num_ch_dec[3], self.num_ch_dec[2])
        self.upconv_1 = FuseBlock(self.num_ch_dec[2], self.num_ch_dec[1] + 2, scale_factor=7 / 4)

    def forward(self, ref_feature):
        x, x2, x1, x0 = ref_feature  
        x = self.upconv_3(x)  
        x = self.upconv_2(x, x2)  
        x = self.upconv_1(x, x1)  
        return x

# =====================================================================
# [新增模块] 1. 注意力融合模块 (Channel + Spatial Attention)
# =====================================================================
class SafeFusionModule(nn.Module):
    def __init__(self, semantic_dim, geo_dim, mid_dims=128, out_dims=64):
        super().__init__()
        combined_dim = semantic_dim + geo_dim
        
        # 1. 特征对齐与初步降维
        self.pre_conv = nn.Sequential(
            nn.Conv2d(combined_dim, mid_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_dims),
            nn.ReLU(inplace=True)
        )

        # 2. 通道注意力 (Channel Attention)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_dims, mid_dims // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dims // 8, mid_dims, kernel_size=1),
            nn.Sigmoid()
        )

        # 3. 空间注意力 (Spatial Attention)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3), 
            nn.Sigmoid()
        )

        # 4. 最终输出映射
        self.post_conv = nn.Sequential(
            nn.Conv2d(mid_dims, out_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True)
        )

    def forward(self, semantic_feat, geo_feat):
        # 尺寸对齐保护
        if geo_feat.shape[2:] != semantic_feat.shape[2:]:
            geo_feat = F.interpolate(geo_feat, size=semantic_feat.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([semantic_feat, geo_feat], dim=1)
        x = self.pre_conv(x)

        # --- 通道注意力应用 ---
        x = x * self.channel_gate(x)

        # --- 空间注意力应用 ---
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_gate(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_weight

        return self.post_conv(x)

# =====================================================================
# [新增模块] 2. 语义适配器 FPN (Semantic Adapter FPN)
# =====================================================================
class SemanticAdapterFPN(nn.Module):
    def __init__(self, feature_channels, out_channels=128):
        super().__init__()
        self.conv_14 = nn.Sequential(
            nn.Conv2d(feature_channels[2], out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_7 = nn.Sequential(
            nn.Conv2d(feature_channels[1], out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(feature_channels[0], out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        in_dim = out_channels * 3
        self.aspp_fuse = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        target_size = features[3].shape[2:] 
        x14 = F.interpolate(self.conv_14(features[1]), size=target_size, mode='bilinear', align_corners=True)
        x7 = F.interpolate(self.conv_7(features[2]), size=target_size, mode='bilinear', align_corners=True)
        x4 = self.conv_4(features[3])
        concat_feat = torch.cat([x14, x7, x4], dim=1)
        fused_feat = self.aspp_fuse(concat_feat)
        return self.final_conv(fused_feat)

# =====================================================================
# 主网络结构: RAFTDepthNormalSafeDPT5
# =====================================================================
class RAFTDepthNormalSafeDPT5(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.model.decode_head.in_channels  
        self.feature_channels = cfg.model.decode_head.feature_channels  
        self.decoder_channels = cfg.model.decode_head.decoder_channels  
        self.use_cls_token = cfg.model.decode_head.use_cls_token
        self.up_scale = cfg.model.decode_head.up_scale
        self.num_register_tokens = cfg.model.decode_head.num_register_tokens
        self.min_val = cfg.data_basic.depth_normalize[0]
        self.max_val = cfg.data_basic.depth_normalize[1]
        self.regress_scale = 100.0

        self.hidden_dims = self.context_dims = cfg.model.decode_head.hidden_channels  
        self.n_gru_layers = cfg.model.decode_head.n_gru_layers  
        self.n_downsample = cfg.model.decode_head.n_downsample  
        self.iters = cfg.model.decode_head.iters  
        self.slow_fast_gru = cfg.model.decode_head.slow_fast_gru  

        # --- 3D 几何组件 ---
        self.num_depth_regressor_anchor = 256  
        self.used_res_channel = self.decoder_channels[1]  
        self.token2feature = EncoderFeature(self.in_channels[0], self.feature_channels, self.use_cls_token, self.num_register_tokens)
        self.decoder_mono = DecoderFeature(self.in_channels, self.decoder_channels)
        
        self.depth_regressor = nn.Sequential(
            nn.Conv2d(self.used_res_channel, self.num_depth_regressor_anchor, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_depth_regressor_anchor, self.num_depth_regressor_anchor, kernel_size=1),
        )
        self.normal_predictor = nn.Sequential(
            nn.Conv2d(self.used_res_channel, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1),
        )

        # --- 2D 语义组件 & 注意力后融合 (DSGI 优化版) ---
        self.safe_semantic_fpn = SemanticAdapterFPN(self.feature_channels, out_channels=128)
        self.safe_fusion = SafeFusionModule(semantic_dim=128, geo_dim=self.hidden_dims[2], mid_dims=128, out_dims=64)
        self.safe_head = nn.Conv2d(64, 2, kernel_size=1)

        # --- RAFT 上下文与更新块 ---
        self.context_feature_encoder = ContextFeatureEncoder(self.feature_channels, [self.hidden_dims, self.context_dims])
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(self.context_dims[i], self.hidden_dims[i] * 3, 3, padding=3 // 2) for i in range(self.n_gru_layers)])
        self.update_block = BasicMultiUpdateBlock(cfg, hidden_dims=self.hidden_dims, out_dims=6)

        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        # 权重初始化
        for module_list in [self.safe_semantic_fpn, self.safe_fusion, self.safe_head]:
            for m in module_list.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)

    def get_bins(self, bins_num):
        depth_bins_vec = torch.linspace(math.log(self.min_val), math.log(self.max_val), bins_num, device="cuda")
        depth_bins_vec = torch.exp(depth_bins_vec)
        return depth_bins_vec

    def register_depth_expectation_anchor(self, bins_num, B):
        depth_bins_vec = self.get_bins(bins_num)
        depth_bins_vec = depth_bins_vec.unsqueeze(0).repeat(B, 1)
        self.register_buffer('depth_expectation_anchor', depth_bins_vec, persistent=False)

    def clamp(self, x):
        y = self.relu(x - self.min_val) + self.min_val
        y = self.max_val - self.relu(self.max_val - y)
        return y

    def regress_depth(self, feature_map_d):
        prob_feature = self.depth_regressor(feature_map_d)
        prob = prob_feature.softmax(dim=1)
        B = prob.shape[0]
        if "depth_expectation_anchor" not in self._buffers:
            self.register_depth_expectation_anchor(self.num_depth_regressor_anchor, B)
        d = compute_depth_expectation(
            prob,
            self.depth_expectation_anchor[:B, ...]).unsqueeze(1)
        return (self.clamp(d) - self.max_val) / self.regress_scale, prob_feature

    def pred_normal(self, feature_map, confidence):
        normal_out = self.normal_predictor(feature_map)
        return norm_normalize(torch.cat([normal_out, confidence], dim=1))

    def upsample_flow(self, flow, mask):
        N, D, H, W = flow.shape
        factor = 2 ** self.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def initialize_flow(self, img):
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)
        return coords0, coords1

    def upsample(self, x, scale_factor=2):
        return interpolate_float32(x, scale_factor=scale_factor * self.up_scale / 8, mode="nearest")

    def run_update_block(self, net_list, inp_list, iter32, iter16, iter08, update):
        return self.update_block(net_list, inp_list, iter32=iter32, iter16=iter16, iter08=iter08, update=update)

    def forward(self, vit_features, **kwargs):
        B, H, W, _, _, num_register_tokens = vit_features[1]
        vit_features = vit_features[0]

        if self.use_cls_token == True:
            vit_features = [[ft[:, 1 + num_register_tokens:, :].view(B, H, W, self.in_channels[0]), \
                             ft[:, 0:1 + num_register_tokens, :].view(B, 1, 1, self.in_channels[0] * (1 + num_register_tokens))]
                            for ft in vit_features]
        else:
            vit_features = [ft.view(B, H, W, self.in_channels[0]) for ft in vit_features]
            
        encoder_features = checkpoint(self.token2feature, vit_features)  

        # --- 独立提取 2D 语义特征 ---
        semantic_feat = checkpoint(self.safe_semantic_fpn, encoder_features)

        # --- 初始化 3D 几何分支 ---
        ref_feat = checkpoint(self.decoder_mono, encoder_features) 
        feature_map = ref_feat[:, :-2, :, :]  
        depth_confidence_map = ref_feat[:, -2:-1, :, :]
        normal_confidence_map = ref_feat[:, -1:, :, :]
        depth_pred, binmap = checkpoint(self.regress_depth, feature_map)  
        normal_pred = checkpoint(self.pred_normal, feature_map, normal_confidence_map)  
        depth_init = torch.cat((depth_pred, depth_confidence_map, normal_pred), dim=1) 

        cnet_list = checkpoint(self.context_feature_encoder, encoder_features[::-1])
        net_list = [torch.tanh(x[0]) for x in cnet_list]  
        inp_list = [torch.relu(x[1]) for x in cnet_list]  
        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]

        coords0, coords1 = self.initialize_flow(net_list[0])
        coords1 = coords1 + depth_init

        if self.training:
            low_resolution_init = [self.clamp(depth_init[:, :1] * self.regress_scale + self.max_val),
                                   depth_init[:, 1:2], norm_normalize(depth_init[:, 2:].clone())]
            init_depth = upflow4(depth_init)
            flow_predictions = [self.clamp(init_depth[:, :1] * self.regress_scale + self.max_val)]
            conf_predictions = [init_depth[:, 1:2]]
            normal_outs = [norm_normalize(init_depth[:, 2:].clone())]
        else:
            flow_predictions, conf_predictions, normal_outs, low_resolution_init = [], [], [], []

        # ================== RAFT 核心几何迭代 ==================
        for itr in range(self.iters):
            flow = coords1 - coords0
            if self.n_gru_layers == 3 and self.slow_fast_gru: 
                net_list = checkpoint(self.run_update_block, net_list, inp_list, True, False, False, False)
            if self.n_gru_layers >= 2 and self.slow_fast_gru: 
                net_list = checkpoint(self.run_update_block, net_list, inp_list, self.n_gru_layers == 3, True, False, False)
            
            net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, None, flow, iter32=self.n_gru_layers == 3, iter16=self.n_gru_layers >= 2)
            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = self.upsample(coords1 - coords0, 4)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(self.clamp(flow_up[:, :1] * self.regress_scale + self.max_val))
            conf_predictions.append(flow_up[:, 1:2])
            normal_outs.append(norm_normalize(flow_up[:, 2:].clone()))

        # ================== 注意力后融合 (Safe预测) ==================
        # 此时 net_list[0] 蕴含了经过多次优化的最强 3D 隐特征
        final_geo_context = net_list[0] 
        
        # 将 2D 语义与 3D 几何特征送入 CBAM-style 注意力模块
        fused_safe_feat = self.safe_fusion(semantic_feat, final_geo_context)
        
        # 经过注意力提纯后的特征，输出安全区 logits
        safe_logits_low_res = self.safe_head(fused_safe_feat)
        
        # 采用双线性插值上采样，避免被深度的 up_mask 撕裂语义边缘
        scale_factor = 2 ** self.n_downsample
        final_safe_pred = interpolate_float32(
            safe_logits_low_res, 
            scale_factor=scale_factor, 
            mode='bilinear', 
            align_corners=True
        )

        # 兼容你的计算图，将 final_safe_pred 复制成列表以匹配迭代次数
        safe_predictions = [final_safe_pred] * len(flow_predictions)

        outputs = dict(
            prediction=flow_predictions[-1],
            predictions_list=flow_predictions,
            safe_prediction=safe_predictions[-1],
            safe_predictions_list=safe_predictions,
            confidence=conf_predictions[-1],
            confidence_list=conf_predictions,
            pred_logit=None,
            prediction_normal=normal_outs[-1],
            normal_out_list=normal_outs,
            low_resolution_init=low_resolution_init,
        )

        return outputs
    

class IterativeCoupledRAFTDepthNormalSafeDPT5(RAFTDepthNormalSafeDPT5):
    """
    继承自晚期融合版本的耦合迭代版本。
    强迫 Safe 预测参与 RAFT 多步流，利用 3D 几何的每一步 Delta 来门控/修正 2D 语义状态，
    以提升跨域 (Cross-domain) 泛化能力。
    """
    def __init__(self, cfg):
        # 1. 完整继承父类的骨架（ViT提取、FPN、深度/法向预测头、RAFT基础组件）
        super().__init__(cfg)

        # 2. [覆盖/新增] 针对多步迭代专门设计的 Safe 分支组件
        # 覆盖父类的 safe_head，因为输入的通道数变成了 hidden_dims[2]
        self.safe_head = nn.Sequential(
            nn.Conv2d(self.hidden_dims[2], 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        # 新增：独立的语义 GRU 上下文初始化
        self.safe_context_encoder = nn.Sequential(
            nn.Conv2d(128, self.hidden_dims[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.safe_zqr_conv = nn.Conv2d(128, self.hidden_dims[2] * 3, kernel_size=3, padding=1)

        # 新增：几何到语义的交叉注意力注入 (G2S Injection)
        # 输入维度: hidden_dims[2] (net_list[0]) + 6 (dn_flow: depth(1)+conf(1)+normal(4))
        self.geo_to_semantic_attention = nn.Sequential(
            nn.Conv2d(self.hidden_dims[2] + 6, self.hidden_dims[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dims[2], self.hidden_dims[2], kernel_size=1),
            nn.Sigmoid() 
        )

        # 新增：用于 Safe 分支残差更新的专属 GRU (2分类 logits)
        self.safe_gru = ConvGRU(self.hidden_dims[2], input_dim=2) 
        
        self._initialize_iterative_weights()

    def _initialize_iterative_weights(self):
        # 仅初始化子类新增或覆盖的组件
        for module_list in [self.safe_context_encoder, self.safe_zqr_conv, self.geo_to_semantic_attention, self.safe_head]:
            for m in module_list.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
        # 初始化门控偏置为 0，让初始状态下 3D 先验对 2D 特征的影响为中性
        init.constant_(self.geo_to_semantic_attention[-2].bias, 0)

    def initialize_flow_safe(self, img):
        # 为 2D logits 初始化 0 坐标体系
        N, _, H, W = img.shape
        coords0 = coords_grid_safe(N, H, W).to(img.device)
        coords1 = coords_grid_safe(N, H, W).to(img.device)
        return coords0, coords1

    def forward(self, vit_features, **kwargs):
        # =================================================================
        # 第一阶段：基础特征提取 (复用父类逻辑)
        # =================================================================
        B, H, W, _, _, num_register_tokens = vit_features[1]
        vit_features = vit_features[0]

        if self.use_cls_token == True:
            vit_features = [[ft[:, 1 + num_register_tokens:, :].view(B, H, W, self.in_channels[0]), \
                             ft[:, 0:1 + num_register_tokens, :].view(B, 1, 1, self.in_channels[0] * (1 + num_register_tokens))]
                            for ft in vit_features]
        else:
            vit_features = [ft.view(B, H, W, self.in_channels[0]) for ft in vit_features]
            
        encoder_features = checkpoint(self.token2feature, vit_features)  

        # =================================================================
        # 第二阶段：初始化双流状态 (3D几何流 + 2D语义流)
        # =================================================================
        # 1. 语义流 (Safe) 初始化
        semantic_feat = checkpoint(self.safe_semantic_fpn, encoder_features)
        safe_net_state = torch.tanh(self.safe_context_encoder(semantic_feat))
        safe_inp_zqr = list(self.safe_zqr_conv(semantic_feat).split(split_size=self.hidden_dims[2], dim=1))

        # 2. 几何流 (Depth/Normal) 初始化
        ref_feat = checkpoint(self.decoder_mono, encoder_features) 
        feature_map = ref_feat[:, :-2, :, :]  
        depth_confidence_map = ref_feat[:, -2:-1, :, :]
        normal_confidence_map = ref_feat[:, -1:, :, :]
        depth_pred, binmap = checkpoint(self.regress_depth, feature_map)  
        normal_pred = checkpoint(self.pred_normal, feature_map, normal_confidence_map)  
        depth_init = torch.cat((depth_pred, depth_confidence_map, normal_pred), dim=1) 

        cnet_list = checkpoint(self.context_feature_encoder, encoder_features[::-1])
        net_list = [torch.tanh(x[0]) for x in cnet_list]  
        inp_list = [torch.relu(x[1]) for x in cnet_list]  
        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]

        # 3. 坐标系初始化
        coords0, coords1 = self.initialize_flow(net_list[0])
        safe_coords0, safe_coords1 = self.initialize_flow_safe(net_list[0]) 
        coords1 = coords1 + depth_init

        if self.training:
            low_resolution_init = [self.clamp(depth_init[:, :1] * self.regress_scale + self.max_val),
                                   depth_init[:, 1:2], norm_normalize(depth_init[:, 2:].clone())]
            init_depth = upflow4(depth_init)
            flow_predictions = [self.clamp(init_depth[:, :1] * self.regress_scale + self.max_val)]
            conf_predictions = [init_depth[:, 1:2]]
            normal_outs = [norm_normalize(init_depth[:, 2:].clone())]
            safe_predictions = []
        else:
            flow_predictions, conf_predictions, normal_outs, low_resolution_init, safe_predictions = [], [], [], [], []

        # =================================================================
        # 第三阶段：耦合多步迭代更新 (Coupled Multi-step Update)
        # =================================================================
        for itr in range(self.iters):
            # --- [几何更新流] (保持 Metric3D 原始逻辑) ---
            flow = coords1 - coords0
            if self.n_gru_layers == 3 and self.slow_fast_gru: 
                net_list = checkpoint(self.run_update_block, net_list, inp_list, True, False, False, False)
            if self.n_gru_layers >= 2 and self.slow_fast_gru: 
                net_list = checkpoint(self.run_update_block, net_list, inp_list, self.n_gru_layers == 3, True, False, False)
            
            net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, None, flow, iter32=self.n_gru_layers == 3, iter16=self.n_gru_layers >= 2)
            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = self.upsample(coords1 - coords0, 4)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(self.clamp(flow_up[:, :1] * self.regress_scale + self.max_val))
            conf_predictions.append(flow_up[:, 1:2])
            normal_outs.append(norm_normalize(flow_up[:, 2:].clone()))

            # --- [语义更新流] (受几何变化量门控约束) ---
            dn_flow = coords1 - coords0           # 当前步的几何残差 (含隐式 3D 先验)
            safe_flow = safe_coords1 - safe_coords0 # 当前步的语义残差
            
            # 核心交叉注意力：计算当前的 3D 几何特征如何指导 2D 语义状态
            geo_gate = self.geo_to_semantic_attention(torch.cat([net_list[0], dn_flow], dim=1))
            modulated_safe_net_state = safe_net_state * geo_gate + safe_net_state 

            # GRU 状态更新：使用被 3D 修正过的状态
            safe_net_state = self.safe_gru(modulated_safe_net_state, *safe_inp_zqr, safe_flow)

            # 预测类别 Logits 残差并累加
            safe_delta_flow = checkpoint(self.safe_head, safe_net_state)
            safe_coords1 = safe_coords1 + safe_delta_flow
            
            # 规避边缘撕裂：使用双线性插值上采样，摒弃深度的 up_mask
            scale_factor = 2 ** self.n_downsample
            final_safe_pred = interpolate_float32(
                safe_coords1, 
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=True
            )
            safe_predictions.append(final_safe_pred)

        outputs = dict(
            prediction=flow_predictions[-1],
            predictions_list=flow_predictions,
            safe_prediction=safe_predictions[-1],
            safe_predictions_list=safe_predictions, # 包含所有迭代步，支持 Sequence Loss
            confidence=conf_predictions[-1],
            confidence_list=conf_predictions,
            pred_logit=None,
            prediction_normal=normal_outs[-1],
            normal_out_list=normal_outs,
            low_resolution_init=low_resolution_init,
        )

        return outputs


class ExplicitGeometryPrior(nn.Module):
    """提取纯物理的深度梯度与法向特征，提供绝对的域无关先验"""
    def __init__(self, out_channels=32):
        super().__init__()
        # 1(深度) + 1(深度梯度) + 3(法向量XYZ) = 5
        self.geo_encoder = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def compute_spatial_gradient(self, depth):
        # 显式计算深度的空间梯度，检测悬崖/障碍物边缘
        grad_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        grad_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        return grad_x + grad_y

    def forward(self, depth, normal):
        depth_grad = self.compute_spatial_gradient(depth)
        explicit_geo = torch.cat([depth, depth_grad, normal], dim=1)
        return self.geo_encoder(explicit_geo)

class ExplicitSafeFusionModule(nn.Module):
    """支持三路输入（2D语义 + 3D隐式 + 3D显式）的注意力融合模块"""
    def __init__(self, semantic_dim, implicit_geo_dim, explicit_geo_dim=32, mid_dims=128, out_dims=64):
        super().__init__()
        combined_dim = semantic_dim + implicit_geo_dim + explicit_geo_dim
            
        self.pre_conv = nn.Sequential(
            nn.Conv2d(combined_dim, mid_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_dims),
            nn.ReLU(inplace=True)
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_dims, mid_dims // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dims // 8, mid_dims, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3), 
            nn.Sigmoid()
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(mid_dims, out_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True)
        )

    def forward(self, semantic_feat, implicit_geo_feat, explicit_geo_feat):
        # 尺寸对齐
        if implicit_geo_feat.shape[2:] != semantic_feat.shape[2:]:
            implicit_geo_feat = F.interpolate(implicit_geo_feat, size=semantic_feat.shape[2:], mode='bilinear', align_corners=True)
        if explicit_geo_feat.shape[2:] != semantic_feat.shape[2:]:
            explicit_geo_feat = F.interpolate(explicit_geo_feat, size=semantic_feat.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([semantic_feat, implicit_geo_feat, explicit_geo_feat], dim=1)
        x = self.pre_conv(x)

        x = x * self.channel_gate(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_gate(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_weight

        return self.post_conv(x)
    


class GeoRAFTDepthNormalSafeDPT5(RAFTDepthNormalSafeDPT5):
    """
    继承自 RAFTDepthNormalSafeDPT5 的增强泛化版本。
    注入了基于物理定律的显式深度梯度与法向先验。
    """
    def __init__(self, cfg):
        # 1. 完整执行父类初始化 (加载主干网络、RAFT循环等所有基础组件)
        super().__init__(cfg)
        
        # 2. 挂载显式物理特征提取器
        self.safe_explicit_geo_extractor = ExplicitGeometryPrior(out_channels=32)
        
        # 3. 覆盖父类的双路融合，替换为我们的三路特征融合
        self.safe_fusion = ExplicitSafeFusionModule(
            semantic_dim=128, 
            implicit_geo_dim=self.hidden_dims[2], 
            explicit_geo_dim=32,
            mid_dims=128, 
            out_dims=64
        )
        self._initialize_explicit_weights()

    def _initialize_explicit_weights(self):
        # 仅初始化子类新增的网络权重
        for m in [self.safe_explicit_geo_extractor, self.safe_fusion]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    init.constant_(layer.weight, 1)
                    init.constant_(layer.bias, 0)

    def forward(self, vit_features, **kwargs):
        # =================================================================
        # 第一部分：完全复用父类的特征提取与 RAFT 迭代流
        # =================================================================
        B, H, W, _, _, num_register_tokens = vit_features[1]
        vit_features = vit_features[0]

        if self.use_cls_token == True:
            vit_features = [[ft[:, 1 + num_register_tokens:, :].view(B, H, W, self.in_channels[0]), \
                             ft[:, 0:1 + num_register_tokens, :].view(B, 1, 1, self.in_channels[0] * (1 + num_register_tokens))]
                            for ft in vit_features]
        else:
            vit_features = [ft.view(B, H, W, self.in_channels[0]) for ft in vit_features]
            
        encoder_features = checkpoint(self.token2feature, vit_features)  
        semantic_feat = checkpoint(self.safe_semantic_fpn, encoder_features)

        ref_feat = checkpoint(self.decoder_mono, encoder_features) 
        feature_map = ref_feat[:, :-2, :, :]  
        depth_confidence_map = ref_feat[:, -2:-1, :, :]
        normal_confidence_map = ref_feat[:, -1:, :, :]
        depth_pred, binmap = checkpoint(self.regress_depth, feature_map)  
        normal_pred = checkpoint(self.pred_normal, feature_map, normal_confidence_map)  
        depth_init = torch.cat((depth_pred, depth_confidence_map, normal_pred), dim=1) 

        cnet_list = checkpoint(self.context_feature_encoder, encoder_features[::-1])
        net_list = [torch.tanh(x[0]) for x in cnet_list]  
        inp_list = [torch.relu(x[1]) for x in cnet_list]  
        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]

        coords0, coords1 = self.initialize_flow(net_list[0])
        coords1 = coords1 + depth_init

        if self.training:
            low_resolution_init = [self.clamp(depth_init[:, :1] * self.regress_scale + self.max_val),
                                   depth_init[:, 1:2], norm_normalize(depth_init[:, 2:].clone())]
            init_depth = upflow4(depth_init)
            flow_predictions = [self.clamp(init_depth[:, :1] * self.regress_scale + self.max_val)]
            conf_predictions = [init_depth[:, 1:2]]
            normal_outs = [norm_normalize(init_depth[:, 2:].clone())]
        else:
            flow_predictions, conf_predictions, normal_outs, low_resolution_init = [], [], [], []

        for itr in range(self.iters):
            flow = coords1 - coords0
            if self.n_gru_layers == 3 and self.slow_fast_gru: 
                net_list = checkpoint(self.run_update_block, net_list, inp_list, True, False, False, False)
            if self.n_gru_layers >= 2 and self.slow_fast_gru: 
                net_list = checkpoint(self.run_update_block, net_list, inp_list, self.n_gru_layers == 3, True, False, False)
            
            net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, None, flow, iter32=self.n_gru_layers == 3, iter16=self.n_gru_layers >= 2)
            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = self.upsample(coords1 - coords0, 4)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(self.clamp(flow_up[:, :1] * self.regress_scale + self.max_val))
            conf_predictions.append(flow_up[:, 1:2])
            normal_outs.append(norm_normalize(flow_up[:, 2:].clone()))

        # =================================================================
        # 第二部分：子类注入的【显式物理特征提取】与【三路融合】
        # =================================================================
        # 1. 隐式 3D 几何特征
        final_geo_context = net_list[0] 
        
        # 2. 从网络最终预测中剥离纯物理量 (深度绝对值、法向 XYZ)
        pred_depth = flow_predictions[-1]
        pred_normal = normal_outs[-1][:, :3, :, :] 
        
        # 3. 提取显式几何先验
        explicit_geo_feat = self.safe_explicit_geo_extractor(pred_depth, pred_normal)
        
        # 4. 执行三路注意力融合 (2D语义 + 3D隐式 + 3D显式物理先验)
        fused_safe_feat = self.safe_fusion(semantic_feat, final_geo_context, explicit_geo_feat)
        
        # 5. 上采样与预测输出
        safe_logits_low_res = self.safe_head(fused_safe_feat)
        
        scale_factor = 2 ** self.n_downsample
        final_safe_pred = interpolate_float32(
            safe_logits_low_res, 
            scale_factor=scale_factor, 
            mode='bilinear', 
            align_corners=True
        )

        safe_predictions = [final_safe_pred] * len(flow_predictions)

        outputs = dict(
            prediction=flow_predictions[-1],
            predictions_list=flow_predictions,
            safe_prediction=safe_predictions[-1],
            safe_predictions_list=safe_predictions,
            confidence=conf_predictions[-1],
            confidence_list=conf_predictions,
            pred_logit=None,
            prediction_normal=normal_outs[-1],
            normal_out_list=normal_outs,
            low_resolution_init=low_resolution_init,
        )

        return outputs

    # def forward(self, vit_features, **kwargs):
    #     ## read vit token to multi-scale features
    #     B, H, W, _, _, num_register_tokens = vit_features[1]
    #     vit_features = vit_features[0]
    #
    #     ## Error logging
    #     if torch.isnan(vit_features[0]).any():
    #         print('vit_feature_nan!!!')
    #     if torch.isinf(vit_features[0]).any():
    #         print('vit_feature_inf!!!')
    #
    #     if self.use_cls_token == True:
    #         vit_features = [[ft[:, 1 + num_register_tokens:, :].view(B, H, W, self.in_channels[0]), \
    #                          ft[:, 0:1 + num_register_tokens, :].view(B, 1, 1, self.in_channels[0] * (
    #                                      1 + num_register_tokens))] for ft in vit_features]
    #     else:
    #         vit_features = [ft.view(B, H, W, self.in_channels[0]) for ft in vit_features]
    #     encoder_features = self.token2feature(vit_features)  # 1/14, 1/14, 1/7, 1/4
    #
    #     ## Error logging
    #     for en_ft in encoder_features:
    #         if torch.isnan(en_ft).any():
    #             print('decoder_feature_nan!!!')
    #             print(en_ft.shape)
    #         if torch.isinf(en_ft).any():
    #             print('decoder_feature_inf!!!')
    #             print(en_ft.shape)
    #
    #     ## decode features to init-depth (and confidence)
    #     ref_feat = self.decoder_mono(encoder_features)  # now, 1/4 for depth
    #
    #     ## Error logging
    #     if torch.isnan(ref_feat).any():
    #         print('ref_feat_nan!!!')
    #     if torch.isinf(ref_feat).any():
    #         print('ref_feat_inf!!!')
    #
    #     feature_map = ref_feat[:, :-2, :, :]  # feature map share of depth and normal prediction
    #     depth_confidence_map = ref_feat[:, -2:-1, :, :]
    #     normal_confidence_map = ref_feat[:, -1:, :, :]
    #     depth_pred, binmap = self.regress_depth(feature_map)  # regress bin for depth
    #     normal_pred = self.pred_normal(feature_map, normal_confidence_map)  # mlp for normal
    #
    #     depth_init = torch.cat((depth_pred, depth_confidence_map, normal_pred), dim=1)  # (N, 1+1+4, H, W)
    #
    #     ## encoder features to context-feature for init-hidden-state and contex-features
    #     cnet_list = self.context_feature_encoder(encoder_features[::-1])
    #     net_list = [torch.tanh(x[0]) for x in cnet_list]  # x_4, x_8, x_16 of hidden state
    #     inp_list = [torch.relu(x[1]) for x in cnet_list]  # x_4, x_8, x_16 context features
    #
    #     # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
    #     inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
    #                 zip(inp_list, self.context_zqr_convs)]
    #
    #     coords0, coords1 = self.initialize_flow(net_list[0])
    #     if depth_init is not None:
    #         coords1 = coords1 + depth_init
    #
    #     if self.training:
    #         low_resolution_init = [self.clamp(depth_init[:, :1] * self.regress_scale + self.max_val),
    #                                depth_init[:, 1:2], norm_normalize(depth_init[:, 2:].clone())]
    #         init_depth = upflow4(depth_init)
    #         flow_predictions = [self.clamp(init_depth[:, :1] * self.regress_scale + self.max_val)]
    #         conf_predictions = [init_depth[:, 1:2]]
    #         normal_outs = [norm_normalize(init_depth[:, 2:].clone())]
    #
    #     else:
    #         flow_predictions = []
    #         conf_predictions = []
    #         samples_pred_list = []
    #         coord_list = []
    #         normal_outs = []
    #         low_resolution_init = []
    #
    #     for itr in range(self.iters):
    #         # coords1 = coords1.detach()
    #         flow = coords1 - coords0
    #         if self.n_gru_layers == 3 and self.slow_fast_gru:  # Update low-res GRU
    #             net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False,
    #                                          update=False)
    #         if self.n_gru_layers >= 2 and self.slow_fast_gru:  # Update low-res GRU and mid-res GRU
    #             net_list = self.update_block(net_list, inp_list, iter32=self.n_gru_layers == 3, iter16=True,
    #                                          iter08=False, update=False)
    #         net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, None, flow,
    #                                                           iter32=self.n_gru_layers == 3,
    #                                                           iter16=self.n_gru_layers >= 2)
    #
    #         # F(t+1) = F(t) + \Delta(t)
    #         coords1 = coords1 + delta_flow
    #
    #         # We do not need to upsample or output intermediate results in test_mode
    #         # if (not self.training) and itr < self.iters-1:
    #         # continue
    #
    #         # upsample predictions
    #         if up_mask is None:
    #             flow_up = self.upsample(coords1 - coords0, 4)
    #         else:
    #             flow_up = self.upsample_flow(coords1 - coords0, up_mask)
    #             # flow_up = self.upsample(coords1-coords0, 4)
    #
    #         flow_predictions.append(self.clamp(flow_up[:, :1] * self.regress_scale + self.max_val))
    #         conf_predictions.append(flow_up[:, 1:2])
    #         normal_outs.append(norm_normalize(flow_up[:, 2:].clone()))
    #
    #     outputs = dict(
    #         prediction=flow_predictions[-1],
    #         predictions_list=flow_predictions,
    #         confidence=conf_predictions[-1],
    #         confidence_list=conf_predictions,
    #         pred_logit=None,
    #         # samples_pred_list=samples_pred_list,
    #         # coord_list=coord_list,
    #         prediction_normal=normal_outs[-1],
    #         normal_out_list=normal_outs,
    #         low_resolution_init=low_resolution_init,
    #     )

        # return outputs


if __name__ == "__main__":
    try:
        from mmcv.utils import Config
    except:
        from mmengine import Config
    cfg = Config.fromfile('/mu.hu/monodepth/mono/configs/RAFTDecoder/vit.raft.full2t.py')
    cfg.model.decode_head.in_channels = [384, 384, 384, 384]
    cfg.model.decode_head.feature_channels = [96, 192, 384, 768]
    cfg.model.decode_head.decoder_channels = [48, 96, 192, 384, 384]
    cfg.model.decode_head.hidden_channels = [48, 48, 48, 48, 48]
    cfg.model.decode_head.up_scale = 7
    
    # cfg.model.decode_head.use_cls_token = True
    # vit_feature = [[torch.rand((2, 20, 60, 384)).cuda(), torch.rand(2, 384).cuda()], \
    #         [torch.rand((2, 20, 60, 384)).cuda(), torch.rand(2, 384).cuda()], \
    #         [torch.rand((2, 20, 60, 384)).cuda(), torch.rand(2, 384).cuda()], \
    #         [torch.rand((2, 20, 60, 384)).cuda(), torch.rand(2, 384).cuda()]]
    
    cfg.model.decode_head.use_cls_token = True
    cfg.model.decode_head.num_register_tokens = 4
    vit_feature = [[torch.rand((2, (74 * 74) + 5, 384)).cuda(),\
                    torch.rand((2, (74 * 74) + 5, 384)).cuda(), \
                    torch.rand((2, (74 * 74) + 5, 384)).cuda(), \
                    torch.rand((2, (74 * 74) + 5, 384)).cuda()], (2, 74, 74, 1036, 1036, 4)]

    decoder = RAFTDepthNormalDPT5(cfg).cuda()
    output = decoder(vit_feature)
    temp = 1




