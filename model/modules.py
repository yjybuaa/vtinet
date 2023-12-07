"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

Based on XMem (https://github.com/hkchengrex/XMem)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.group_modules import *
from model import resnet
from model.cbam import CBAM


class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]
        g = self.distributor(x, g)
        g = self.block1(g)
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])
        g = self.block2(g+r)
        return g


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)

        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu        # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1    # 1/4, 64
        self.layer2 = network.layer2    # 1/8, 128
        self.layer3 = network.layer3    # 1/16, 256

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g)         # 1/2, 64
        g = self.maxpool(g)     # 1/4, 64
        g2 = self.relu(g) 

        g4 = self.layer1(g2)    # 1/4
        g8 = self.layer2(g4)    # 1/8
        g16 = self.layer3(g8)   # 1/16

        g2 = g2.view(batch_size, num_objects, *g2.shape[1:])
        g4 = g4.view(batch_size, num_objects, *g4.shape[1:])
        g8 = g8.view(batch_size, num_objects, *g8.shape[1:])
        g16 = g16.view(batch_size, num_objects, *g16.shape[1:])
        g = self.fuser(image_feat_f16, g16)

        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)

        return g, h, g16, g8, g4, g2
 

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu        # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1      # 1/4, 256
        self.layer2 = network.layer2    # 1/8, 512
        self.layer3 = network.layer3    # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)        # 1/2, 64
        x = self.maxpool(x)     # 1/4, 64
        f4 = self.res2(x)       # 1/4, 256
        f8 = self.layer2(f4)    # 1/8, 512
        f16 = self.layer3(f8)   # 1/16, 1024

        return f16, f8, f4, x


class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f)
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g


class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x, need_s, need_e):
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection


class Decoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None
        
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2]

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)
        msoutput = [g16, g8, g4]

        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        
        # logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        # logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, msoutput


class CFI0(nn.Module):    
    def __init__(self, out_dim):
        super(CFI0, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)        
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        
    def forward(self, rgb, thermal):
        x_rgb = self.layer_10(rgb)
        x_thermal = self.layer_20(thermal)
        rgb_w = nn.Sigmoid()(x_rgb)
        thermal_w = nn.Sigmoid()(x_thermal)
        x_rgb_w = rgb.mul(thermal_w)
        x_thermal_w = thermal.mul(rgb_w)
        x_rgb_r = x_rgb_w + rgb
        x_thermal_r = x_thermal_w + thermal
        
        # RGBT feature fusion
        x_rgb_r = self.layer_11(x_rgb_r)
        x_thermal_r = self.layer_21(x_thermal_r)
        ful_mul = torch.mul(x_rgb_r, x_thermal_r)         
        x_in1 = torch.reshape(x_rgb_r,[x_rgb_r.shape[0],1,x_rgb_r.shape[1],x_rgb_r.shape[2],x_rgb_r.shape[3]])
        x_in2 = torch.reshape(x_thermal_r,[x_thermal_r.shape[0],1,x_thermal_r.shape[1],x_thermal_r.shape[2],x_thermal_r.shape[3]])
        x_cat = torch.cat((x_in1, x_in2),dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul,ful_max),dim=1)
        out1 = self.layer_ful1(ful_out)
        
        return out1


class CFI(nn.Module):
    def __init__(self,in_dim, out_dim, divide):
        super(CFI, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        self.reduc_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        self.reduc_01 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=1), act_fn)
        self.reduc_02 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=1), act_fn)
        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)        
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)
        self.layer_ful2 = nn.Sequential(nn.Conv2d(out_dim+out_dim//divide, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

    def forward(self, rgb, thermal, xx):
        x_rgb = self.reduc_1(rgb)
        x_thermal = self.reduc_2(thermal)
        x_rgb1 = self.layer_10(x_rgb)
        x_thermal1 = self.layer_20(x_thermal)
        rgb_w = nn.Sigmoid()(x_rgb1)
        thermal_w = nn.Sigmoid()(x_thermal1)
        rgb_w = nn.Sigmoid()(x_rgb1)
        thermal_w = nn.Sigmoid()(x_thermal1)
        x_rgb_w = x_rgb.mul(thermal_w)
        x_thermal_w = x_thermal.mul(rgb_w)
        x_rgb_r = x_rgb_w + x_rgb
        x_thermal_r = x_thermal_w + x_thermal

        # RGBT feature fusion
        x_rgb_r = self.layer_11(x_rgb_r)
        x_thermal_r = self.layer_21(x_thermal_r)
        ful_mul = torch.mul(x_rgb_r, x_thermal_r)         
        x_in1 = torch.reshape(x_rgb_r,[x_rgb_r.shape[0],1,x_rgb_r.shape[1],x_rgb_r.shape[2],x_rgb_r.shape[3]])
        x_in2 = torch.reshape(x_thermal_r,[x_thermal_r.shape[0],1,x_thermal_r.shape[1],x_thermal_r.shape[2],x_thermal_r.shape[3]])
        x_cat = torch.cat((x_in1, x_in2),dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul,ful_max),dim=1)
        out1 = self.layer_ful1(ful_out)
        out2 = self.layer_ful2(torch.cat([out1, xx],dim=1))

        return out2


class CFU(nn.Module):    
    def __init__(self, in_dim):
        super(CFU, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.layer_10 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)   
        self.layer_cat1 = nn.Sequential(nn.Conv2d(in_dim*2, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),)        
        
    def forward(self, x_ful, x1, x2):
        if len(x_ful.shape) == 5:
            # shape is b*t*c*h*w
            # b, t = thermal.shape[:2]
            b, t = x_ful.shape[:2]
            x_ful = x_ful.view(b*t, *x_ful.shape[-3:])
            x1 = x1.view(b*t, *x1.shape[-3:])
            x2 = x2.view(b*t, *x2.shape[-3:])
            x_ful_1 = x_ful.mul(x1)
            x_ful_2 = x_ful.mul(x2)
            x_ful_w = self.layer_cat1(torch.cat([x_ful_1, x_ful_2],dim=1))
            out = self.relu(x_ful + x_ful_w)
            out = out.view(b, t, *out.shape[-3:])
        else:
            x_ful_1 = x_ful.mul(x1)
            x_ful_2 = x_ful.mul(x2)
            x_ful_w = self.layer_cat1(torch.cat([x_ful_1, x_ful_2],dim=1))
            out = self.relu(x_ful + x_ful_w)
        
        return out


class Decoder_fuse(nn.Module):
    def __init__(self, hidden_dim=64, val_dim=512):
        super().__init__()
        
        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None
        
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256)  # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)
        self.fusion_modules = nn.ModuleList()
        self.fusion_modules.append(CFU(512))
        self.fusion_modules.append(CFU(256))
        self.fusion_modules.append(CFU(256))

    def forward(self, fuse_feats, rgb_feats, thermal_feats, hidden_state, memory_readout, h_out=True):
        f16, f8, f4 = fuse_feats[0], fuse_feats[1], fuse_feats[2]
        f16_rgb, f8_rgb, f4_rgb = rgb_feats[0], rgb_feats[1], rgb_feats[2]
        f16_thermal, f8_thermal, f4_thermal = thermal_feats[0], thermal_feats[1], thermal_feats[2]
        batch_size, num_objects = memory_readout.shape[:2]
        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        # first CFU
        g16 = self.fusion_modules[0](g16, f16_rgb, f16_thermal)
        g8 = self.up_16_8(f8, g16)
        # second CFU
        g8 = self.fusion_modules[1](g8, f8_rgb, f8_thermal)
        g4 = self.up_8_4(f4, g8)
        # third CFU
        g4 = self.fusion_modules[2](g4, f4_rgb, f4_thermal)
        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, logits
