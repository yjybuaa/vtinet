"""
Network for VTiNet
Based on XMem (https://github.com/hkchengrex/XMem)
"""

import torch
import torch.nn as nn

from model.aggregate import aggregate
from model.modules import *
from model.memory_util import *


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


class VTiNet(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)

        self.single_object = config.get('single_object', False)
        print(f'Single object mode: {self.single_object}')

        self.key_encoder_rgb = KeyEncoder()
        self.key_encoder_thermal = KeyEncoder()

        self.value_encoder_rgb = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object)
        self.value_encoder_thermal = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object)

        # Projection from f16 feature space to key/value space
        self.key_proj_rgb = KeyProjection(1024, self.key_dim)
        self.key_proj_thermal = KeyProjection(1024, self.key_dim)
        self.key_proj_fuse = KeyProjection(1024, self.key_dim)

        self.decoder_rgb = Decoder(self.value_dim, self.hidden_dim)
        self.decoder_thermal = Decoder(self.value_dim, self.hidden_dim)

        # freeze rgb encoder
        for p in self.key_encoder_rgb.parameters():
            p.requires_grad_(False)
        for p in self.value_encoder_rgb.parameters():
            p.requires_grad_(False)
        for p in self.key_proj_rgb.parameters():
            p.requires_grad_(False)
        for p in self.decoder_rgb.parameters():
            p.requires_grad_(False)

        # fusion encoders
        self.fusion_encoder_list = nn.ModuleList()
        self.fusion_encoder_list.append(CFI0(64))
        self.fusion_encoder_list.append(CFI(256, 256, 4))   # 64 -> 256
        self.fusion_encoder_list.append(CFI(512, 512, 2))   # 256 -> 512
        self.fusion_encoder_list.append(CFI(1024, 1024, 2)) # 512 -> 1024

        self.fusion_pool_1 = maxpool()
        self.fusion_pool_2 = maxpool()

        # fusion value encoders
        self.fusion_value_encoder_list = nn.ModuleList()
        scale = 1
        self.fusion_value_encoder_list.append(CFI0(64*scale))
        self.fusion_value_encoder_list.append(CFI(64*scale, 64*scale, 1))       # 64 -> 64
        self.fusion_value_encoder_list.append(CFI(128*scale, 128*scale, 2))     # 64 -> 128
        self.fusion_value_encoder_list.append(CFI(256*scale, 256*scale, 2))     # 128 -> 256

        self.fusion_value_pool_1 = maxpool()
        self.fusion_value_pool_2 = maxpool()

        # fusion fuser
        self.fuser_fuse = FeatureFusionBlock(1024, 256, 512, 512)
        self.hidden_reinforce_fuse = HiddenReinforcer(512, 64)

        self.decoder_fuse = Decoder_fuse()

        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)

    def encode_key_rgb(self, frame, thermal, need_sk=True, need_ek=True): 
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
    
        f16, f8, f4, f2= self.key_encoder_rgb(frame)
        key, shrinkage, selection = self.key_proj_rgb(f16, need_sk, need_ek)

        if need_reshape:
            # B*C*T*H*W
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])
            f2 = f2.view(b, t, *f2.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4, f2

    def encode_key_thermal(self, frame, thermal, need_sk=True, need_ek=True): 
        # Determine input shape
        if len(thermal.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = thermal.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            thermal = thermal.flatten(start_dim=0, end_dim=1)
        elif len(thermal.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
    
        f16, f8, f4, f2 = self.key_encoder_thermal(thermal)
        key, shrinkage, selection = self.key_proj_thermal(f16, need_sk, need_ek)

        if need_reshape:
            # B*C*T*H*W
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])
            f2 = f2.view(b, t, *f2.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4, f2

    def fuse_key(self, rgb_feats, thermal_feats):

        need_sk = True
        need_ek = True

        # Determine input shape
        if len(rgb_feats[-1].shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            # b, t = thermal.shape[:2]
            b, t = rgb_feats[-1].shape[:2]
            # flatten so that we can feed them into a 2D CNN
            # thermal = thermal.flatten(start_dim=0, end_dim=1)
            f2 = self.fusion_encoder_list[0](rgb_feats[-1].view(b*t, *rgb_feats[-1].shape[-3:]), thermal_feats[-1].view(b*t, *thermal_feats[-1].shape[-3:]))
            f4 = self.fusion_encoder_list[1](rgb_feats[-2].view(b*t, *rgb_feats[-2].shape[-3:]), thermal_feats[-2].view(b*t, *thermal_feats[-2].shape[-3:]), f2)
            f8 = self.fusion_encoder_list[2](rgb_feats[-3].view(b*t, *rgb_feats[-3].shape[-3:]), thermal_feats[-3].view(b*t, *thermal_feats[-3].shape[-3:]), self.fusion_pool_1(f4))
            f16 = self.fusion_encoder_list[3](rgb_feats[-4].view(b*t, *rgb_feats[-4].shape[-3:]), thermal_feats[-4].view(b*t, *thermal_feats[-4].shape[-3:]), self.fusion_pool_2(f8))

        elif len(rgb_feats[-1].shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
            f2 = self.fusion_encoder_list[0](rgb_feats[-1], thermal_feats[-1])
            f4 = self.fusion_encoder_list[1](rgb_feats[-2], thermal_feats[-2], f2)
            f8 = self.fusion_encoder_list[2](rgb_feats[-3], thermal_feats[-3], self.fusion_pool_1(f4))
            f16 = self.fusion_encoder_list[3](rgb_feats[-4], thermal_feats[-4], self.fusion_pool_2(f8))

        else:
            raise NotImplementedError

        key, shrinkage, selection = self.key_proj_rgb(f16, need_sk, need_ek)

        if need_reshape:
            # B*C*T*H*W
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])
            f2 = f2.view(b, t, *f2.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4, f2

    def encode_value_rgb(self, frame, image_feat_f16, h16, masks, is_deep_update=True): 
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i!=j]]
                , dim=1, keepdim=True)
            for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        g, h16, g16, g8, g4, g2 = self.value_encoder_rgb(frame, image_feat_f16, h16, masks, others, is_deep_update)

        return g, h16, g16, g8, g4, g2

    def encode_value_thermal(self, frame, image_feat_f16, h16, masks, is_deep_update=True): 
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i!=j]]
                , dim=1, keepdim=True)
            for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        g, h16, g16, g8, g4, g2 = self.value_encoder_thermal(frame, image_feat_f16, h16, masks, others, is_deep_update)

        return g, h16, g16, g8, g4, g2

    def fuse_value(self, rgb_feats, thermal_feats, image_feat_f16, h):

        # Determine input shape
        if len(rgb_feats[-1].shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            # b, t = thermal.shape[:2]
            b, t = rgb_feats[-1].shape[:2]

            f2 = self.fusion_value_encoder_list[0](rgb_feats[-1].view(b*t, *rgb_feats[-1].shape[-3:]), thermal_feats[-1].view(b*t, *thermal_feats[-1].shape[-3:]))
            f4 = self.fusion_value_encoder_list[1](rgb_feats[-2].view(b*t, *rgb_feats[-2].shape[-3:]), thermal_feats[-2].view(b*t, *thermal_feats[-2].shape[-3:]), f2)
            f4 = self.fusion_value_pool_1(f4)
            f8 = self.fusion_value_encoder_list[2](rgb_feats[-3].view(b*t, *rgb_feats[-3].shape[-3:]), thermal_feats[-3].view(b*t, *thermal_feats[-3].shape[-3:]), f4)
            f8 = self.fusion_value_pool_2(f8)
            f16 = self.fusion_value_encoder_list[3](rgb_feats[-4].view(b*t, *rgb_feats[-4].shape[-3:]), thermal_feats[-4].view(b*t, *thermal_feats[-4].shape[-3:]), f8)

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])

            g = self.fuser_fuse(image_feat_f16, f16)

            h = self.hidden_reinforce_fuse(g, h)

        elif len(rgb_feats[-1].shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
            
            f2 = self.fusion_value_encoder_list[0](rgb_feats[-1], thermal_feats[-1])
            f4 = self.fusion_value_encoder_list[1](rgb_feats[-2], thermal_feats[-2], f2)
            f4 = self.fusion_value_pool_1(f4)
            f8 = self.fusion_value_encoder_list[2](rgb_feats[-3], thermal_feats[-3], f4)
            f8 = self.fusion_value_pool_2(f8)
            f16 = self.fusion_value_encoder_list[3](rgb_feats[-4], thermal_feats[-4], f8)

            g = self.fuser_fuse(image_feat_f16, f16)

            h = self.hidden_reinforce_fuse(g, h)
        else:
            raise NotImplementedError

        return g, h


    # Used in training only. 
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key, query_selection, memory_key, 
                    memory_shrinkage, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
        memory = readout(affinity, memory_value)
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

        return memory

    # def segment(self, multi_scale_features, memory_readout,
    #                 hidden_state, selector=None, h_out=True, strip_bg=True): 
    def segment(self, rgb_feats, thermal_feats, fuse_feats,
                    selector=None, h_out=True, strip_bg=True): 

        multi_scale_features_rgb = rgb_feats[:-2]
        memory_readout_rgb = rgb_feats[-2]
        hidden_state_rgb = rgb_feats[-1]

        multi_scale_features_thermal = thermal_feats[:-2]
        memory_readout_thermal = thermal_feats[-2]
        hidden_state_thermal = thermal_feats[-1]

        multi_scale_features_fuse = fuse_feats[:-2]
        memory_readout_fuse = fuse_feats[-2]
        hidden_state_fuse = fuse_feats[-1]

        hidden_state_rgb, ms_features_rgb = self.decoder_rgb(*multi_scale_features_rgb, hidden_state_rgb, memory_readout_rgb, h_out=h_out)
        hidden_state_thermal, ms_features_thermal = self.decoder_thermal(*multi_scale_features_thermal, hidden_state_thermal, memory_readout_thermal, h_out=h_out)

        hidden_state_fuse, logits = self.decoder_fuse(multi_scale_features_fuse, ms_features_rgb, ms_features_thermal, hidden_state_fuse, memory_readout_fuse, h_out=h_out)
        
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector
            
        logits, prob = aggregate(prob, dim=1, return_logits=True)
        if strip_bg:
            # Strip away the background
            prob = prob[:, 1:]

        return hidden_state_rgb, hidden_state_thermal, hidden_state_fuse, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            key_rgb, shrinkage_rgb, selection_rgb, f16_rgb, f8_rgb, f4_rgb, f2_rgb \
                = self.encode_key_rgb(*args, **kwargs)
            key_thermal, shrinkage_thermal, selection_thermal, f16_thermal, f8_thermal, f4_thermal, f2_thermal \
                = self.encode_key_thermal(*args, **kwargs)
            return [key_rgb, shrinkage_rgb, selection_rgb, f16_rgb, f8_rgb, f4_rgb, f2_rgb], \
                [key_thermal, shrinkage_thermal, selection_thermal, f16_thermal, f8_thermal, f4_thermal, f2_thermal]
        elif mode == 'encode_value_rgb':
            return self.encode_value_rgb(*args, **kwargs)
        elif mode == 'encode_value_thermal':
            return self.encode_value_thermal(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        elif mode == 'encode_fuse_key':
            return self.fuse_key(*args, **kwargs)
        elif mode == 'encode_fuse_value':
            return self.fuse_value(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        If model_path is provided, we load these from the model weights
        The actual parameters are then updated to the config in-place

        Otherwise we load it either from the config or default
        """
        if model_path is not None:
            # load the model and key/value/hidden dimensions with some hacks
            # config is updated with the loaded parameters
            model_weights = torch.load(model_path, map_location=map_location)['network']
            # model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim = model_weights['key_proj_rgb.key_proj.weight'].shape[0]
            self.value_dim = model_weights['value_encoder_rgb.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'decoder_rgb.hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['decoder_rgb.hidden_update.transform.weight'].shape[0]//3
            print(f'Hyperparameters read from the model weights: '
                    f'C^k={self.key_dim}, C^v={self.value_dim}, C^h={self.hidden_dim}')
        else:
            model_weights = None
            # load dimensions from config or default
            if 'key_dim' not in config:
                self.key_dim = 64
                print(f'key_dim not found in config. Set to default {self.key_dim}')
            else:
                self.key_dim = config['key_dim']

            if 'value_dim' not in config:
                self.value_dim = 512
                print(f'value_dim not found in config. Set to default {self.value_dim}')
            else:
                self.value_dim = config['value_dim']

            if 'hidden_dim' not in config:
                self.hidden_dim = 64
                print(f'hidden_dim not found in config. Set to default {self.hidden_dim}')
            else:
                self.hidden_dim = config['hidden_dim']

            self.disable_hidden = (self.hidden_dim <= 0)

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Maps SO weight (without other_mask) to MO weight (with other_mask)

        # src_dict = src_dict['network']

        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    print('Converting weights from single object to multiple objects.')
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        model_dict = self.state_dict()

        state_dict_rgb = {(k.split('.')[0]+'_rgb.'+k.removeprefix(k.split('.')[0]+'.')):v for k,v in src_dict.items() if (k.split('.')[0]+'_rgb.'+k.removeprefix(k.split('.')[0]+'.')) in model_dict.keys()}
        state_dict_thermal = {(k.split('.')[0]+'_thermal.'+k.removeprefix(k.split('.')[0]+'.')):v for k,v in src_dict.items() if (k.split('.')[0]+'_thermal.'+k.removeprefix(k.split('.')[0]+'.')) in model_dict.keys()} 
        state_dict_others = {k:v for k,v in src_dict.items() if k in model_dict.keys()}

        state_dict = {**state_dict_rgb, **state_dict_thermal, **state_dict_others}

        self.load_state_dict(state_dict, strict=False)
