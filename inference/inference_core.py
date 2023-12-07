from inference.memory_manager import MemoryManager
from model.network import VTiNet
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by, unpad


class InferenceCore:
    def __init__(self, network:VTiNet, config):
        self.config = config
        self.network = network
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)

        self.clear_memory()
        self.all_labels = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        self.memory_rgb = MemoryManager(config=self.config)
        self.memory_thermal = MemoryManager(config=self.config)
        self.memory_fuse = MemoryManager(config=self.config)

    def update_config(self, config):
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)
        self.memory_rgb.update_config(config)
        self.memory_thermal.update_config(config)
        self.memory_fuse.update_config(config)

    def set_all_labels(self, all_labels):
        # self.all_labels = [l.item() for l in all_labels]
        self.all_labels = all_labels

    def step(self, rgb, thermal, mask=None, valid_labels=None, end=False):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        self.curr_ti += 1
        rgb, self.pad = pad_divide_by(rgb, 16)
        rgb = rgb.unsqueeze(0) # add the batch dimension

        thermal, _ = pad_divide_by(thermal, 16)
        thermal = thermal.unsqueeze(0) # add the batch dimension

        is_mem_frame = ((self.curr_ti-self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end)
        need_segment = (self.curr_ti > 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels)))
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame) or  # synchronized
            (not self.deep_update_sync and self.curr_ti-self.last_deep_update_ti >= self.deep_update_every) # no-sync
        ) and (not end)
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (not end)

        # key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image, 
        #                                             need_ek=(self.enable_long_term or need_segment), 
        #                                             need_sk=is_mem_frame)
        # encode key
        key_rgb, shrinkage_rgb, selection_rgb, f16_rgb, f8_rgb, f4_rgb, f2_rgb = self.network.encode_key_rgb(rgb, 0, 
                                                    need_ek=(self.enable_long_term or need_segment), 
                                                    need_sk=is_mem_frame)
        key_thermal, shrinkage_thermal, selection_thermal, f16_thermal, f8_thermal, f4_thermal, f2_thermal = self.network.encode_key_thermal(0, thermal, 
                                                    need_ek=(self.enable_long_term or need_segment), 
                                                    need_sk=is_mem_frame)
        
        key_fuse, shrinkage_fuse, selection_fuse, f16_fuse, f8_fuse, f4_fuse, f2_fuse \
                = self.network.fuse_key([f16_rgb, f8_rgb, f4_rgb, f2_rgb], [f16_thermal, f8_thermal, f4_thermal, f2_thermal])         
        
        multi_scale_features_rgb = (f16_rgb, f8_rgb, f4_rgb)
        multi_scale_features_thermal = (f16_thermal, f8_thermal, f4_thermal)
        multi_scale_features_fuse = (f16_fuse, f8_fuse, f4_fuse)

        # segment the current frame is needed
        if need_segment:
            memory_readout_rgb = self.memory_rgb.match_memory(key_rgb, selection_rgb).unsqueeze(0)
            memory_readout_thermal = self.memory_thermal.match_memory(key_thermal, selection_thermal).unsqueeze(0)
            memory_readout_fuse = self.memory_fuse.match_memory(key_fuse, selection_fuse).unsqueeze(0)

            # hidden, _, pred_prob_with_bg = self.network.segment(multi_scale_features, memory_readout_rgb, 
            #                         self.memory.get_hidden(), h_out=is_normal_update, strip_bg=False)
            
            hidden_rgb, hidden_thermal, hidden_fuse, _, pred_prob_with_bg = self.network.segment([f16_rgb, f8_rgb, f4_rgb, memory_readout_rgb, self.memory_rgb.get_hidden()], \
                    [f16_thermal, f8_thermal, f4_thermal, memory_readout_thermal, self.memory_thermal.get_hidden()], \
                        [f16_fuse, f8_fuse, f4_fuse, memory_readout_fuse, self.memory_fuse.get_hidden()], \
                        h_out=is_normal_update, strip_bg=False)
            
            # remove batch dim
            pred_prob_with_bg = pred_prob_with_bg[0]
            pred_prob_no_bg = pred_prob_with_bg[1:]
            if is_normal_update:
                self.memory_rgb.set_hidden(hidden_rgb)
                self.memory_thermal.set_hidden(hidden_thermal)
                self.memory_fuse.set_hidden(hidden_fuse)
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        # use the input mask if any
        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)

            if pred_prob_no_bg is not None:
                # if we have a predicted mask, we work on it
                # make pred_prob_no_bg consistent with the input mask
                mask_regions = (mask.sum(0) > 0.5)
                pred_prob_no_bg[:, mask_regions] = 0
                # shift by 1 because mask/pred_prob_no_bg do not contain background
                mask = mask.type_as(pred_prob_no_bg)
                if valid_labels is not None:
                    shift_by_one_non_labels = [i for i in range(pred_prob_no_bg.shape[0]) if (i+1) not in valid_labels]
                    # non-labelled objects are copied from the predicted mask
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]
            pred_prob_with_bg = aggregate(mask, dim=0)

            # also create new hidden states
            self.memory_rgb.create_hidden_state(len(self.all_labels), key_rgb)
            self.memory_thermal.create_hidden_state(len(self.all_labels), key_thermal)
            self.memory_fuse.create_hidden_state(len(self.all_labels), key_fuse)

        # save as memory if needed
        if is_mem_frame:
            # value, hidden = self.network.encode_value(image, f16, self.memory.get_hidden(), 
            #                         pred_prob_with_bg[1:].unsqueeze(0), is_deep_update=is_deep_update)
            v16_rgb, hidden_rgb, g16_rgb, g8_rgb, g4_rgb, g2_rgb \
                        = self.network.encode_value_rgb(rgb, f16_rgb, self.memory_rgb.get_hidden(), pred_prob_with_bg[1:].unsqueeze(0), is_deep_update=is_deep_update)
            
            v16_thermal, hidden_thermal, g16_thermal, g8_thermal, g4_thermal, g2_thermal \
                        = self.network.encode_value_thermal(thermal, f16_thermal, self.memory_thermal.get_hidden(), pred_prob_with_bg[1:].unsqueeze(0), is_deep_update=is_deep_update)

            v16_fuse, hidden_fuse = self.network.fuse_value([g16_rgb, g8_rgb, g4_rgb, g2_rgb], [g16_thermal, g8_thermal, g4_thermal, g2_thermal], f16_fuse, self.memory_fuse.get_hidden())

            self.memory_rgb.add_memory(key_rgb, shrinkage_rgb, v16_rgb, self.all_labels, 
                                    selection=selection_rgb if self.enable_long_term else None)
            self.memory_thermal.add_memory(key_thermal, shrinkage_thermal, v16_thermal, self.all_labels, 
                                    selection=selection_thermal if self.enable_long_term else None)
            self.memory_fuse.add_memory(key_fuse, shrinkage_fuse, v16_fuse, self.all_labels, 
                                    selection=selection_fuse if self.enable_long_term else None)
            
            self.last_mem_ti = self.curr_ti

            if is_deep_update:
                self.memory_rgb.set_hidden(hidden_rgb)
                self.memory_thermal.set_hidden(hidden_thermal)
                self.memory_fuse.set_hidden(hidden_fuse)

                self.last_deep_update_ti = self.curr_ti
                
        return unpad(pred_prob_with_bg, self.pad)
