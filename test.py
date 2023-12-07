"""
Inference code for VTiNet
Based on XMem (https://github.com/hkchengrex/XMem)
"""

import os
from os import path
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from inference.data.test_datasets import RGBTTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import VTiNet
from inference.inference_core import InferenceCore

from progressbar import progressbar


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', required=True)

# Data options
parser.add_argument('--rgbt_path',required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--save_all', action='store_true', 
			help='Save all frames. Useful only in YouTubeVOS/long-time video', )

parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
		
# Long-term memory options
parser.add_argument('--disable_long_term', action='store_true')
parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
												type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=10)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

# Multi-scale options
parser.add_argument('--save_scores', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--size', default=480, type=int, 
			help='Resize the shorter side to this size. -1 to use original resolution. ')

args = parser.parse_args()
config = vars(args)
config['enable_long_term'] = not config['disable_long_term']

if args.output is None:
	args.output = f'../output/{args.dataset}_{args.split}'
	print(f'Output path not provided. Defaulting to {args.output}')

"""
Data preparation
"""
out_path = args.output
meta_dataset = RGBTTestDataset(args.rgbt_path)
torch.autograd.set_grad_enabled(False)

# Set up loader
meta_loader = meta_dataset.get_datasets()

# Load our checkpoint
network = VTiNet(config, args.model).cuda().eval()
if args.model is not None:
	# model_weights = torch.load(args.model)['network']
	model_weights = torch.load(args.model)
	network.load_weights(model_weights, init_as_zero_if_needed=True)
else:
	print('No model loaded.')

total_process_time = 0
total_frames = 0

# Start eval
for vid_reader in progressbar(meta_loader, max_value=len(meta_dataset), redirect_stdout=True):

	loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
	vid_name = vid_reader.vid_name
	vid_length = len(loader)
	# no need to count usage for LT if the video is not that long anyway
	config['enable_long_term_count_usage'] = (
		config['enable_long_term'] and
		(vid_length
			/ (config['max_mid_term_frames']-config['min_mid_term_frames'])
			* config['num_prototypes'])
		>= config['max_long_term_elements']
	)

	mapper = MaskMapper()
	processor = InferenceCore(network, config=config)
	first_mask_loaded = False

	for ti, data in enumerate(loader):
		with torch.cuda.amp.autocast(enabled=not args.benchmark):
			rgb = data['rgb'].cuda()[0]
			thermal = data['thermal'].cuda()[0]
			msk = data.get('mask')
			info = data['info']
			frame = info['frame'][0]
			shape = info['shape']
			need_resize = info['need_resize'][0]

			"""
			For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
			Seems to be very similar in testing as my previous timing method 
			with two cuda sync + time.time() in STCN though 
			"""
			start = torch.cuda.Event(enable_timing=True)
			end = torch.cuda.Event(enable_timing=True)
			start.record()

			if not first_mask_loaded:
				if msk is not None:
					first_mask_loaded = True
				else:
					# no point to do anything without a mask
					continue

			if args.flip:
				rgb = torch.flip(rgb, dims=[-1])
				msk = torch.flip(msk, dims=[-1]) if msk is not None else None

			# Map possibly non-continuous labels to continuous ones
			if msk is not None:
				msk, labels = mapper.convert_mask(msk[0].numpy())
				msk = torch.Tensor(msk).cuda()
				if need_resize:
					msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
				processor.set_all_labels(list(mapper.remappings.values()))
			else:
				labels = None

			# Run the model on this frame
			prob = processor.step(rgb, thermal, msk, labels, end=(ti==vid_length-1))

			# Upsample to original size if needed
			if need_resize:
				prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

			end.record()
			torch.cuda.synchronize()
			total_process_time += (start.elapsed_time(end)/1000)
			total_frames += 1

			if args.flip:
				prob = torch.flip(prob, dims=[-1])

			# Probability mask -> index mask
			out_mask = torch.argmax(prob, dim=0)
			out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

			# Save the mask
			if args.save_all or info['save'][0]:

				this_out_path = path.join(out_path, vid_name)
				os.makedirs(this_out_path, exist_ok=True)
				out_mask = mapper.remap_index_mask(out_mask)
				out_img = Image.fromarray(out_mask)
				if vid_reader.get_palette() is not None:
					out_img.putpalette(vid_reader.get_palette())

				frame_idx = '{:0>5}'.format(int(frame[:-4]))
				out_img.save(os.path.join(this_out_path, frame_idx+'.png'))


print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')
