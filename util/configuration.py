"""
Training congfigurations for VTiNet
Based on XMem (https://github.com/hkchengrex/XMem)
"""
from argparse import ArgumentParser


def none_or_default(x, default):
    return x if x is not None else default

class Configuration():
    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()

        # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
        parser.add_argument('--benchmark', action='store_true')
        parser.add_argument('--no_amp', action='store_true')

        # Data parameters
        parser.add_argument('--rgbt_root', help='RGBT data root')
        parser.add_argument('--save_path', help='save path')
        parser.add_argument('--num_workers', help='Total number of dataloader workers across all GPUs processes', type=int, default=16)

        parser.add_argument('--key_dim', default=64, type=int)
        parser.add_argument('--value_dim', default=512, type=int)
        parser.add_argument('--hidden_dim', default=64, help='Set to =0 to disable', type=int)

        parser.add_argument('--deep_update_prob', default=0.2, type=float)

        # Training parameters
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--iterations', default=3000, type=int)
        # fine-tune means fewer augmentations to train the sensory memory
        parser.add_argument('--finetune', default=800, type=int)
        parser.add_argument('--steps', nargs="*", default=[2000], type=int)
        parser.add_argument('--lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--lr_thermal', help='Initial learning rate', default=5e-6, type=float)
        parser.add_argument('--num_ref_frames', default=3, type=int)
        parser.add_argument('--num_frames', default=8, type=int)
        parser.add_argument('--start_warm', default=500, type=int)
        parser.add_argument('--end_warm', default=2500, type=int)
        
        parser.add_argument('--gamma', help='LR := LR*gamma at every decay step', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.05, type=float)

        # On top of XMem pre-trained model
        parser.add_argument('--load_network', help='Path to pretrained network weight only')
        parser.add_argument('--load_checkpoint', help='Path to the checkpoint file, including network, optimizer and such')

        # Logging information
        parser.add_argument('--log_text_interval', default=100, type=int)
        parser.add_argument('--log_image_interval', default=1000, type=int)
        parser.add_argument('--save_network_interval', default=100, type=int)
        parser.add_argument('--save_checkpoint_interval', default=200, type=int)
        parser.add_argument('--exp_id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard', default='NULL')
        parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

        self.args['amp'] = not self.args['no_amp']


    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)
