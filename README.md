## VTiNet for Visible-Thermal Video Object Segmentation

Dataset and Code for the paper:
>Unveiling the Power of Visible-Thermal Video Object Segmentation

### The VisT300 Dataset

[Google Drive](https://drive.google.com/drive/folders/1OYD1dfIDi-JTJ6wLjtPxCD0lWKc-TXmu?usp=sharing)

```
VisT300
├── train
|   └── RGBImages
|       └── video1
|           ├── 00000.jpg
|           ├── 00005.jpg
|           ├── xxxxx.jpg
|       ...
|   └── ThermalImages
|       └── video1
|           ├── 00000.jpg
|           ├── 00005.jpg
|           ├── xxxxx.jpg
|       ...
|   └── Annotations
|       └── video1
|           ├── 00000.png
|           ├── 00005.png
|           ├── xxxxx.png
|       ...
├── test (same organization as the train set)
```

### VTiNet

PyTorch implementation of VTiNet. We test the code in the following environments, other versions may also be compatible: ```Python=3.9, PyTorch=1.10.1, CUDA=11.3```

- Install
```shell
pip install -r requirements.txt
```

- Train
```shell
torchrun --master_port 10010 --nproc_per_node=2 train.py --exp_id vist300 --rgbt_root [path to VisT300/train] --save_path [path to save checkpoints] --load_network [path to pretrained xmem]
```
[pretrained xmem](https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth)

- Test
```shell
python test.py --model [path to vtinet checkpoint] --rgbt_path [path to path to VisT300/test] --save_path [path to results]
```
[vtinet checkpoint](https://drive.google.com/drive/folders/1OYD1dfIDi-JTJ6wLjtPxCD0lWKc-TXmu?usp=sharing)

- Evaluate
```shell
python eval.py -g [path to VisT300/test/Annotations] -r [path to results]
```
