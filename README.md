**Anchor DETR**: Query Design for Transformer-Based Detector
========


## Introduction
This repository is an official implementation of the [Anchor DETR](https://arxiv.org/abs/2109.07107).
We encode the anchor points as the object queries in DETR.
Multiple patterns are attached to each anchor point to solve the difficulty: "one region, multiple objects".
We also propose an attention variant RCDA to reduce the memory cost for high-resolution features.


![DETR](.github/pipeline.png)


## Main Results



|                    | feature       |  epochs |  AP     |  GFLOPs  | Infer Speed (FPS) |
|:------------------:|:-------------:|:-------:|:-------:|:--------:|:-----------------:|
| DETR               |  DC5          |  500    |  43.3   |  187     | 10 (12)           |
| SMCA               |  multi-level  |  50     |  43.7   |  152     | 10                |
| Deformable DETR    |  multi-level  |  50     |  43.8   |  173     | 15                |
| Conditional DETR   |  DC5          |  50     |  43.8   |  195     | 10                |
| Anchor DETR        |  DC5          |  50     |  44.2   |  151     | 16 (19)           |


*Note:*
1. The results are based on ResNet-50 backbone.
2. Inference speeds are measured on NVIDIA Tesla V100 GPU.
3. The values in parentheses of the Infer Speed indicate the speed with torchscript optimization.


## Model
| name             | backbone  |  AP     |  URL  |
|:----------------:|:---------:|:-------:|:-----:|
| AnchorDETR-C5    |  R50      |  42.1   | [model](https://drive.google.com/file/d/1FKDrTL7qg9riNN5a910Gzf4aZYJTHdT-/view?usp=sharing) / [log](https://drive.google.com/file/d/1b3jy9xkpLA0vi0GWlchtg4SY5jIqVz5S/view?usp=sharing) |
| AnchorDETR-DC5   |  R50      |  44.2   | [model](https://drive.google.com/file/d/1ggsdoBOZa53S4h6Ur3rlK1-7-eABBlid/view?usp=sharing) / [log](https://drive.google.com/file/d/1S3rtBYMsAv437hGL0nm3JlYp6P0nqZfj/view?usp=sharing) |
| AnchorDETR-C5    |  R101     |  43.5   | [model](https://drive.google.com/file/d/19CQqNvrrpdpSxIyn-2IPmLZOf2KP-Zft/view?usp=sharing) / [log](https://drive.google.com/file/d/1O4K00CLiMBaNu0x61xECg7Kek2Rf-tUr/view?usp=sharing) |
| AnchorDETR-DC5   |  R101     |  45.1   | [model](https://drive.google.com/file/d/1bEnFnHCoDSVQ1u_q7B0gR3yxhq12Wevp/view?usp=sharing) / [log](https://drive.google.com/file/d/1wPeEf84zil8yPBLEnweONXadr5LrwXXv/view?usp=sharing) |

*Note:* the models and logs are also available at [Baidu Netdisk](https://pan.baidu.com/s/1Fgx-YPQ0WdTuZIsbOv6hLw) with code `f56r`.

## Usage

### Installation
First, clone the repository locally:
```
git clone https://github.com/megvii-research/AnchorDETR.git
```
Then, install dependencies:
```
pip install -r requirements.txt
```

### Training
To train AnchorDETR on a single node with 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py  --coco_path /path/to/coco 
```

### Evaluation
To evaluate AnchorDETR on a single node with 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --eval --coco_path /path/to/coco --resume /path/to/checkpoint.pth 
```

To evaluate AnchorDETR with a single GPU:
```
python main.py --eval --coco_path /path/to/coco --resume /path/to/checkpoint.pth
```


## Citation

If you find this project useful for your research, please consider citing the paper.
```
@misc{wang2021anchor,
      title={Anchor DETR: Query Design for Transformer-Based Detector},
      author={Yingming Wang and Xiangyu Zhang and Tong Yang and Jian Sun},
      year={2021},
      eprint={2109.07107},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact
If you have any questions, feel free to open an issue or contact us at wangyingming@megvii.com.