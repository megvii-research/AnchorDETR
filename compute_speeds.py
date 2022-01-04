# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

# taken from https://gist.github.com/fmassa/c0fbb9fe7bf53b533b5cc241f5c8234c with a few modifications

# taken from detectron2 / fvcore with a few modifications
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/analysis.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import numpy as np
import tqdm
import torch
import time


from main import get_args_parser as get_main_args_parser
from models import build_model
from datasets import build_dataset


def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()


def measure_time(model, inputs, N=10):
    warmup(model, inputs)
    s = time.time()
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    return t


def fmt_res(data):
    return data.mean(), data.std(), data.min(), data.max()


def benchmark():

    main_args = get_main_args_parser().parse_args()

    dataset = build_dataset('val', main_args)
    model, _, _ = build_model(main_args)
    model.cuda()
    model.eval()

    images = []
    for idx in range(100):
        img, t = dataset[idx]
        images.append(img)

    with torch.no_grad():
        tmp = []
        for img in tqdm.tqdm(images):
            inputs = [img.to('cuda')]
            t = measure_time(model, inputs)
            tmp.append(t)

    res = {'time': fmt_res(np.array(tmp))}

    return res

if __name__ == '__main__':
    res = benchmark()
    print(res)
