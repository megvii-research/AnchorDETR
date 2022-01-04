# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from collections import Counter
import tqdm
from fvcore.nn import flop_count_table  # can also try flop_count_str
from detectron2.utils.analysis import FlopCountAnalysis

from main import get_args_parser as get_main_args_parser
from models import build_model
from datasets import build_dataset


def do_flop():
    main_args = get_main_args_parser().parse_args()

    dataset = build_dataset('val', main_args)
    model, _, _ = build_model(main_args)
    model.cuda()
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(100), dataset):  # noqa
        flops = FlopCountAnalysis(model, [data[0].cuda()])
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    print("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    print("Average GFlops for each type of operators:\n"+ str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()]))
    print("Total GFlops: {:.1f}±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9))


if __name__ == "__main__":
    do_flop()
