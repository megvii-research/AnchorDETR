# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import io

from main import get_args_parser as get_main_args_parser
from models import build_model

from util.misc import  nested_tensor_from_tensor_list


def export_onnx():

    main_args = get_main_args_parser().parse_args()
    main_args.aux_loss = False

    model, _, _ = build_model(main_args)
    model.cuda()
    # model=torch.jit.script(model)
    model.eval()

    dummy_image = torch.rand(1, 3, 800, 800).cuda()
    dummy_image = nested_tensor_from_tensor_list(dummy_image)
    model(dummy_image)

    onnx_io = io.BytesIO()
    torch.onnx.export(model, (torch.rand(1, 3, 800, 800).cuda(),), onnx_io,
                      opset_version=12,
                      input_names=["inputs"], output_names=["pred_logits", "pred_boxes"])
    print('done')
    return

if __name__ == '__main__':
    export_onnx()
