# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import onnx
import onnxruntime
from onnxsim.onnx_simplifier import simplify
import numpy as np
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
    model.eval()

    dummy_image = torch.rand(1, 3, 800, 800)
    with torch.no_grad():
        res1=model(dummy_image.cuda())

    onnx_path = 'anchor-detr-dc5.onnx'
    torch.onnx.export(model, (dummy_image.cuda(),), onnx_path,
                      opset_version=12,
                      input_names=["inputs"], output_names=["pred_logits", "pred_boxes"])
    onnx.checker.check_model(onnx_path)

    model_sim, check_ok=simplify(onnx.load(onnx_path))
    onnx.save_model(model_sim,onnx_path)

    ort_session = onnxruntime.InferenceSession(onnx_path)
    res2=ort_session.run(None, {"inputs":dummy_image.cpu().numpy()})
    res1 = list(res1.values())
    assert (np.abs(res1[0].cpu().numpy()-res2[0]).max() < 1e-5) and (np.abs(res1[1].cpu().numpy()-res2[1]).max() < 1e-5), "inaccurate results"

    print('done')
    return

if __name__ == '__main__':
    export_onnx()
