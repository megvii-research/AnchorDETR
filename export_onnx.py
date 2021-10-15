# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import onnx
import onnxruntime
from onnxsim.onnx_simplifier import simplify

import torch

import argparse
from main import get_args_parser as get_main_args_parser
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('onnx', add_help=False)
    parser.add_argument('--checkpoiont', default='', type=str,help='path to the checkpoiont')
    parser.add_argument('--device', default='cpu', type=str, choices=['cpu', 'cuda'])
    return parser

def export_onnx():

    args = get_args_parser().parse_args()
    device = args.device
    checkpoiont = args.checkpoiont

    main_args = get_main_args_parser().parse_args()
    main_args.aux_loss = False

    model, _, _ = build_model(main_args)
    if checkpoiont:
        model.load_state_dict(torch.load(checkpoiont)['model'])
    model.to(device)
    model.eval()

    dummy_image = torch.rand(1, 3, 800, 800)
    with torch.no_grad():
        res1=model(dummy_image.to(device))

    onnx_path = 'anchor-detr-dc5.onnx'
    torch.onnx.export(model, (dummy_image.to(device),), onnx_path,
                      opset_version=12,
                      input_names=["inputs"], output_names=["pred_logits", "pred_boxes"])
    onnx.checker.check_model(onnx_path)

    model_sim, check_ok=simplify(onnx.load(onnx_path))
    onnx.save_model(model_sim,onnx_path)

    ort_session = onnxruntime.InferenceSession(onnx_path)
    res2=ort_session.run(None, {"inputs":dummy_image.cpu().numpy()})
    res1 = list(res1.values())
    torch.testing.assert_allclose(res1[0].cpu().numpy(), res2[0], rtol=1e-03, atol=1e-05)
    torch.testing.assert_allclose(res1[1].cpu().numpy(), res2[1], rtol=1e-03, atol=1e-05)

    print('done')
    return

if __name__ == '__main__':
    export_onnx()
