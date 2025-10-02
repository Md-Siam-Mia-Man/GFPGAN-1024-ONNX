# import cv2
import numpy as np
import time
import torch
import pdb
from collections import OrderedDict

import sys

sys.path.append(".")
sys.path.append("./lib")  # If you have a lib folder, otherwise remove or adjust
import torch.nn as nn
from torch.autograd import Variable
import onnxruntime
import timeit
import onnx

# Try to import the optimizer, but don't fail if it's not there
try:
    from onnx import optimizer as onnx_optimizer_internal

    optimizer_available = True
except ImportError:
    print(
        "Warning: `from onnx import optimizer` failed. ONNX model optimization using internal onnx.optimizer will be skipped."
    )
    print(
        "         For more advanced ONNX model optimization, consider installing the 'onnxoptimizer' package."
    )
    optimizer_available = False


import argparse

# Assuming architecture_1024.py is in the same directory or accessible via PYTHONPATH
from architecture_1024 import GFPGAN_1024

parser = argparse.ArgumentParser("ONNX converter")
parser.add_argument("--model", type=str, default=None, help="src model path")
parser.add_argument("--export", type=str, default=None, help="dst model path")
parser.add_argument(
    "--size", type=int, default=None, help="img size"
)  # This is input image size
args = parser.parse_args()

model = args.model
onnx_model_path = args.export
img_size = args.size  # Input image size to the model, e.g., 512

model = GFPGAN_1024(out_size=1024)
x = torch.rand(1, 3, img_size, img_size)

ckpt = torch.load(model, map_location="cpu", weights_only=False)
state_dict = ckpt["g_ema"] if "g_ema" in ckpt else ckpt

new_state_dict_for_loading = OrderedDict()
print("\nTransforming state_dict keys from checkpoint...")
for k_orig, v_orig in state_dict.items():
    k_transformed = k_orig

    if k_orig.startswith("toRGB.") or k_orig.startswith("final_rgb."):
        # print(f"Skipping U-Net specific key: {k_orig}") # Optional: uncomment for debugging
        continue

    if k_orig.startswith("stylegan_decoder.style_mlp.") or k_orig.startswith(
        "stylegan_decoder.noises."
    ):
        # print(f"Skipping StyleGAN internal key: {k_orig}") # Optional: uncomment for debugging
        continue

    if k_orig.startswith("stylegan_decoder."):
        is_param_of_submodule = False

        if ".modulated_conv.modulation." in k_orig:
            is_param_of_submodule = True
        elif k_orig.endswith(".constant_input.weight"):
            is_param_of_submodule = True

        if is_param_of_submodule:
            last_dot_index = k_orig.rfind(".")
            module_part_str = k_orig[:last_dot_index]
            param_name = k_orig[last_dot_index + 1 :]
            k_transformed = module_part_str.replace(".", "dot") + "." + param_name
        elif k_orig.startswith("stylegan_decoder."):
            k_transformed = k_orig.replace(".", "dot")

    new_state_dict_for_loading[k_transformed] = v_orig
    # if k_transformed != k_orig: # Optional: uncomment for debugging
    #     print(f"Transformed key: {k_orig} -> {k_transformed}")

model.load_state_dict(new_state_dict_for_loading, strict=True)
model.eval()

print(f"\nExporting to ONNX: {onnx_model_path}")
torch.onnx.export(
    model,
    x,
    onnx_model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)

print("\nAttempting ONNX model optimization (fuse_bn_into_conv)...")
if optimizer_available:
    try:
        original_model_onnx = onnx.load(onnx_model_path)
        passes = ["fuse_bn_into_conv"]
        optimized_model_onnx = onnx_optimizer_internal.optimize(
            original_model_onnx, passes
        )
        onnx.save(optimized_model_onnx, onnx_model_path)
        print("ONNX model optimization with internal onnx.optimizer attempted.")
    except Exception as e:
        print(
            f"ONNX optimization with internal onnx.optimizer skipped or failed during processing: {e}"
        )
else:
    print(
        "Skipping ONNX optimization as `onnx.optimizer` could not be imported or is not available for graph passes."
    )


print("\nTesting ONNX model with onnxruntime...")
ort_session = onnxruntime.InferenceSession(onnx_model_path)
print("ONNX Model Inputs:")
for var in ort_session.get_inputs():
    print(f"  Name: {var.name}, Shape: {var.shape}, Type: {var.type}")
print("ONNX Model Outputs:")
for var in ort_session.get_outputs():
    print(f"  Name: {var.name}, Shape: {var.shape}, Type: {var.type}")

ort_input_shape = ort_session.get_inputs()[0].shape
img_h_onnx = ort_input_shape[2] if isinstance(ort_input_shape[2], int) else img_size
img_w_onnx = ort_input_shape[3] if isinstance(ort_input_shape[3], int) else img_size

img_np = np.random.rand(1, 3, img_h_onnx, img_w_onnx).astype(np.float32)

ort_inputs = {ort_session.get_inputs()[0].name: img_np}

print(f"\nRunning inference with onnxruntime on input shape: {img_np.shape}...")
t_start = timeit.default_timer()
ort_outs = ort_session.run(None, ort_inputs)
t_end = timeit.default_timer()

print(f"ONNXRuntime infer time: {t_end - t_start:.4f}s")
if ort_outs:
    print(f"Output shape: {ort_outs[0].shape}")
else:
    print("ONNXRuntime did not return any outputs.")

print("\nScript finished.")
