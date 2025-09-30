# GFPGAN-1024 - ONNX

This is a GFPGAN-1024 cleaned up reconsitution code and script to create a onnx model.

The original work is from https://github.com/Glat0s/GFPGAN-1024-onnx.git

## Convert GFPGAN-1024 torch to onnx.

```
pip install -r requirements.txt

wget https://github.com/LeslieZhoa/GFPGAN-1024/releases/download/v0.0/final.pth

python torch2onnx.py --model final.pth --export gfpgan-1024.onnx --size 512 
python torch2onnx.py --model final.pth --export gfpgan-1024.onnx --size 512 
```

## GFPGAN-1024 ONNX Download
https://github.com/Glat0s/GFPGAN-1024-onnx/releases/download/v0.0.1/gfpgan-1024.onnx
