# GFPGAN-1024 - ONNX

This is a GFPGAN-1024 cleaned up reconsitution code and script to create a onnx model.

The original work is from https://github.com/Glat0s/GFPGAN-1024-onnx.git

## Convert GFPGAN-1024 torch to onnx.

```
pip install -r requirements.txt

wget https://github.com/LeslieZhoa/GFPGAN-1024/releases/download/v0.0/final.pth

python conversion.py --model final.pth --export gfpgan-1024.onnx --size 512
python conversion.py --model final.pth --export gfpgan-1024.onnx --size 512
```

```
# python inference.py --model GFPGANv1.4.onnx --input ./input/Adele_crop.png


# python inference.py --model GFPGANv1.2.onnx --input ./input/Adele_crop.png --output Adele_v2.jpg
# python inference.py --model GFPGANv1.2.onnx --input ./input/Julia_Roberts_crop.png --output Julia_Roberts_v2.jpg
# python inference.py --model GFPGANv1.2.onnx --input ./input/Justin_Timberlake_crop.png --output Justin_Timberlake_v2.jpg
# python inference.py --model GFPGANv1.2.onnx --input ./input/Paris_Hilton_crop.png --output Paris_Hilton_v2.jpg
```


## GFPGAN-1024 ONNX Download

https://github.com/Glat0s/GFPGAN-1024-onnx/releases/download/v0.0.1/gfpgan-1024.onnx
