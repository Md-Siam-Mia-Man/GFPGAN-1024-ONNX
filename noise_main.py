import torch
noise_dict = {}
size = [(1, 1, 4, 4),(1, 1, 8, 8),(1, 1, 16, 16),(1, 1, 32, 32),(1, 1, 64, 64),(1, 1, 128, 128),(1, 1, 256, 256),(1, 1, 512, 512), (1, 1, 1024, 1024)] # Added 1024x1024
for s in size: 
    out = torch.rand(s)#.cuda()
    # Ensure the noise is generated on CPU if CUDA is not available or not intended here.
    # For ONNX export, usually CPU is fine for parameter initialization.
    noise = torch.empty_like(out).normal_() 
    noise_dict[s[2]] = noise