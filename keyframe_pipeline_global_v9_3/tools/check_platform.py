#!/usr/bin/env python3
import os, shutil, torch
print("environment:", "colab" if os.path.isdir("/content") else "linux-vm")
print("python CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    p=torch.cuda.get_device_properties(0)
    print("vram GiB:", round(p.total_memory/1024**3,2))
print("ffmpeg:", shutil.which("ffmpeg"))
print("recommended amp:", "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else ("fp16" if torch.cuda.is_available() else "off"))
