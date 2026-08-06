"""Split a directory of images into N tar packets for pipelined transfer:
send packet i while Colab is already unpacking+embedding packet i-1, instead
of waiting for one giant transfer to finish before any processing starts.
"""
import tarfile
import time
from pathlib import Path

SRC_DIR = Path(r"C:\Users\duyla\AppData\Local\Temp\claude\D--Coding-side-project-AIC-2026-Data-Processing\bd97fb00-842e-49b0-8536-a82c89db6f6b\scratchpad\local_extract_named")
PACKET_DIR = Path(r"C:\Users\duyla\AppData\Local\Temp\claude\D--Coding-side-project-AIC-2026-Data-Processing\bd97fb00-842e-49b0-8536-a82c89db6f6b\scratchpad\packets")
PACKET_SIZE = 50

PACKET_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(SRC_DIR.glob("*.jpg"))
packets = [images[i : i + PACKET_SIZE] for i in range(0, len(images), PACKET_SIZE)]

t0 = time.perf_counter()
manifest = {"n_images": len(images), "n_packets": len(packets), "packet_size": PACKET_SIZE, "packets": []}
for i, packet_images in enumerate(packets):
    packet_path = PACKET_DIR / f"packet_{i:03d}.tar"
    with tarfile.open(packet_path, "w") as tar:
        for img_path in packet_images:
            tar.add(img_path, arcname=img_path.name)
    manifest["packets"].append({"index": i, "file": packet_path.name, "n_images": len(packet_images)})
elapsed = time.perf_counter() - t0

import json
(PACKET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

total_size = sum(p.stat().st_size for p in PACKET_DIR.glob("packet_*.tar"))
print(f"packed {len(images)} images into {len(packets)} packets in {elapsed:.2f}s, total_size={total_size/1e6:.1f}MB")
