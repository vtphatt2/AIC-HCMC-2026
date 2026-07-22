# Bao Cao Topic-Temporal Sandbox

> **Nguon:** Tu dong tu notebook 11_topic_aware_temporal_fusion_eda.ipynb
> **Muc tieu:** Calibrate 3 advanced heuristics: temporal decay, topic-adaptive weights, chunk length.

## 1. Temporal Bounds (Section 1)

| Parameter | Value | Source |
|-----------|-------|--------|
| Lambda (exp) | 0.357 | notebook 05 |
| Half-life | 1.95s | computed |
| Cutoff | 8.4s | 3/lambda |
| Gaussian sigma | 1.656s | from half-life |

**Recommendation:** Gaussian adaptive (sigma=1.5s) for tolerance zone preservation.

## 2. Topic-Adaptive Weights (Section 2)

| Genre Type | W_visual | W_ocr | W_transcript |
|------------|----------|-------|--------------|
| Abstract | 0.35 | 0.35 | 0.30 |
| Concrete | 0.50 | 0.25 | 0.25 |

Abstract genres: Doi song, Kinh te, Moi truong, Phap luat, Thoi su, Van hoa
Concrete genres: Am thuc, Du lich, Giai tri, Giao thong, The thao

## 3. Chunk Length Sweet Spot (Section 3)

**Optimal: 35 words per chunk**
**Recommended range: 20-50 words**
**Window overlap: 5-10 words**

## 4. Production Deployment

- **Temporal:** Gaussian adaptive decay in `stable_fusion.py`
- **Topic:** Genre-filter -> dynamic weight matrix
- **Chunk:** chunk_size=35, overlap=5 in `transcript_search.py`
