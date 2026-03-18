# Attention Is All You Need — Artifacts

From-scratch implementation and head-analysis artifact for [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

Accompanies the blog post: **[Attention Is All You Need: Early Layers Diffuse, Late Layers Precise, the Head Entropy Gradient the Paper Never Measured](https://yashpatel.xyz)**

---

## Files

| File | Description |
|------|-------------|
| `attention_from_scratch.py` | Scaled dot-product attention and multi-head attention in ~200 lines of pure PyTorch, verified against `F.scaled_dot_product_attention`. Includes VRAM scaling experiment across 5 sequence lengths. |
| `attention_head_analysis.py` | Loads GPT-2 small (117M params) and classifies all 144 heads (12 layers × 12 heads) by Shannon entropy and diagonal locality score across 100 English sentences. |

## Hardware

RTX 3090, 24GB VRAM. Both scripts also run on CPU.

- `attention_from_scratch.py` uses < 1GB VRAM even at seq_len=1024
- `attention_head_analysis.py` uses ~600MB VRAM for GPT-2 small

## Setup

```bash
pip install -r requirements.txt
# For GPU support (CUDA 12.1):
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

## Run

```bash
python attention_from_scratch.py   # ~2 minutes
python attention_head_analysis.py  # ~5 minutes
```

## Outputs

**`attention_from_scratch.py`**
- `attention_vram_quadratic.png` — peak VRAM vs sequence length with O(n²) quadratic fit
- `attention_heatmap.png` — attention weights for head 0 on a 10-token English sentence

**`attention_head_analysis.py`**
- `head_entropy_heatmap.png` — 12×12 heatmap of mean Shannon entropy per head
- `head_layer_depth_gradient.png` — entropy gradient by layer depth + head type distribution
- `head_type_examples.png` — one representative attention matrix per head type (copy, local, broad, mixed)

---

## Key Finding

Shannon entropy falls monotonically with layer depth in GPT-2 small. Early layers (0–3) have mean entropy ~1.42 nats; late layers (8–11) have mean entropy ~0.50 nats. The two populations are nearly 3× apart — a signal the original paper never quantified.

```
KEY FINDING: layer-depth entropy gradient
  Early layers (0-3)  mean entropy : 1.421 nats
  Late  layers (8-11) mean entropy : 0.497 nats
  Gradient            (late-early) : -0.924 nats
```
