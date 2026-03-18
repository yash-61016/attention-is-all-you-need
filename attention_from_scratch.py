"""
Attention Is All You Need — From-Scratch Implementation
========================================================
Paper: Attention Is All You Need (arXiv: 1706.03762)
What this implements: Scaled dot-product attention and multi-head attention
                      with explicit QKV projections. No library hides the mechanism.
Hardware: RTX 3090, 24GB VRAM (also runs on CPU)
Time to run: ~2 minutes (includes VRAM scaling experiment across 5 sequence lengths)

Dependencies (pinned):
    torch==2.1.0          # pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    matplotlib==3.7.1     # pip install matplotlib==3.7.1
    numpy==1.24.3         # pip install numpy==1.24.3

Run: python attention_from_scratch.py
"""

import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Chart style
# ---------------------------------------------------------------------------

PALETTE = {
    "navy":      "#0D1B2A",
    "dark_blue": "#1B263B",
    "steel":     "#415A77",
    "slate":     "#778DA9",
    "bone":      "#E0E1DD",
}
EXTENDED = {
    "light_blue": "#A3B8C8",
    "success":    "#6B9E7D",
}
CMAP_DEEP_SEA_LIGHT = LinearSegmentedColormap.from_list(
    "deep_sea_light",
    ["#FFFFFF", PALETTE["bone"], EXTENDED["light_blue"], PALETTE["slate"], PALETTE["steel"]],
    N=256,
)
plt.colormaps.register(CMAP_DEEP_SEA_LIGHT, name="deep_sea_light", force=True)

def annotate_points(ax, x_vals, y_vals, fmt="{:.2f}", offset=(0, 10), fontsize=8.5, color=None):
    c = color or PALETTE["dark_blue"]
    for xv, yv in zip(x_vals, y_vals):
        ax.annotate(fmt.format(yv), xy=(xv, yv), xytext=offset,
                    textcoords="offset points", ha="center", fontsize=fontsize,
                    color=c, weight="500")

def add_source_label(ax, text="yashpatel.xyz", fontsize=7):
    ax.annotate(text, xy=(1, 0), xycoords="axes fraction",
                xytext=(-8, -28), textcoords="offset points",
                ha="right", fontsize=fontsize, color=PALETTE["slate"], alpha=0.5, style="italic")

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.alpha":        0.5,
})

# ---------------------------------------------------------------------------
# Section 1: Setup — hardware check and config
# ---------------------------------------------------------------------------

# Paper defaults (Table 3, base model)
D_MODEL   = 512   # embedding dimension — every vector in the model is this size
NUM_HEADS = 8     # number of parallel attention heads
D_K       = D_MODEL // NUM_HEADS   # 64 — dimension per head for Q and K
D_V       = D_MODEL // NUM_HEADS   # 64 — dimension per head for V (same as d_k here)
DROPOUT   = 0.1

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {vram_gb:.1f} GB")
    assert vram_gb >= 2, f"Need >=2 GB VRAM, have {vram_gb:.1f} GB"
else:
    DEVICE = torch.device("cpu")
    print("No GPU found — running on CPU (VRAM measurements will show 0.0 GB)")

print(f"Device : {DEVICE}")
print(f"Config : d_model={D_MODEL}, num_heads={NUM_HEADS}, d_k={D_K}\n")


# ---------------------------------------------------------------------------
# Section 2: The mechanism
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    """
    Implements: Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V
                                     ───────────────────────────
                                     This is Equation 1 in the paper.

    Why scale by sqrt(d_k)?
        When d_k is large, the dot product Q·K^T grows in magnitude proportionally
        to sqrt(d_k). Without scaling, large values push softmax into its flat region
        where gradients vanish. Dividing by sqrt(d_k) keeps the input variance ~1.
        See paper footnote 4 (page 4).

    Why -inf masking instead of clamping after softmax?
        After softmax: exp(-inf) = 0 exactly. This is numerically clean.
        Clamping attention weights *after* softmax to 0 would break the row-sum=1
        property, which would distort the weighted average of values.
    """

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query : torch.Tensor,                # shape: (batch, heads, seq_len, d_k)
        key   : torch.Tensor,                # shape: (batch, heads, seq_len, d_k)
        value : torch.Tensor,                # shape: (batch, heads, seq_len, d_v)
        mask  : Optional[torch.Tensor] = None  # shape: (batch, 1, seq_len, seq_len) — bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Returns:
            output           : (batch, heads, seq_len, d_v) — context vectors
            attention_weights: (batch, heads, seq_len, seq_len) — for visualisation
        """

        d_k = query.size(-1)  # 64 per head

        # scores[b, h, i, j] = dot(query_i, key_j): how much should token i attend to token j?
        # key.transpose(-2, -1): (batch, heads, seq, d_k) → (batch, heads, d_k, seq)
        # matmul: (batch, heads, seq, d_k) @ (batch, heads, d_k, seq) → (batch, heads, seq, seq)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # Without scaling: d_k=64 inflates dot products ~8×, softmax saturates, gradients vanish.

        if mask is not None:
            # -inf BEFORE softmax: exp(-inf) = 0 exactly, preserving row-sum = 1.
            # Clamping to 0 after softmax would break the weighted-average property.
            scores = scores.masked_fill(mask == False, float('-inf'))
            # 5-token causal mask after fill:
            #   token 0: [  s00, -inf, -inf, -inf, -inf ]  ← only sees itself
            #   token 1: [  s10,  s11, -inf, -inf, -inf ]
            #   token 2: [  s20,  s21,  s22, -inf, -inf ]

        attention_weights = torch.softmax(scores, dim=-1)  # -inf positions become exactly 0
        attention_weights = self.dropout(attention_weights)

        # attention_weights : (batch, heads, seq, seq)
        # value             : (batch, heads, seq, d_k)
        # output            : (batch, heads, seq, d_k) — weighted blend of value vectors
        output = torch.matmul(attention_weights, value)

        return output, attention_weights  # weights returned for visualisation only


class MultiHeadAttention(nn.Module):
    """
    Wraps ScaledDotProductAttention with h parallel heads.

    Key idea: instead of one attention pass over d_model=512 dimensions,
    we run h=8 independent passes over d_k=64 dimensions each.
    Each head can learn a different relationship (syntactic, semantic, positional).
    Outputs are concatenated and projected back to d_model.

    Paper equation:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
        where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)

    W_Q_i, W_K_i, W_V_i have shape (d_model, d_k) per head.
    We implement this efficiently by using a single (d_model, d_model) linear
    and then reshaping — the math is equivalent.
    """

    def __init__(self, d_model: int = D_MODEL, num_heads: int = NUM_HEADS, dropout: float = DROPOUT):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads  # 64

        # THREE separate projection matrices — this is the point.
        # Each linear learns a completely different transformation of the input.
        # W_q asks: "what am I looking for?"
        # W_k asks: "what do I advertise to others?"
        # W_v asks: "what do I actually contribute if selected?"
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # (512 → 512), then split into 8×64
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection: after concatenating 8 heads of size 64 → project back to 512
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        query : torch.Tensor,                # (batch, seq_len, d_model)
        key   : torch.Tensor,                # (batch, seq_len, d_model)
        value : torch.Tensor,                # (batch, seq_len, d_model)
        mask  : Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = query.size(0)
        seq_len    = query.size(1)

        # ── 1. Project inputs with three separate weight matrices ──────────────
        # All three start at (batch, seq_len, d_model) = (1, 10, 512) for the concrete example
        Q = self.W_q(query)   # query asks: "what am I looking for?"
        K = self.W_k(key)     # key  asks:  "what do I offer to others?"
        V = self.W_v(value)   # value holds: "what I actually contribute"

        # ── 2. Split d_model into num_heads separate d_k-sized heads ──────────
        # Reshape: (batch, seq_len, d_model) → (batch, seq_len, heads, d_k)
        # Transpose to: (batch, heads, seq_len, d_k) so attention operates per head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V are now: (batch=1, heads=8, seq_len=9, d_k=64)

        # ── 3. Run scaled dot-product attention on all heads simultaneously ────
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # attn_output : (batch, heads, seq_len, d_k)
        # attn_weights: (batch, heads, seq_len, seq_len)

        # ── 4. Concatenate heads back into d_model ─────────────────────────────
        # Transpose back: (batch, heads, seq_len, d_k) → (batch, seq_len, heads, d_k)
        # .contiguous() ensures memory is laid out correctly for .view()
        # .view(-1, d_model) merges heads×d_k back to 512
        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        # attn_output now: (batch, seq_len, d_model) = (1, 9, 512)

        # ── 5. Final output projection ─────────────────────────────────────────
        output = self.W_o(attn_output)  # (1, 9, 512) — mixes information across heads

        return output, attn_weights


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build the autoregressive (causal) mask for decoder self-attention.

    Returns a lower-triangular boolean tensor:
        True  = this position CAN attend (present or past token)
        False = this position CANNOT attend (future token → becomes -inf)

    Shape: (1, 1, seq_len, seq_len) — broadcast across batch and head dims.

    Example for seq_len=4:
        [[[ True, False, False, False],
          [ True,  True, False, False],
          [ True,  True,  True, False],
          [ True,  True,  True,  True]]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


# ---------------------------------------------------------------------------
# Section 3: Run on a concrete example
# ---------------------------------------------------------------------------

def run_concrete_example():
    """
    Run attention on an actual English sentence so the data flow is legible.
    Prints intermediate tensor shapes at each stage.
    """
    print("=" * 70)
    print("SECTION 3 — CONCRETE EXAMPLE: actual English sentence")
    print("=" * 70)

    # Use recognisable tokens — not random tensors
    sentence   = "the cat sat on the mat because it was tired"
    tokens     = sentence.split()
    # Build minimal vocabulary: sort alphabetically so IDs are deterministic
    vocab      = {word: idx for idx, word in enumerate(sorted(set(tokens)))}
    token_ids  = torch.tensor([vocab[t] for t in tokens], device=DEVICE)  # (10,)
    seq_len    = len(tokens)

    print(f"\nInput sentence : '{sentence}'")
    print(f"Tokens         : {tokens}")
    print(f"Token IDs      : {token_ids.tolist()}")
    print(f"Vocab size     : {len(vocab)}")
    print(f"Sequence length: {seq_len}")

    # Embed tokens: lookup table maps each integer ID → 512-float vector
    # (In a real transformer this is trained; here random init is fine for shape demo)
    embedding     = nn.Embedding(len(vocab), D_MODEL).to(DEVICE)
    x             = embedding(token_ids).unsqueeze(0)  # (1, seq_len, 512)
    print(f"\nAfter embedding : {x.shape}  (batch=1, seq={seq_len}, d_model={D_MODEL})")

    # Build model and mask
    mha  = MultiHeadAttention().to(DEVICE)
    mha.eval()  # eval mode disables dropout — attention weights will sum to exactly 1.0
    mask = create_causal_mask(seq_len, DEVICE)
    print(f"Causal mask     : {mask.shape}")
    print(f"Mask (first 5 rows/cols):\n{mask[0, 0, :5, :5].int()}")

    # Forward pass
    print(f"\n{'─' * 50}")
    print("Running forward pass …")
    start      = time.time()
    output, attn_weights = mha(x, x, x, mask)  # self-attention: Q=K=V=x
    elapsed    = time.time() - start
    print(f"Forward pass : {elapsed*1000:.2f} ms")

    print(f"\nOutput shape          : {output.shape}")        # (1, seq_len, 512)
    print(f"Attention weights     : {attn_weights.shape}")   # (1, 8, seq_len, seq_len)

    # Sanity check: each row of attention weights must sum to 1.0
    row_sums = attn_weights[0, 0].sum(dim=-1)  # sum over key dimension, head 0
    print(f"\nAttention weight row sums (head 0) — should all be ≈1.0:")
    print(f"  {row_sums.detach().cpu().numpy()}")

    # Show which tokens token "it" (index 4) attends to — causal: only past tokens
    it_idx = tokens.index("it")
    print(f"\nToken '{tokens[it_idx]}' (pos {it_idx}) attends to (head 0):")
    weights = attn_weights[0, 0, it_idx].detach().cpu().numpy()
    for tok, w in zip(tokens, weights):
        bar = "█" * int(w * 30)
        print(f"  {tok:10s}  {w:.4f}  {bar}")

    return tokens, attn_weights


# ---------------------------------------------------------------------------
# Section 4: Verification — compare against PyTorch reference
# ---------------------------------------------------------------------------

def verify_against_reference():
    """
    Checks that the from-scratch implementation matches F.scaled_dot_product_attention
    within floating-point tolerance. A correct implementation shows max_diff < 1e-5.
    """
    print("\n" + "=" * 70)
    print("SECTION 4 — VERIFICATION: compare against F.scaled_dot_product_attention")
    print("=" * 70)

    torch.manual_seed(42)
    batch, heads, seq, dk = 1, 8, 9, 64

    Q = torch.randn(batch, heads, seq, dk, device=DEVICE)
    K = torch.randn(batch, heads, seq, dk, device=DEVICE)
    V = torch.randn(batch, heads, seq, dk, device=DEVICE)

    # Causal mask — boolean, shape (1, 1, seq, seq)
    mask = create_causal_mask(seq, DEVICE)

    # ── Your implementation ──────────────────────────────────────────────────
    attn_module   = ScaledDotProductAttention(dropout=0.0)  # dropout=0 for deterministic comparison
    our_out, _ = attn_module(Q, K, V, mask)

    # ── PyTorch reference (requires torch >= 2.0) ────────────────────────────
    # F.scaled_dot_product_attention expects is_causal=True or an additive mask.
    # We convert our boolean mask to an additive mask for apples-to-apples comparison.
    additive_mask = torch.zeros(1, 1, seq, seq, device=DEVICE)
    additive_mask = additive_mask.masked_fill(mask == False, float('-inf'))

    with torch.no_grad():
        ref_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=additive_mask, dropout_p=0.0)

    if our_out is not None:
        max_diff = (our_out - ref_out).abs().max().item()
        print(f"\nMax absolute difference vs PyTorch reference : {max_diff:.2e}")
        if max_diff < 1e-4:
            print("PASS — implementation matches reference")
        else:
            print("FAIL — outputs diverge; check scaling and mask application")
    else:
        print("(Skipping verification — forward() returned None)")


# ---------------------------------------------------------------------------
# Section 5: Visualisation — VRAM scaling + attention heatmap
# ---------------------------------------------------------------------------

def measure_peak_vram(model: nn.Module, seq_len: int) -> float:
    """
    Measure peak VRAM consumed by one forward pass at a given sequence length.

    Uses reset_peak_memory_stats() + max_memory_allocated() — more reliable than
    measuring memory_allocated() delta, which can be confounded by the caching allocator.

    Returns peak VRAM in GB.
    """
    if not torch.cuda.is_available():
        return 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    x    = torch.randn(1, seq_len, D_MODEL, device=DEVICE)
    mask = create_causal_mask(seq_len, DEVICE)

    with torch.no_grad():
        _ = model(x, x, x, mask)

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / 1e9  # bytes → GB


def plot_vram_quadratic(seq_lengths: list, vram_gb: list, save_path: str):
    """
    Plot measured VRAM vs sequence length alongside a quadratic fit.
    The attention score matrix is O(n²): (batch × heads × n × n × 4 bytes).

    For reference, theoretical score matrix size at each length:
        n=512  → 1 × 8 × 512  × 512  × 4B =  8.4 MB
        n=1024 → 1 × 8 × 1024 × 1024 × 4B = 33.6 MB
        n=2048 → 1 × 8 × 2048 × 2048 × 4B = 134 MB
        n=4096 → 1 × 8 × 4096 × 4096 × 4B = 536 MB  (Flash Attention was invented for this)
    """
    # Convert GB → MB so labels are distinct even at small model scale
    # (toy MHA model uses ~20–130 MB, not GB — GB format rounds everything to "0.02 GB")
    vram_mb = [v * 1024 for v in vram_gb]

    _, ax = plt.subplots(figsize=(9, 5))

    ax.plot(seq_lengths, vram_mb, 'o-',
            color=PALETTE['steel'], linewidth=2.2, markersize=7,
            markeredgecolor='white', markeredgewidth=1.5,
            label='Measured peak VRAM', zorder=5)

    # Quadratic reference fit through measured points
    coeffs   = np.polyfit(seq_lengths, vram_mb, 2)
    x_smooth = np.linspace(min(seq_lengths), max(seq_lengths), 300)
    ax.plot(x_smooth, np.polyval(coeffs, x_smooth), '--',
            color=PALETTE['slate'], alpha=0.6, linewidth=1.5,
            label='O(n²) quadratic fit')
    ax.fill_between(x_smooth, 0, np.polyval(coeffs, x_smooth),
                    color=PALETTE['steel'], alpha=0.04)

    annotate_points(ax, seq_lengths, vram_mb, fmt="{:.0f} MB", offset=(0, 12))

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Peak VRAM (MB)')
    ax.set_title('Multi-Head Attention — VRAM Scales as O(n²)')
    ax.legend(loc='upper left')
    add_source_label(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_attention_heatmap(tokens: list, attn_weights: torch.Tensor, save_path: str):
    """
    Visualise attention weights for head 0 as a heatmap.
    The lower-triangular structure makes the causal masking visible.
    """
    weights = attn_weights[0, 0].detach().cpu().numpy()  # head 0, shape (seq, seq)

    _, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(weights, cmap=CMAP_DEEP_SEA_LIGHT, vmin=0, vmax=weights.max(), aspect='auto')

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)

    # Cell separators — linewidth=0.5 to stay visible at dpi=200
    for i in range(len(tokens)):
        ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.6)
        ax.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.6)

    ax.set_xlabel('Key (token being attended to)')
    ax.set_ylabel('Query (token doing the attending)')
    ax.set_title('Attention Weights — Head 0 (causal mask visible)\n'
                 'Upper triangle = -inf before softmax → 0 after')

    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.ax.set_ylabel('Attention weight', fontsize=9, color=PALETTE['steel'])
    cbar.ax.tick_params(labelsize=8, colors=PALETTE['steel'])
    cbar.outline.set_edgecolor(PALETTE['bone'])
    cbar.outline.set_linewidth(0.5)

    # Watermark — y-offset increased to -36 so it clears the rotated x-axis labels
    ax.annotate(
        "yashpatel.xyz",
        xy=(1, 0), xycoords="axes fraction",
        xytext=(-8, -36), textcoords="offset points",
        ha="right", fontsize=7, color=PALETTE['slate'], alpha=0.5, style="italic",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {save_path}")


def run_vram_experiment(model: nn.Module):
    """
    Run forward passes at increasing sequence lengths and record peak VRAM.
    Demonstrates the quadratic growth that motivated Flash Attention (2022).
    """
    print("\n" + "=" * 70)
    print("SECTION 5 — VISUALISATION: VRAM scaling + attention heatmap")
    print("=" * 70)

    seq_lengths = [64, 128, 256, 512, 1024]
    vram_values = []

    print(f"\n{'Seq length':>12}  {'Peak VRAM (GB)':>16}  {'Score matrix size':>18}")
    print("─" * 52)

    for n in seq_lengths:
        # Theoretical score matrix: batch × heads × n × n × 4 bytes (float32)
        theoretical_mb = (1 * NUM_HEADS * n * n * 4) / 1e6
        peak_gb        = measure_peak_vram(model, n)
        vram_values.append(peak_gb)
        print(f"{n:>12}  {peak_gb:>14.4f} GB  {theoretical_mb:>14.1f} MB score matrix")

    plot_vram_quadratic(seq_lengths, vram_values,
                        save_path="attention_vram_quadratic.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Section 3: concrete example with real English text
    tokens, attn_weights = run_concrete_example()

    # Section 4: verify the implementation is numerically correct
    verify_against_reference()

    # Section 5: VRAM scaling plot
    mha = MultiHeadAttention().to(DEVICE)
    run_vram_experiment(mha)

    # Attention heatmap
    if attn_weights is not None:
        plot_attention_heatmap(tokens, attn_weights,
                               save_path="attention_heatmap.png")

    print("\n" + "=" * 70)
    print("DONE")
    print("Outputs: attention_vram_quadratic.png, attention_heatmap.png")
    print("=" * 70)
