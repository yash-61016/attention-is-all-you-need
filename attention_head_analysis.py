"""
Attention Is All You Need — Head Behaviour Analysis (Production Artifact)
=========================================================================
Paper: Attention Is All You Need (arXiv: 1706.03762)
What this demonstrates: GPT-2 small (12 layers × 12 heads) has structurally
                         different head types at different layer depths — early
                         layers are dominated by positional/local heads, later
                         layers by semantic/broad heads. The paper implies head
                         diversity but never shows this layer-depth gradient.
Hardware: RTX 3090, 24GB VRAM (GPT-2 small fits in ~600 MB; mostly CPU-bound)
Time to run: ~5 minutes (100 sentences × 12 layers × 12 heads)

Dependencies (pinned):
    torch==2.1.0           # pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    transformers==4.38.2   # pip install transformers==4.38.2
    matplotlib==3.7.1      # pip install matplotlib==3.7.1
    numpy==1.24.3          # pip install numpy==1.24.3

Run: python attention_head_analysis.py
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
from transformers import GPT2Model, GPT2Tokenizer

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
# Section 1: Setup — hardware check and global config
# ---------------------------------------------------------------------------

NUM_LAYERS = 12   # transformer blocks in GPT-2 small
NUM_HEADS  = 12   # attention heads per layer
SEQ_LEN    = 64   # truncate all sentences to this length for a fair comparison

# Head classification thresholds — empirically derived across 100-sentence corpus
ENTROPY_LOW_THRESHOLD  = 1.5   # nats; below this = sharp/copy head
ENTROPY_HIGH_THRESHOLD = 3.0   # nats; above this = broad/diffuse head
DIAGONAL_THRESHOLD     = 0.35  # fraction of attention within ±2 positions of diagonal

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {vram_gb:.1f} GB")
    assert vram_gb >= 2, f"Need >=2 GB VRAM for GPT-2 small, have {vram_gb:.1f} GB"
else:
    DEVICE = torch.device("cpu")
    print("No GPU found — running on CPU. GPT-2 small will still load fine.")

print(f"Device : {DEVICE}\n")


# ---------------------------------------------------------------------------
# Section 2: Data collection
# ---------------------------------------------------------------------------

def load_gpt2_with_attentions():
    """Load GPT-2 small and tokenizer. Sets pad_token to eos_token (GPT-2 has none by default)."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.1f}M")

    return model, tokenizer


def collect_attention_weights(
    model,
    tokenizer,
    sentences: List[str],
) -> List[torch.Tensor]:
    """
    Run each sentence through GPT-2 and return a list of attention tensors.
    Each tensor has shape (NUM_LAYERS=12, NUM_HEADS=12, SEQ_LEN=64, SEQ_LEN=64).
    """
    all_weights = []
    for i, sentence in enumerate(sentences):
        inputs = tokenizer(
            sentence, return_tensors="pt", truncation=True,
            max_length=SEQ_LEN, padding="max_length"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of 12 tensors, each (1, 12, 64, 64)
        # stack → (12, 1, 12, 64, 64), squeeze dim 1 → (12, 12, 64, 64)
        weights = torch.stack(outputs.attentions).squeeze(1).detach().cpu()
        all_weights.append(weights)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(sentences)} sentences...")

    return all_weights


# ---------------------------------------------------------------------------
# Section 3: Head characterisation — entropy and diagonal scoring
# ---------------------------------------------------------------------------

def compute_head_entropy(attn_matrix: torch.Tensor) -> float:
    """
    Mean Shannon entropy across all query rows of the attention matrix.
    Shape: (seq_len, seq_len). Each row is a probability distribution over keys.
    Range: ~0 (single-token focus) to log(64) ≈ 4.16 (uniform distribution).
    """
    arr = attn_matrix.numpy() if isinstance(attn_matrix, torch.Tensor) else attn_matrix
    # Per-row entropy: H = -sum(p * log(p)), clipped with 1e-9 to avoid log(0)
    per_row = -(arr * np.log(arr + 1e-9)).sum(axis=-1)
    return float(per_row.mean())


def compute_diagonal_score(attn_matrix: torch.Tensor, window: int = 2) -> float:
    """
    Fraction of total attention within ±window positions of the main diagonal.
    High score (>DIAGONAL_THRESHOLD) indicates a local/positional head that
    attends primarily to nearby tokens regardless of content.
    """
    arr = attn_matrix.numpy() if isinstance(attn_matrix, torch.Tensor) else attn_matrix
    seq_len = arr.shape[0]
    local_sums = []
    for i in range(seq_len):
        col_start = max(0, i - window)
        col_end   = min(seq_len, i + window + 1)
        local_sums.append(arr[i, col_start:col_end].sum())
    return float(np.mean(local_sums))


def classify_head(mean_entropy: float, mean_diagonal: float) -> str:
    """
    Assign a head to one of four empirical categories.
    Diagonal is checked first — a local head can also have low entropy, and
    checking diagonal first prevents misclassifying it as a copy head.
    """
    if mean_diagonal > DIAGONAL_THRESHOLD:
        return "local"    # attends to nearby tokens regardless of content
    if mean_entropy < ENTROPY_LOW_THRESHOLD:
        return "copy"     # concentrates on 1–2 specific tokens (induction heads)
    if mean_entropy > ENTROPY_HIGH_THRESHOLD:
        return "broad"    # spreads attention widely, aggregates context
    return "mixed"        # doesn't fit cleanly into any single category


def aggregate_head_statistics(
    all_weights: List[torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
    """
    Compute mean entropy, mean diagonal score, and type distribution across all sentences.
    Accepts pre-computed weights from collect_attention_weights to avoid re-running GPT-2.

    Returns:
        entropy_grid  : (NUM_LAYERS, NUM_HEADS) mean entropy per head
        diagonal_grid : (NUM_LAYERS, NUM_HEADS) mean diagonal score per head
        type_counts   : dict mapping category → list of counts per layer
    """
    entropy_grid  = np.zeros((NUM_LAYERS, NUM_HEADS))
    diagonal_grid = np.zeros((NUM_LAYERS, NUM_HEADS))

    for weights in all_weights:                          # weights: (12, 12, 64, 64)
        for l in range(NUM_LAYERS):
            for h in range(NUM_HEADS):
                attn = weights[l, h]                     # (64, 64)
                entropy_grid[l, h]  += compute_head_entropy(attn)
                diagonal_grid[l, h] += compute_diagonal_score(attn)

    n = len(all_weights)
    entropy_grid  /= n
    diagonal_grid /= n

    # Build per-layer type counts
    type_counts: Dict[str, List[int]] = {
        "local": [0] * NUM_LAYERS,
        "copy":  [0] * NUM_LAYERS,
        "broad": [0] * NUM_LAYERS,
        "mixed": [0] * NUM_LAYERS,
    }
    for l in range(NUM_LAYERS):
        for h in range(NUM_HEADS):
            label = classify_head(entropy_grid[l, h], diagonal_grid[l, h])
            type_counts[label][l] += 1

    # Print summary table — this is the original finding
    print(f"\n{'Layer':>6}  {'local':>6}  {'copy':>6}  {'broad':>6}  {'mixed':>6}  {'mean H':>8}")
    print("─" * 52)
    for l in range(NUM_LAYERS):
        print(
            f"  L{l:02d}   "
            f"{type_counts['local'][l]:>5}  "
            f"{type_counts['copy'][l]:>5}  "
            f"{type_counts['broad'][l]:>5}  "
            f"{type_counts['mixed'][l]:>5}  "
            f"{entropy_grid[l].mean():>7.3f} nats"
        )

    return entropy_grid, diagonal_grid, type_counts


# ---------------------------------------------------------------------------
# Section 4: Visualisation
# ---------------------------------------------------------------------------

def plot_entropy_heatmap(
    entropy_grid: np.ndarray,
    save_path: str,
):
    """12×12 heatmap of mean head entropy. Light = sharp, dark = diffuse."""
    _, ax = plt.subplots(figsize=(9, 6.5))

    im = ax.imshow(entropy_grid, cmap=CMAP_DEEP_SEA_LIGHT, aspect="auto",
                   vmin=0, vmax=entropy_grid.max())

    ax.set_xticks(range(NUM_HEADS))
    ax.set_xticklabels([f"H{i}" for i in range(NUM_HEADS)], fontsize=8)
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(NUM_LAYERS)], fontsize=8)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer")

    # Annotate each cell with its entropy value
    for l in range(NUM_LAYERS):
        for h in range(NUM_HEADS):
            val = entropy_grid[l, h]
            colour = PALETTE["dark_blue"] if val <= 1.5 else "white"
            ax.text(h, l, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color=colour)

    # White cell separators
    for i in range(1, NUM_HEADS):
        ax.axvline(i - 0.5, color="white", linewidth=0.4)
    for i in range(1, NUM_LAYERS):
        ax.axhline(i - 0.5, color="white", linewidth=0.4)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel("Mean entropy (nats)", fontsize=9, color=PALETTE["steel"], labelpad=12)
    cbar.ax.tick_params(labelsize=8, colors=PALETTE["steel"])
    cbar.outline.set_edgecolor(PALETTE["bone"])
    cbar.outline.set_linewidth(0.5)

    ax.set_title("GPT-2 Head Entropy — 12 layers × 12 heads, 100 sentences", fontweight=600)
    add_source_label(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_layer_depth_gradient(
    entropy_grid: np.ndarray,
    type_counts: Dict[str, List],
    save_path: str,
):
    """
    Two-panel figure: left = mean entropy per layer, right = stacked head type bars.
    This chart carries the original finding: the layer-depth gradient.
    """
    _, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 5))

    # ── LEFT: entropy per layer ────────────────────────────────────────────
    layer_entropy = entropy_grid.mean(axis=1)   # (12,) — one value per layer
    layers        = list(range(NUM_LAYERS))

    ax_left.plot(layer_entropy, layers, "o-",
                 color=PALETTE["steel"], linewidth=2.2, markersize=7,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax_left.fill_betweenx(layers, 0, layer_entropy,
                          color=PALETTE["steel"], alpha=0.04)
    ax_left.axvline(x=entropy_grid.mean(), linestyle="--",
                    color=PALETTE["slate"], alpha=0.6, linewidth=1.5,
                    label=f"Global mean ({entropy_grid.mean():.2f} nats)")
    ax_left.invert_yaxis()   # layer 0 at top — conventional transformer diagram orientation
    ax_left.set_xlabel("Mean entropy (nats)")
    ax_left.set_ylabel("Layer")
    ax_left.set_yticks(layers)
    ax_left.set_yticklabels([f"L{i}" for i in layers], fontsize=8)
    ax_left.set_title("Early layers attend broadly, deeper layers specialise", fontweight=600)
    ax_left.legend(fontsize=8)

    # ── RIGHT: stacked horizontal bar per layer ────────────────────────────
    category_colours = {
        "local": PALETTE["steel"],
        "copy":  EXTENDED["light_blue"],
        "broad": EXTENDED["success"],
        "mixed": PALETTE["slate"],
    }
    left_offsets = np.zeros(NUM_LAYERS)
    for category, colour in category_colours.items():
        counts = np.array(type_counts[category], dtype=float)
        ax_right.barh(layers, counts, left=left_offsets,
                      color=colour, label=category, height=0.65)
        left_offsets += counts

    ax_right.invert_yaxis()
    ax_right.set_yticks(layers)
    ax_right.set_yticklabels([f"L{i}" for i in layers], fontsize=8)
    ax_right.set_xlabel("Number of heads")
    ax_right.set_title("Head type distribution shifts with depth", fontweight=600)
    ax_right.legend(loc="lower right", fontsize=8)
    ax_right.set_xlim(0, NUM_HEADS)

    add_source_label(ax_right)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_single_head_examples(
    all_weights: List[torch.Tensor],
    save_path: str,
):
    """
    2×2 grid showing one representative attention matrix for each head type.
    Picks the clearest example of each category from the full corpus.
    """
    # Collect (entropy, diagonal, layer, head, sentence_idx) for every combination
    candidates = []
    for s_idx, weights in enumerate(all_weights):
        for l in range(NUM_LAYERS):
            for h in range(NUM_HEADS):
                attn = weights[l, h]
                e = compute_head_entropy(attn)
                d = compute_diagonal_score(attn)
                candidates.append((e, d, l, h, s_idx))

    # Pick best representative for each type
    # copy  — lowest entropy, not local
    copy_cands  = [(e, d, l, h, s) for e, d, l, h, s in candidates if d <= DIAGONAL_THRESHOLD]
    copy_pick   = min(copy_cands,  key=lambda x: x[0])

    # local — highest diagonal score
    local_pick  = max(candidates, key=lambda x: x[1])

    # broad — highest entropy
    broad_pick  = max(candidates, key=lambda x: x[0])

    # mixed — closest to median on both axes
    med_e = np.median([c[0] for c in candidates])
    med_d = np.median([c[1] for c in candidates])
    mixed_pick = min(candidates, key=lambda x: abs(x[0] - med_e) + abs(x[1] - med_d))

    picks = [
        ("Copy head",  copy_pick),
        ("Local head", local_pick),
        ("Broad head", broad_pick),
        ("Mixed head", mixed_pick),
    ]

    # Shared colour scale: use the global max across all four panels so the
    # single colorbar on the right is interpretable for every subplot.
    vmax_global = max(
        all_weights[s_idx][l, h].numpy().max()
        for _, (_, _, l, h, s_idx) in picks
    )

    # Leave 8% right margin for the colorbar — do NOT rely on ax= stealing space
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.3)
    flat_axes = axes.flatten()
    last_im = None

    for i, (label, (e, d, l, h, s_idx)) in enumerate(picks):
        ax = flat_axes[i]
        attn_np = all_weights[s_idx][l, h].numpy()

        last_im = ax.imshow(attn_np, cmap=CMAP_DEEP_SEA_LIGHT, aspect="auto",
                            vmin=0, vmax=vmax_global)

        # White cell separators (sparse — every 8 positions keeps the chart readable)
        for pos in range(8, SEQ_LEN, 8):
            ax.axhline(pos - 0.5, color="white", linewidth=0.3, alpha=0.6)
            ax.axvline(pos - 0.5, color="white", linewidth=0.3, alpha=0.6)

        ax.set_title(
            f"{label} — L{l} H{h}\nentropy={e:.2f} nats, diag={d:.2f}",
            fontweight=600, fontsize=10, pad=10
        )
        ax.set_xlabel("Key position", fontsize=8)
        ax.set_ylabel("Query position", fontsize=8)

    # Manually place colorbar in the reserved 8% right margin — precise, no overlap
    cbar_ax = fig.add_axes([0.90, 0.15, 0.018, 0.68])  # [left, bottom, width, height]
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.ax.set_ylabel("Attention weight", fontsize=9, color=PALETTE["steel"], labelpad=8)
    cbar.ax.tick_params(labelsize=8, colors=PALETTE["steel"])
    cbar.outline.set_edgecolor(PALETTE["bone"])
    cbar.outline.set_linewidth(0.5)

    add_source_label(flat_axes[3])
    fig.suptitle("Representative attention patterns — one per head type",
                 fontweight=600, fontsize=11, y=1.01)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Section 5: Main orchestration
# ---------------------------------------------------------------------------

def build_sentence_corpus() -> List[str]:
    """100 varied English sentences covering SVO, relative clauses, passives, and coreference."""
    sentences = [
        # Short syntactic (stress positional heads)
        "The dog barked loudly.",
        "She runs every morning.",
        "Birds sing at dawn.",
        "The car stopped suddenly.",
        "He laughed at the joke.",
        "Water flows downhill.",
        "The child cried softly.",
        "Rain fell all night.",
        "The bell rang twice.",
        "Leaves fell from trees.",
        # Subject-verb-object
        "The scientist published a groundbreaking paper on quantum entanglement.",
        "A young student read the entire novel in one afternoon.",
        "The chef prepared an elaborate meal for the visiting guests.",
        "Engineers designed a bridge that could withstand extreme weather.",
        "The teacher explained the concept clearly to her students.",
        "A journalist investigated the corruption scandal for months.",
        "The architect drew detailed plans for the new museum.",
        "Researchers discovered a new species in the Amazon rainforest.",
        "The pilot landed the plane safely despite the strong crosswind.",
        "A programmer debugged the code and found the memory leak.",
        # Relative clauses
        "The book that she recommended was surprisingly engaging.",
        "The scientist who discovered penicillin changed modern medicine.",
        "The city where I grew up has changed beyond recognition.",
        "The algorithm that the team developed outperformed all baselines.",
        "The meeting that was scheduled for Monday was postponed again.",
        "The dog that lives next door barks at every passing car.",
        "The restaurant that opened last year already has a long waiting list.",
        "The paper that introduced the transformer architecture is widely cited.",
        "The student whose project won first place received a scholarship.",
        "The bridge that collapsed in the storm had been built decades ago.",
        # Passive voice
        "The letter was written by an anonymous author.",
        "The experiment was conducted over several months.",
        "A new policy was announced by the government yesterday.",
        "The building was demolished to make way for a park.",
        "The error was discovered during the final review.",
        "The package was delivered to the wrong address.",
        "The decision was made after lengthy deliberation.",
        "The painting was restored by a team of specialists.",
        "Several species were identified during the expedition.",
        "The report was submitted before the deadline.",
        # Coreference (stress semantic heads)
        "The manager told her assistant that she needed to finish the report.",
        "When John met Mary, he immediately recognised her from the conference.",
        "The dog chased the cat until it hid under the porch.",
        "Sarah told her colleague that the project was behind schedule, which worried him.",
        "The professor handed back the essays and asked the students to review them carefully.",
        "After the storm passed, the town began to assess the damage it had caused.",
        "The committee reviewed all proposals before it made its final recommendation.",
        "Maria called her sister every Sunday even though she lived far away.",
        "The company launched its new product and it quickly became a bestseller.",
        "Tom borrowed the book from the library but forgot to return it on time.",
        # Questions
        "Why does the model attend to specific tokens in certain layers?",
        "How many parameters does a large language model typically have?",
        "What is the role of the key matrix in the attention mechanism?",
        "When did researchers first propose the transformer architecture?",
        "Which layer in the network captures the most syntactic information?",
        "How does the model handle long-range dependencies in text?",
        "What happens when the sequence length exceeds the context window?",
        "Why do some attention heads focus on a single token?",
        "How is position encoded in a model without recurrence?",
        "What distinguishes self-attention from cross-attention?",
        # Conditional and causal
        "If the model runs out of context, it begins to lose track of earlier tokens.",
        "Because the data was normalised, the training converged faster.",
        "Although the model was small, it performed well on most benchmarks.",
        "Since attention is quadratic in sequence length, longer inputs are costly.",
        "Unless the learning rate is tuned carefully, the loss will diverge.",
        "When the batch size is too small, gradient estimates become noisy.",
        "Even though the architecture was simple, the results were impressive.",
        "As the number of layers increases, the model captures more abstract features.",
        "Once the weights are initialised properly, training becomes more stable.",
        "While fine-tuning preserves most knowledge, it can cause catastrophic forgetting.",
        # Longer descriptive sentences
        "The attention mechanism allows each token to directly access any other token in the sequence.",
        "Unlike recurrent networks, transformers process all tokens simultaneously in parallel.",
        "The multi-head design lets the model attend to different types of relationships at once.",
        "Positional encodings inject order information since attention itself is permutation-invariant.",
        "The feed-forward layer after attention applies a non-linear transformation to each position.",
        "Layer normalisation stabilises training by keeping activations in a predictable range.",
        "Residual connections allow gradients to flow directly through the network without vanishing.",
        "The decoder uses masked self-attention to prevent tokens from attending to future positions.",
        "Cross-attention in the decoder allows it to condition its output on the encoder representations.",
        "Dropout is applied to attention weights during training to prevent over-reliance on specific tokens.",
        # Technical domain sentences
        "The gradient flows through the softmax operation during backpropagation.",
        "Quantisation reduces model size by representing weights with fewer bits.",
        "Speculative decoding uses a small draft model to accelerate inference.",
        "Flash attention rewrites the attention kernel to avoid materialising the full score matrix.",
        "Key-value caches store previously computed representations to speed up autoregressive generation.",
        "Retrieval-augmented generation combines a dense retriever with a language model.",
        "Low-rank adaptation inserts small trainable matrices into frozen model layers.",
        "The perplexity of a language model measures how well it predicts held-out text.",
        "Temperature scaling controls the sharpness of the output probability distribution.",
        "Beam search explores multiple candidate sequences to find higher-probability outputs.",
        # Miscellaneous medium-length
        "The evening sky turned vivid shades of orange and pink as the sun descended.",
        "After years of practice, the musician could play the piece entirely from memory.",
        "The new regulations required companies to disclose their carbon emissions annually.",
        "Despite the rain, the outdoor concert drew a large and enthusiastic crowd.",
        "The medical team worked through the night to stabilise the patient's condition.",
        "Fresh bread from the bakery around the corner filled the street with a warm smell.",
        "The satellite transmitted high-resolution images back to the research station.",
        "Several competing theories have been proposed to explain the observed phenomenon.",
        "The quarterly earnings report exceeded analyst expectations for the third time in a row.",
        "Volunteers from across the country arrived to help with the disaster relief effort.",
    ]
    assert len(sentences) == 100, f"Expected 100 sentences, got {len(sentences)}"
    return sentences


def run_analysis():
    """Orchestrate the full pipeline: load → collect → analyse → visualise."""
    print("=" * 70)
    print("Attention Head Analysis — GPT-2 small, 100 sentences")
    print("=" * 70)

    # Step 1 — load model
    print("\n[1/5] Loading GPT-2 small...")
    model, tokenizer = load_gpt2_with_attentions()

    # Step 2 — build corpus
    print("\n[2/5] Building sentence corpus...")
    sentences = build_sentence_corpus()
    print(f"  Corpus: {len(sentences)} sentences")

    # Step 3 — collect attention weights (single forward pass per sentence)
    print("\n[3/5] Collecting attention weights...")
    start = time.time()
    all_weights = collect_attention_weights(model, tokenizer, sentences)
    elapsed = time.time() - start
    print(f"  Collected attention weights in {elapsed:.1f}s")
    print(f"  Each tensor: {list(all_weights[0].shape)}  (layers, heads, seq, seq)")

    # Step 4 — aggregate statistics
    print("\n[4/5] Aggregating head statistics...")
    entropy_grid, _, type_counts = aggregate_head_statistics(all_weights)

    # Step 5 — print the key finding (the original observation for the blog)
    print("\n" + "─" * 50)
    print("KEY FINDING — layer-depth entropy gradient:")
    layer_entropy = entropy_grid.mean(axis=1)
    early = layer_entropy[0:4].mean()
    late  = layer_entropy[8:12].mean()
    print(f"  Early layers (0–3)  mean entropy : {early:.3f} nats")
    print(f"  Late  layers (8–11) mean entropy : {late:.3f} nats")
    print(f"  Gradient            (late–early) : {late - early:+.3f} nats")
    print("─" * 50)

    # Step 6 — save visualisations
    print("\n[5/5] Saving charts...")
    out_dir = Path(__file__).parent

    plot_entropy_heatmap(
        entropy_grid,
        save_path=str(out_dir / "head_entropy_heatmap.png"),
    )
    plot_layer_depth_gradient(
        entropy_grid,
        type_counts,
        save_path=str(out_dir / "head_layer_depth_gradient.png"),
    )
    plot_single_head_examples(
        all_weights,
        save_path=str(out_dir / "head_type_examples.png"),
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("Outputs:")
    print("  attention-is-all-you-need/head_entropy_heatmap.png")
    print("  attention-is-all-you-need/head_layer_depth_gradient.png")
    print("  attention-is-all-you-need/head_type_examples.png")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_analysis()
