import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches

# --- Color scheme ---
VIBRANT_BLUE = '#0066FF'      # Encoder
VIBRANT_ORANGE = '#FF6600'    # Decoder
VIBRANT_CYAN = '#00CCFF'      # Chain output
VIBRANT_PURPLE = '#8A2BE2'    # Input/Output labels

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# --- Big component boxes ---
encoder_box = patches.FancyBboxPatch(
    (0.5, 2), 3, 2,
    boxstyle="round,pad=0.3",
    facecolor=VIBRANT_BLUE,
    edgecolor='white',
    linewidth=3,
    alpha=0.95
)
ax.add_patch(encoder_box)

decoder_box = patches.FancyBboxPatch(
    (3.5, 2), 3, 2,
    boxstyle="round,pad=0.3",
    facecolor=VIBRANT_ORANGE,
    edgecolor='white',
    linewidth=3,
    alpha=0.95
)
ax.add_patch(decoder_box)

chain_box = patches.FancyBboxPatch(
    (6.5, 2), 3, 2,
    boxstyle="round,pad=0.3",
    facecolor=VIBRANT_CYAN,
    edgecolor='white',
    linewidth=3,
    alpha=0.95
)
ax.add_patch(chain_box)

# --- Arrows between blocks (now with real length) ---
arrow_style = mpatches.ArrowStyle('Simple', head_length=8, head_width=6)

arrow1 = mpatches.FancyArrowPatch(
    (3.5, 3), (3.5 + 0.5, 3),   # from right edge of encoder towards decoder
    arrowstyle=arrow_style,
    color='black',
    lw=2
)
ax.add_patch(arrow1)

arrow2 = mpatches.FancyArrowPatch(
    (6.5, 3), (6.5 + 0.5, 3),   # from right edge of decoder towards chain box
    arrowstyle=arrow_style,
    color='black',
    lw=2
)
ax.add_patch(arrow2)

# --- Section labels ---
ax.text(2, 4.4, 'Encoder', ha='center', va='center',
        fontsize=15, fontweight='bold', color='white')
ax.text(5, 4.4, 'Decoder', ha='center', va='center',
        fontsize=15, fontweight='bold', color='white')
ax.text(8, 4.4, 'Causal Chain Output', ha='center', va='center',
        fontsize=15, fontweight='bold', color='white')

# --- Sub-labels tied to thesis terminology ---
ax.text(
    2, 3,
    'CPG + GraphCodeBERT\nRelation-aware GAT\nCAL + ACC',
    ha='center', va='center',
    fontsize=10, color='white'
)

ax.text(
    5, 3,
    'CKG-guided decoding\nFeasibility constraints\nBeam search',
    ha='center', va='center',
    fontsize=10, color='white'
)

ax.text(
    8, 3,
    'Executable root → propagation → sink\ninterprocedural chain',
    ha='center', va='center',
    fontsize=10, color='white'
)

# --- Input/Output labels ---
ax.text(0.5, 5.4, 'Input: Source Code + CPG', ha='left', va='center',
        fontsize=12, fontweight='bold', color=VIBRANT_PURPLE)
ax.text(9.5, 5.4, 'Output: Vulnerability Chain', ha='right', va='center',
        fontsize=12, fontweight='bold', color=VIBRANT_PURPLE)

plt.title('Chain-Centric Model Architecture', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
