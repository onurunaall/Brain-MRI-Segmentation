import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
ax.set_aspect("equal")
ax.axis("off")


def draw_prism(
    x,
    y,
    w,
    h,
    dx,
    dy,
    facecol="#A9C7E8",
    topcol="#D4E5F7",
    sidecol="#7B9EC7",
    edgecol="#3B5F88",
    lw=0.8,
):
    """Draws a standalone 3D isometric rectangular prism."""
    front = patches.Polygon(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        closed=True,
        facecolor=facecol,
        edgecolor=edgecol,
        linewidth=lw,
        zorder=3,
    )
    top = patches.Polygon(
        [
            [x, y + h],
            [x + dx, y + h + dy],
            [x + w + dx, y + h + dy],
            [x + w, y + h],
        ],
        closed=True,
        facecolor=topcol,
        edgecolor=edgecol,
        linewidth=lw,
        zorder=4,
    )
    side = patches.Polygon(
        [
            [x + w, y],
            [x + w + dx, y + dy],
            [x + w + dx, y + h + dy],
            [x + w, y + h],
        ],
        closed=True,
        facecolor=sidecol,
        edgecolor=edgecol,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(side)


def draw_slab_stack(
    x_start,
    y,
    num_slabs,
    slab_w,
    h,
    dx,
    dy,
    gap=0.10,
    facecol="#A9C7E8",
    topcol="#D4E5F7",
    sidecol="#7B9EC7",
    edgecol="#3B5F88",
):
    """Draws a stack of discrete isometric slabs representing channel depth."""
    centers = []
    for i in range(num_slabs):
        sx = x_start + i * (slab_w + gap)
        draw_prism(
            sx,
            y,
            slab_w,
            h,
            dx,
            dy,
            facecol=facecol,
            topcol=topcol,
            sidecol=sidecol,
            edgecol=edgecol,
        )
        centers.append(sx + slab_w / 2)
    total_w = num_slabs * slab_w + (num_slabs - 1) * gap
    return centers, total_w


def draw_arrow(
    x1, y1, x2, y2, color="#404040", lw=2.2, mutation_scale=13, ls="-"
):
    """Draws an arrow vector."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        color=color,
        linewidth=lw,
        linestyle=ls,
        zorder=6,
    )
    ax.add_patch(arrow)


# ==================== DIMENSIONS & ELEVATIONS ====================
slab_w, gap = 0.36, 0.10
dx, dy = 0.36, 0.26

# Elevations: Level 0 (top) down to Level 4 (bottleneck)
# Fixed vertical gaps of 1.1 units between all consecutive levels
y0, h0 = 15.0, 5.0  # Level 0 (HxW)
y1, h1 = 10.1, 3.8  # Level 1 (H/2 x W/2)
y2, h2 = 6.2, 2.8  # Level 2 (H/4 x W/4)
y3, h3 = 3.2, 1.9  # Level 3 (H/8 x W/8)
y4, h4 = 1.0, 1.1  # Level 4: Bottleneck (H/16 x W/16)


# ==================== ENCODER STAGES ====================
# Input: 1 slab (neutral grey)
c_in, w_in = draw_slab_stack(
    0.8,
    y0,
    1,
    slab_w,
    h0,
    dx,
    dy,
    facecol="#DDE2E8",
    topcol="#EEF2F7",
    sidecol="#BFC6D0",
    edgecol="#6C7684",
)

# ConvIn (8 ch): 2 slabs
c_ci, w_ci = draw_slab_stack(2.0, y0, 2, slab_w, h0, dx, dy)

# DownBlock 0 (32 ch): 3 slabs
# Starts directly under Down Arrow 0 (c_ci[-1] = 2.46)
x_d0 = c_ci[-1] - slab_w / 2
c_d0, w_d0 = draw_slab_stack(x_d0, y1, 3, slab_w, h1, dx, dy)

# DownBlock 1 (64 ch): 4 slabs
x_d1 = c_d0[-1] - slab_w / 2
c_d1, w_d1 = draw_slab_stack(x_d1, y2, 4, slab_w, h2, dx, dy)

# DownBlock 2 (128 ch): 5 slabs
x_d2 = c_d1[-1] - slab_w / 2
c_d2, w_d2 = draw_slab_stack(x_d2, y3, 5, slab_w, h3, dx, dy)

# DownBlock 3: Bottleneck (256 ch): 6 slabs
x_d3 = c_d2[-1] - slab_w / 2
c_d3, w_d3 = draw_slab_stack(
    x_d3,
    y4,
    6,
    slab_w,
    h4,
    dx,
    dy,
    facecol="#8EAFCE",
    topcol="#B9D2E8",
    sidecol="#668CAE",
    edgecol="#2B4E6F",
)


# ==================== DECODER STAGES ====================
# UpBlock 0 (128 ch): 5 slabs
# Starts directly over the Bottleneck's exit arrow (c_d3[-1])
x_u0 = c_d3[-1] - slab_w / 2
c_u0, w_u0 = draw_slab_stack(x_u0, y3, 5, slab_w, h3, dx, dy)

# UpBlock 1 (64 ch): 4 slabs
x_u1 = c_u0[-1] - slab_w / 2
c_u1, w_u1 = draw_slab_stack(x_u1, y2, 4, slab_w, h2, dx, dy)

# UpBlock 2 (32 ch): 3 slabs
x_u2 = c_u1[-1] - slab_w / 2
c_u2, w_u2 = draw_slab_stack(x_u2, y1, 3, slab_w, h1, dx, dy)

# UpBlock 3 (1 ch): 1 slab
x_u3 = c_u2[-1] - slab_w / 2
c_u3, w_u3 = draw_slab_stack(x_u3, y0, 1, slab_w, h0, dx, dy)

# ConvOut (1 ch): 1 slab (mint green)
c_out, w_out = draw_slab_stack(
    x_u3 + slab_w + 0.9,
    y0,
    1,
    slab_w,
    h0,
    dx,
    dy,
    facecol="#A3D9B5",
    topcol="#D0F0DC",
    sidecol="#74BA8C",
    edgecol="#2B7345",
)


# ==================== RECTILINEAR ARROWS ====================
# Horizontal: Input -> ConvIn
draw_arrow(
    0.8 + slab_w + dx + 0.05,
    y0 + h0 / 2,
    2.0 - 0.05,
    y0 + h0 / 2,
    color="#4A4A4A",
    lw=1.8,
)

# Vertical Downsampling: from base of last slab to top of next first slab
draw_arrow(c_ci[-1], y0, c_ci[-1], y1 + h1 + dy, color="#D4AC0D", lw=2.4)
draw_arrow(c_d0[-1], y1, c_d0[-1], y2 + h2 + dy, color="#D4AC0D", lw=2.4)
draw_arrow(c_d1[-1], y2, c_d1[-1], y3 + h3 + dy, color="#D4AC0D", lw=2.4)
draw_arrow(c_d2[-1], y3, c_d2[-1], y4 + h4 + dy, color="#D4AC0D", lw=2.4)

# Vertical Upsampling: from top of last slab to base of next first slab
draw_arrow(c_d3[-1], y4 + h4 + dy, c_d3[-1], y3, color="#388E3C", lw=2.4)
draw_arrow(c_u0[-1], y3 + h3 + dy, c_u0[-1], y2, color="#388E3C", lw=2.4)
draw_arrow(c_u1[-1], y2 + h2 + dy, c_u1[-1], y1, color="#388E3C", lw=2.4)
draw_arrow(c_u2[-1], y1 + h1 + dy, c_u2[-1], y0, color="#388E3C", lw=2.4)

# Horizontal: UpBlock 3 -> ConvOut
draw_arrow(
    x_u3 + slab_w + dx + 0.05,
    y0 + h0 / 2,
    x_u3 + slab_w + 0.9 - 0.05,
    y0 + h0 / 2,
    color="#4A4A4A",
    lw=1.8,
)

# Additive Skip Connections: Strictly horizontal across levels 1, 2, 3
skips = [
    (x_d0 + w_d0 + dx, x_u2, y1 + h1 / 2),
    (x_d1 + w_d1 + dx, x_u1, y2 + h2 / 2),
    (x_d2 + w_d2 + dx, x_u0, y3 + h3 / 2),
]

for x_start, x_end, y_pos in skips:
    draw_arrow(
        x_start + 0.08,
        y_pos,
        x_end - 0.08,
        y_pos,
        color="#4A4A4A",
        lw=2.0,
        ls="-",
    )
    # Centered 'torch.add' indicator above each skip vector
    ax.text(
        (x_start + x_end) / 2,
        y_pos + 0.22,
        "torch.add",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#3E4A59",
    )


# ==================== EXTERIOR MARGIN LABELS ====================
# Left margin: Encoder metadata
ax.text(
    0.8 + slab_w / 2,
    y0 - 0.3,
    "Input\n(1×H×W)",
    ha="center",
    va="top",
    fontsize=8.5,
    fontweight="bold",
    color="#333",
)
ax.text(
    2.0 + w_ci / 2,
    y0 - 0.3,
    "ConvIn\n(8×H×W)",
    ha="center",
    va="top",
    fontsize=8.5,
    fontweight="bold",
    color="#333",
)
ax.text(
    x_d0 - 0.2,
    y1 + h1 / 2,
    "DownBlock 0\n(32×H/2×W/2)",
    ha="right",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#333",
)
ax.text(
    x_d1 - 0.2,
    y2 + h2 / 2,
    "DownBlock 1\n(64×H/4×W/4)",
    ha="right",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#333",
)
ax.text(
    x_d2 - 0.2,
    y3 + h3 / 2,
    "DownBlock 2\n(128×H/8×W/8)",
    ha="right",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#333",
)
ax.text(
    x_d3 + w_d3 / 2,
    y4 - 0.3,
    "DownBlock 3 (Bottleneck)\n(256×H/16×W/16)",
    ha="center",
    va="top",
    fontsize=8.5,
    fontweight="bold",
    color="#1B3A57",
)

# Right margin: Decoder metadata
ax.text(
    x_u0 + w_u0 + dx + 0.2,
    y3 + h3 / 2,
    "UpBlock 0\n(128×H/8×W/8)",
    ha="left",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#333",
)
ax.text(
    x_u1 + w_u1 + dx + 0.2,
    y2 + h2 / 2,
    "UpBlock 1\n(64×H/4×W/4)",
    ha="left",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#333",
)
ax.text(
    x_u2 + w_u2 + dx + 0.2,
    y1 + h1 / 2,
    "UpBlock 2\n(32×H/2×W/2)",
    ha="left",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#333",
)
ax.text(
    x_u3 + slab_w / 2,
    y0 - 0.3,
    "UpBlock 3\n(1×H×W)",
    ha="center",
    va="top",
    fontsize=8.5,
    fontweight="bold",
    color="#333",
)
ax.text(
    c_out[0],
    y0 - 0.3,
    "ConvOut\n(1×H×W)",
    ha="center",
    va="top",
    fontsize=8.5,
    fontweight="bold",
    color="#1E6536",
)


# ==================== LEGEND ====================
legend_items = [
    patches.Patch(
        facecolor="#A9C7E8",
        edgecolor="#3B5F88",
        label="Feature Map Slab (Channels)",
    ),
    patches.Patch(
        facecolor="#DDE2E8", edgecolor="#6C7684", label="Input Image"
    ),
    patches.Patch(facecolor="#A3D9B5", edgecolor="#2B7345", label="Output Map"),
    FancyArrowPatch(
        (0, 0),
        (1, 0),
        color="#D4AC0D",
        lw=2.2,
        arrowstyle="-|>",
        mutation_scale=11,
        label="DownBlock (Conv stride 2)",
    ),
    FancyArrowPatch(
        (0, 0),
        (1, 0),
        color="#388E3C",
        lw=2.2,
        arrowstyle="-|>",
        mutation_scale=11,
        label="UpBlock (ConvTranspose 2x2)",
    ),
    FancyArrowPatch(
        (0, 0),
        (1, 0),
        color="#4A4A4A",
        lw=2.0,
        arrowstyle="-|>",
        mutation_scale=11,
        label="Additive Skip (torch.add)",
    ),
]

ax.legend(
    handles=legend_items,
    loc="lower right",
    frameon=True,
    facecolor="#FCFCFC",
    edgecolor="#D0D0D0",
    fontsize=9.0,
    bbox_to_anchor=(0.99, 0.03),
    handlelength=2.2,
    labelspacing=0.5,
)

plt.xlim(-1.8, 16.2)
plt.ylim(0.0, 21.5)
plt.tight_layout()

plt.savefig("unet_setup.png", dpi=300, bbox_inches="tight")
plt.savefig("unet_setup.svg", bbox_inches="tight")
print("Rendered unet_setup.png and unet_setup.svg")
