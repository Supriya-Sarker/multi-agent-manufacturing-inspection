import matplotlib.pyplot as plt

categories = [
    "Weld state", "Bead quality", "Arc stability", "Melt pool stability",
    "Melt pool length", "Spatter particle size", "Image explanation", "Spatter volume"
]

colors = [
    "#A6C8E0", "#FFBE7D", "#8CD17D", "#F28E8B",
    "#D4A6C8", "#C49A8B", "#FFB6C1", "#D3D3D3"
]

# data = [
#     # ---- Row 1: Bead 4 ----
#     [62, 57, 57, 48, 38, 19, 19, 14],   # zero-shot prompt 1
#     [96, 74, 74, 78, 70, 35, 35, 30],   # zero-shot prompt 2
#     [94, 53, 53, 53, 41, 29, 29, 35],   # 3-shot prompt 2

#     # ---- Row 2: Bead 2 ----
#     [57, 71, 57, 43, 48, 29, 33, 38],   # zero-shot prompt 1
#     [86, 43, 81, 33, 45, 43, 24, 48],   # zero-shot prompt 2
#     [86, 59, 59, 41, 55, 45, 14, 27],   # 3-shot prompt 2

# ]

# titles = [
#     "Bead 4 | Zero-shot | Prompt 1",
#     "Bead 4 | Zero-shot | Prompt 2",
#     "Bead 4 | 3-shot | Prompt 2",
#     "Bead 2 | Zero-shot | Prompt 1",
#     "Bead 2 | Zero-shot | Prompt 2",
#     "Bead 2 | 3-shot | Prompt 2",
# ]

data = [
    # Previous result: merged frames, zero-shot prompt 2, without feature extraction
    [93, 43, 61, 50, 61, 52, 20, 48],

    # New result from printed performance matrix
    # Table values:
    # weld_state 0.95, arc_stability 0.26, spatter_particle_size 0.40,
    # spatter_volume 0.64, melt_pool_length 0.76,
    # melt_pool_stability 0.36, quality 0.43, justification 0.14
    [95, 43, 26, 36, 76, 40, 14, 64],
]

titles = [
    "Merged Frames | Zero-shot | Prompt 2\nwithout feature extraction",
    "Merged Frames | Zero-shot | Prompt 3\nwith overlay + metadata"
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes = axes.flatten()

for i, ax in enumerate(axes):
    bars = ax.bar(categories, data[i], color=colors)

    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar, value in zip(bars, data[i]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value}%",
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold'
        )

    ax.set_title(titles[i], fontweight='bold', fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontweight='bold')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig("with_vs_without_feature_extraction_result.png", dpi=300, bbox_inches='tight')
plt.show()