"""
Demonstration of model-based coloring for benchmark visualizations.
This script shows what the charts would look like with model-specific colors.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Create sample benchmark data
models = [
    "paraphrase_multilingual_mpnet_base_v2",
    "bge_m3",
    "distiluse_base_multilingual_cased_v2",
    "paraphrase_multilingual_MiniLM_L12_v2",
    "jina_embeddings_v2_base_de",
    "biolord_2023_m",
    "cross_en_de_roberta_sentence_transformer",
    "intfloat_multilingual_e5_base",
]

# Sample metrics data
data = {
    "Model": models,
    "MRR (Dense)": [0.274, 0.626, 0.140, 0.258, 0.724, 0.808, 0.419, 0.312],
    "HR@1 (Dense)": [0.15, 0.45, 0.08, 0.13, 0.59, 0.71, 0.25, 0.21],
    "HR@3 (Dense)": [0.27, 0.64, 0.12, 0.29, 0.68, 0.82, 0.39, 0.36],
    "HR@5 (Dense)": [0.38, 0.72, 0.17, 0.42, 0.75, 0.89, 0.52, 0.47],
    "HR@10 (Dense)": [0.51, 0.82, 0.28, 0.56, 0.85, 0.94, 0.67, 0.59],
    "OntSim@1 (Dense)": [0.22, 0.59, 0.12, 0.24, 0.68, 0.78, 0.32, 0.29],
    "OntSim@3 (Dense)": [0.32, 0.67, 0.18, 0.35, 0.76, 0.86, 0.45, 0.40],
    "OntSim@5 (Dense)": [0.41, 0.75, 0.25, 0.48, 0.82, 0.91, 0.58, 0.54],
    "OntSim@10 (Dense)": [0.59, 0.85, 0.38, 0.62, 0.88, 0.96, 0.72, 0.70],
}

df = pd.DataFrame(data)

# Sort models by MRR
df = df.sort_values(by="MRR (Dense)", ascending=False).reset_index(drop=True)

# Create a colormap for models
model_names = df["Model"].tolist()
model_colors = plt.cm.get_cmap("tab10", len(model_names))
color_map = {model: model_colors(i) for i, model in enumerate(model_names)}

# Create output directory
output_dir = "model_colored_visualizations"
os.makedirs(output_dir, exist_ok=True)

# 1. MRR COMPARISON CHART
plt.figure(figsize=(14, 6))
ax = plt.axes()

x = np.arange(len(df))
bars = ax.bar(x, df["MRR (Dense)"], color=[color_map[m] for m in df["Model"]])

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

# Create legend for model colors
model_patches = [
    plt.Rectangle((0, 0), 1, 1, color=color_map[model]) for model in model_names
]
plt.legend(model_patches, model_names, loc="upper right")

plt.xlabel("Model", fontsize=14)
plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=14)
plt.title("MRR Comparison Across Models", fontsize=16)
plt.xticks(x, df["Model"], rotation=45, ha="right")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mrr_comparison_model_colors.png"), dpi=300)
plt.close()

# 2. COMBINED HR COMPARISON CHART
hr_metrics = ["HR@1 (Dense)", "HR@3 (Dense)", "HR@5 (Dense)", "HR@10 (Dense)"]
fig, axes = plt.subplots(
    len(hr_metrics), 1, figsize=(12, 4 * len(hr_metrics)), squeeze=False
)

for i, metric in enumerate(hr_metrics):
    ax = axes[i, 0]
    k = metric.split("@")[1].split()[0]

    bars = ax.bar(x, df[metric], color=[color_map[m] for m in df["Model"]])

    # Add value labels on bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Only add legend to first plot
    if i == 0:
        model_patches = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[model]) for model in model_names
        ]
        ax.legend(model_patches, model_names, loc="upper right")

    ax.set_ylim(0, 1.0)
    ax.set_title(f"Hit Rate @ {k} Comparison", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax.set_ylabel(f"HR@{k}")

    # Only show x-label on bottom plot
    if i == len(hr_metrics) - 1:
        ax.set_xlabel("Model")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "combined_hr_model_colors.png"), dpi=300)
plt.close()

# 3. COMBINED ONTSIM COMPARISON CHART
ont_metrics = [
    "OntSim@1 (Dense)",
    "OntSim@3 (Dense)",
    "OntSim@5 (Dense)",
    "OntSim@10 (Dense)",
]
fig, axes = plt.subplots(
    len(ont_metrics), 1, figsize=(12, 4 * len(ont_metrics)), squeeze=False
)

for i, metric in enumerate(ont_metrics):
    ax = axes[i, 0]
    k = metric.split("@")[1].split()[0]

    bars = ax.bar(x, df[metric], color=[color_map[m] for m in df["Model"]])

    # Add value labels on bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Only add legend to first plot
    if i == 0:
        model_patches = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[model]) for model in model_names
        ]
        ax.legend(model_patches, model_names, loc="upper right")

    ax.set_ylim(0, 1.0)
    ax.set_title(f"Ontology Similarity @ {k} Comparison", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax.set_ylabel(f"OntSim@{k}")

    # Only show x-label on bottom plot
    if i == len(ont_metrics) - 1:
        ax.set_xlabel("Model")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "combined_ontsim_model_colors.png"), dpi=300)
plt.close()

print("Model-colored visualizations created in:", output_dir)
