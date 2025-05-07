#!/usr/bin/env python3
"""
Utility functions for creating benchmark result visualizations.
"""
import logging
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

K_VALUES_DEFAULT = [1, 3, 5, 10]


def _save_plot(
    fig, output_dir: str, filename_prefix: str, timestamp: str
) -> Optional[str]:
    """Helper to save a plot and close the figure."""
    try:
        output_file = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.png")
        fig.tight_layout()
        fig.savefig(output_file)
        plt.close(fig)
        logger.info(
            f"{filename_prefix.replace('_', ' ').title()} visualization saved to: {output_file}"
        )
        return output_file
    except Exception as e:
        logger.error(f"Error saving plot {filename_prefix}: {e}")
        plt.close(fig)
        return None


def plot_mrr_comparison(
    summaries_df: pd.DataFrame, output_dir: str, timestamp: str
) -> Optional[str]:
    """Plots MRR (Dense vs Re-Ranked if available) by model."""
    if summaries_df.empty:
        logger.warning("No data to plot for MRR comparison.")
        return None

    # Check if we have the new format (with 'Dense' and 'Re-Ranked' columns)
    new_format = "Dense" in summaries_df.columns

    fig, ax = plt.subplots(figsize=(12, 7))

    models = summaries_df["model"].unique()
    x = np.arange(len(models))
    width = 0.35

    mrr_dense_values = []
    mrr_reranked_values = []

    # Check if we have reranked data in the new format
    has_reranked_data = (
        new_format
        and "Re-Ranked" in summaries_df.columns
        and not summaries_df["Re-Ranked"].isna().all()
    )

    for model_name in models:
        model_data = summaries_df[summaries_df["model"] == model_name]
        if new_format:
            # New format with 'Dense' and 'Re-Ranked' columns
            model_row = model_data.iloc[0]
            dense_mrr = model_row.get("Dense", 0.0)
            mrr_dense_values.append(dense_mrr)

            if (
                has_reranked_data
                and "Re-Ranked" in model_row
                and not pd.isna(model_row["Re-Ranked"])
            ):
                reranked_mrr = model_row["Re-Ranked"]
                mrr_reranked_values.append(reranked_mrr)
            elif has_reranked_data:
                mrr_reranked_values.append(0.0)
        else:
            # Old format
            model_summary = model_data.iloc[0]
            # Dense MRR
            dense_mrr = model_summary.get(
                "avg_mrr_dense",
                model_summary.get("avg_mrr", model_summary.get("mrr", 0.0)),
            )
            mrr_dense_values.append(dense_mrr)

            if has_reranked_data:
                reranked_mrr = model_summary.get("avg_mrr_reranked", 0.0)
                mrr_reranked_values.append(reranked_mrr)

    # Configure plot aesthetics
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Mean Reciprocal Rank (MRR)")
    ax.set_title("MRR Comparison: Dense vs Re-Ranked Retrieval")
    ax.set_ylim(0, 1.0)  # MRR is always between 0 and 1
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)

    # Create the bars (without error bars)
    if has_reranked_data:
        # Two sets of bars when we have both dense and reranked data
        rects1 = ax.bar(
            x - width / 2,
            mrr_dense_values,
            width,
            label="Dense Retrieval",
            color="#1f77b4",
        )

        rects2 = ax.bar(
            x + width / 2,
            mrr_reranked_values,
            width,
            label="Re-Ranked",
            color="#ff7f0e",
        )

        # Add the legend
        ax.legend()
    else:
        # Just one set of bars for dense retrieval
        rects1 = ax.bar(
            x, mrr_dense_values, width, label="Dense Retrieval", color="#1f77b4"
        )

    # Add values on top of bars
    def autolabel(rects):
        """Attach a text label above each bar showing its value."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Add labels to the bars
    autolabel(rects1)
    if has_reranked_data:
        autolabel(rects2)

    return _save_plot(fig, output_dir, "mrr_comparison", timestamp)


def plot_metric_at_k_bars(
    flat_df: pd.DataFrame,
    metric_name: str,
    y_label: str,
    output_dir: str,
    timestamp: str,
) -> Optional[str]:
    """Plots a given metric@K (e.g., Hit Rate, OntSim) as grouped/faceted bar charts."""
    if (
        flat_df.empty
        or "value" not in flat_df.columns
        or flat_df["value"].isnull().all()
    ):
        logger.warning(f"No data to plot for {metric_name} bar chart.")
        return None

    models = flat_df["model"].unique()
    num_models = len(models)

    if num_models == 0:
        logger.warning(f"No models found in data for {metric_name} bar chart.")
        return None

    if num_models > 1:
        fig, axes = plt.subplots(
            num_models, 1, figsize=(10, 4 * num_models), sharex=True, sharey=True
        )
        if num_models == 1:  # if only one model, axes is not a list
            axes = [axes]
        for i, model_name in enumerate(models):
            model_df = flat_df[flat_df["model"] == model_name]
            ax = axes[i]
            sns.barplot(
                data=model_df,
                x="k",
                y="value",
                hue="method",
                palette={"Dense": "#1f77b4", "Re-Ranked": "#ff7f0e"},
                ax=ax,
                errorbar="sd",
            )
            for c in ax.containers:
                ax.bar_label(c, fmt="%.3f", fontsize=8, padding=3)
            ax.set_title(f"{metric_name}: Dense vs Re-Ranked - {model_name}")
            ax.set_xlabel("K value")
            ax.set_ylabel(y_label)
            ax.legend(title="Method", loc="upper left")
            ax.set_ylim(0, 1.05)
        fig.suptitle(
            f"{metric_name} at K: Dense vs Re-Ranked by Model",
            fontsize=16,
            y=1.02 if num_models > 1 else 1.0,
        )
    else:  # Single model case
        fig, ax = plt.subplots(figsize=(10, 6))
        model_name = models[0]
        model_df = flat_df[flat_df["model"] == model_name]
        sns.barplot(
            data=model_df,
            x="k",
            y="value",
            hue="method",
            palette={"Dense": "#1f77b4", "Re-Ranked": "#ff7f0e"},
            ax=ax,
            errorbar="sd",
        )
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=9, padding=3)
        ax.set_title(f"{metric_name}: Dense vs Re-Ranked - {model_name}")
        ax.set_xlabel("K value")
        ax.set_ylabel(y_label)
        ax.legend(title="Method", loc="upper left")
        ax.set_ylim(0, 1.05)

    return _save_plot(
        fig, output_dir, f"{metric_name.lower().replace(' ', '_')}_bars", timestamp
    )


def plot_metric_at_k_lines(
    flat_df: pd.DataFrame,
    metric_name: str,
    y_label: str,
    output_dir: str,
    timestamp: str,
) -> Optional[str]:
    """Plots a given metric@K as line plots, faceted by model if multiple."""
    if (
        flat_df.empty
        or "value" not in flat_df.columns
        or flat_df["value"].isnull().all()
    ):
        logger.warning(f"No data to plot for {metric_name} line chart.")
        return None

    # Convert 'k' to integer to avoid categorical warnings
    if "k" in flat_df.columns:
        flat_df["k"] = flat_df["k"].astype(int)

    num_models = flat_df["model"].nunique()

    if num_models > 1:
        # Create a separate plot for each model to avoid FacetGrid issues
        fig, axes = plt.subplots(
            1, num_models, figsize=(5 * num_models, 6), sharey=True
        )

        for i, (model_name, model_df) in enumerate(flat_df.groupby("model")):
            ax = axes[i] if num_models > 1 else axes

            # Plot Dense values
            dense_df = model_df[model_df["method"] == "Dense"]
            if not dense_df.empty:
                sns.lineplot(
                    data=dense_df,
                    x="k",
                    y="value",
                    color="#1f77b4",  # Blue
                    marker="o",
                    label="Dense",
                    errorbar="sd",
                    ax=ax,
                )

            # Plot Re-Ranked values
            reranked_df = model_df[model_df["method"] == "Re-Ranked"]
            if not reranked_df.empty:
                sns.lineplot(
                    data=reranked_df,
                    x="k",
                    y="value",
                    color="#ff7f0e",  # Orange
                    marker="s",
                    label="Re-Ranked",
                    dashes=[2, 2],  # Format fix: use list instead of tuple
                    errorbar="sd",
                    ax=ax,
                )

            ax.set_title(f"{model_name}")
            ax.set_xlabel("K Value")
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle="--", alpha=0.7)

            # Only show y-label on the first plot
            if i == 0:
                ax.set_ylabel(y_label)

        # Add a common legend and title
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            title="Method",
            ncol=2,
        )

        # Remove individual legends
        for ax in axes.flatten() if num_models > 1 else [axes]:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        plt.suptitle(f"{metric_name} Trends vs K: Dense vs Re-Ranked", y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    else:  # Single model
        # Use a simpler approach for a single model
        fig, ax = plt.subplots(figsize=(10, 6))
        model_name = flat_df["model"].unique()[0]

        # Plot each method separately to avoid style conflicts
        for method, color, marker, dash in zip(
            ["Dense", "Re-Ranked"],
            ["#1f77b4", "#ff7f0e"],
            ["o", "s"],
            [None, [2, 2]],  # Format fix: use list instead of tuple
        ):
            method_df = flat_df[flat_df["method"] == method]
            if not method_df.empty:
                sns.lineplot(
                    data=method_df,
                    x="k",
                    y="value",
                    color=color,
                    marker=marker,
                    dashes=dash,
                    label=method,
                    errorbar="sd",
                    ax=ax,
                )

        ax.set_title(f"{metric_name} Trends vs K: Dense vs Re-Ranked - {model_name}")
        ax.set_xlabel("K Value")
        ax.set_ylabel(y_label)
        ax.legend(title="Method")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.7)
        # Add text labels for points if desired
        # for method in flat_df['method'].unique():
        #     method_df = flat_df[flat_df['method'] == method]
        #     for _, row in method_df.iterrows():
        #         ax.text(row["k"], row["value"] + 0.02, f'{row["value"]:.3f}', ha="center", va="bottom", fontsize=8)

    return _save_plot(
        fig, output_dir, f"{metric_name.lower().replace(' ', '_')}_lines", timestamp
    )
