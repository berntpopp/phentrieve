from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ExtractionReporter:
    """Generate reports and visualizations for extraction benchmarks."""

    def __init__(self, output_format: str = "markdown"):
        self.output_format = output_format
        sns.set_theme(style="whitegrid")

    def generate_report(self, results: list[dict]) -> str:
        """Generate comprehensive benchmark report."""
        if self.output_format == "markdown":
            return self._markdown_report(results)
        elif self.output_format == "html":
            return self._html_report(results)
        elif self.output_format == "latex":
            return self._latex_report(results)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def _markdown_report(self, results: list[dict]) -> str:
        """Generate markdown report."""
        lines = ["# HPO Extraction Benchmark Report\n"]

        # Summary table
        lines.append("## Summary\n")
        lines.append("| Model | Micro F1 | Macro F1 | Precision | Recall |")
        lines.append("|-------|----------|----------|-----------|--------|")

        for result in results:
            model = result.get("metadata", {}).get("model", "Unknown")
            micro = result.get("corpus_metrics", {}).get("micro", {})
            macro = result.get("corpus_metrics", {}).get("macro", {})

            lines.append(
                f"| {model} | {micro.get('f1', 0):.3f} | "
                f"{macro.get('f1', 0):.3f} | {micro.get('precision', 0):.3f} | "
                f"{micro.get('recall', 0):.3f} |"
            )

        lines.append("\n## Details\n")
        # Add more detailed sections as needed

        return "\n".join(lines)

    def _html_report(self, results: list[dict]) -> str:
        """Generate HTML report."""
        # Placeholder
        return "<html><body><h1>Extraction Report</h1></body></html>"

    def _latex_report(self, results: list[dict]) -> str:
        """Generate LaTeX report."""
        # Placeholder
        return "\\documentclass{article}\\begin{document}\\title{Extraction Report}\\end{document}"

    def plot_pr_curve(self, results: dict, save_path: Path):
        """Plot precision-recall curve."""
        plt.figure(figsize=(8, 6))

        # Extract PR points
        precisions = results.get("pr_curve", {}).get("precisions", [])
        recalls = results.get("pr_curve", {}).get("recalls", [])

        plt.plot(recalls, precisions, "b-", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve for HPO Extraction")
        plt.grid(True, alpha=0.3)

        # Add F1 iso-curves
        import numpy as np

        for f1 in [0.2, 0.4, 0.6, 0.8]:
            x = np.linspace(0.01, 1)
            y = f1 * x / (2 * x - f1)
            plt.plot(x[y >= 0], y[y >= 0], "k--", alpha=0.2)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_assertion_confusion_matrix(
        self, confusion_matrix: np.ndarray, save_path: Path
    ):
        """Plot confusion matrix for assertion detection."""
        plt.figure(figsize=(8, 6))

        labels = ["PRESENT", "ABSENT", "UNCERTAIN"]
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )

        plt.title("Assertion Detection Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_hierarchical_matching_distribution(
        self, match_types: dict[str, int], save_path: Path
    ):
        """Plot distribution of hierarchical match types."""
        plt.figure(figsize=(10, 6))

        # Create bar plot
        types = list(match_types.keys())
        counts = list(match_types.values())

        bars = plt.bar(types, counts, color="steelblue")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(count),
                ha="center",
                va="bottom",
            )

        plt.xlabel("Match Type")
        plt.ylabel("Count")
        plt.title("Distribution of Hierarchical Match Types")
        plt.xticks(rotation=45)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
