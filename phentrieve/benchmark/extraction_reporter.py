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
        has_ontology_metrics = any(
            result.get("corpus_metrics", {}).get("ontology_metrics")
            for result in results
        )
        if has_ontology_metrics:
            lines.append(
                "| Model | Micro F1 | Macro F1 | Precision | Recall | "
                "Soft Micro F1 | Partial Micro F1 |"
            )
            lines.append(
                "|-------|----------|----------|-----------|--------|"
                "---------------|------------------|"
            )
        else:
            lines.append("| Model | Micro F1 | Macro F1 | Precision | Recall |")
            lines.append("|-------|----------|----------|-----------|--------|")

        for result in results:
            model = result.get("metadata", {}).get("model", "Unknown")
            corpus_metrics = result.get("corpus_metrics", {})
            micro = corpus_metrics.get("micro", {})
            macro = corpus_metrics.get("macro", {})

            row = (
                f"| {model} | {micro.get('f1', 0):.3f} | "
                f"{macro.get('f1', 0):.3f} | {micro.get('precision', 0):.3f} | "
                f"{micro.get('recall', 0):.3f}"
            )
            if has_ontology_metrics:
                ontology_metrics = corpus_metrics.get("ontology_metrics") or {}
                soft_micro = ontology_metrics.get("soft", {}).get("micro", {})
                partial_micro = ontology_metrics.get("partial", {}).get("micro", {})
                row += (
                    f" | {soft_micro.get('f1', 0):.3f} | "
                    f"{partial_micro.get('f1', 0):.3f}"
                )
            lines.append(f"{row} |")

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
