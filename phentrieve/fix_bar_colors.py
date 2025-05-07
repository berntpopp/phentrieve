"""
Script to fix the bar colors in visualization charts.
This will update both the HR@k and OntSim@k bar colors to use model-based coloring.
"""

import os
import re

# Path to the orchestrator file
file_path = "phentrieve/evaluation/comparison_orchestrator.py"

# Read the original content
with open(file_path, "r") as f:
    content = f.read()

# Fix colors in HR@k case 2 (single metric)
hr_pattern = r'(# Case 2: Single metric type\s+else:\s+metric = metrics\[0\]\s+)bars = ax\.bar\(x, sorted_df\[metric\], color="tab:blue"\)'
hr_replacement = r'\1# Use model-specific colors for better visualization\n                        bars = ax.bar(x, sorted_df[metric], color=[color_map[m] for m in sorted_df["Model"]])'

content = re.sub(hr_pattern, hr_replacement, content)

# Fix colors in OntSim@k case 1 (comparison)
ontsim_pattern1 = r'(dense_metric = f"OntSim@{k} \(Dense\)"\s+reranked_metric = f"OntSim@{k} \(Re-Ranked\)"\s+\s+bar_width = 0\.35\s+)dense_bars = ax\.bar\(x - bar_width/2, sorted_df\[dense_metric\], \s+width=bar_width, label="Dense", color="tab:blue"\)\s+reranked_bars = ax\.bar\(x \+ bar_width/2, sorted_df\[reranked_metric\], \s+width=bar_width, label="Re-Ranked", color="tab:orange"\)'

ontsim_replacement1 = r'\1# Apply model-based colors with different alpha for Dense vs Reranked\n                        dense_bars = ax.bar(x - bar_width/2, sorted_df[dense_metric], \n                                    width=bar_width, label="Dense", \n                                    color=[color_map[m] for m in sorted_df["Model"]], alpha=0.9)\n                        reranked_bars = ax.bar(x + bar_width/2, sorted_df[reranked_metric], \n                                       width=bar_width, label="Re-Ranked", \n                                       color=[color_map[m] for m in sorted_df["Model"]], alpha=0.6)'

content = re.sub(ontsim_pattern1, ontsim_replacement1, content, flags=re.DOTALL)

# Fix colors in OntSim@k case 2 (single metric)
ontsim_pattern2 = r'(# Case 2: Single metric type\s+else:\s+metric = metrics\[0\]\s+)bars = ax\.bar\(x, sorted_df\[metric\], color="tab:blue"\)'

ontsim_replacement2 = r'\1# Use model-specific colors for better visualization\n                        bars = ax.bar(x, sorted_df[metric], color=[color_map[m] for m in sorted_df["Model"]])'

content = re.sub(ontsim_pattern2, ontsim_replacement2, content)

# Fix the legend in OntSim@k case 1
legend_pattern = r"(if i == 0:  # Only add legend to first plot\s+)ax\.legend\(\)"

legend_replacement = r'\1# Create legend for dense vs reranked types\n                            handles, labels = ax.get_legend_handles_labels()\n                            ax.legend(handles, labels, loc="upper right")\n                            \n                            # Only add model color legend to the first plot\n                            if len(model_names) <= 8:  # Only if we have a reasonable number of models\n                                model_patches = [plt.Rectangle((0,0),1,1, color=color_map[model]) \n                                                 for model in model_names]\n                                ax2 = ax.twinx()  # Create a second y-axis\n                                ax2.set_yticks([])\n                                ax2.legend(model_patches, model_names, loc="upper left", \n                                           bbox_to_anchor=(1.05, 1), title="Models")'

content = re.sub(legend_pattern, legend_replacement, content)

# Add model legend to OntSim@k case 2
ontsim_add_legend = r"(# Add value labels.*?\n.*?fontsize=9\))"

ontsim_legend_addition = r'\1\n                            \n                        # Add model color legend to the first plot\n                        if i == 0 and len(model_names) <= 8:\n                            model_patches = [plt.Rectangle((0,0),1,1, color=color_map[model]) \n                                             for model in model_names]\n                            ax.legend(model_patches, model_names, loc="upper right",\n                                      title="Models")'

content = re.sub(ontsim_add_legend, ontsim_legend_addition, content, flags=re.DOTALL)

# Write updated content back to file
with open(file_path, "w") as f:
    f.write(content)

print("Applied model-specific coloring to all bar charts.")
print("Run 'python -m phentrieve benchmark visualize' to see the changes.")
