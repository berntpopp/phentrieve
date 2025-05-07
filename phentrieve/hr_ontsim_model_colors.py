"""
Updated HR@k and OntSim@k visualization with model-specific colors.
Copy these functions to replace the corresponding sections in comparison_orchestrator.py
"""

def generate_hr_visualization(sorted_df, hr_by_k, model_names, color_map, output_dir):
    """
    Generates the HR@k visualizations with model-specific colors.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # Determine how many k-values we have for HR metrics
    hr_k_values = sorted([int(k) for k in hr_by_k.keys()])
    n_hr_plots = len(hr_k_values)
    
    if n_hr_plots > 0:
        # Create a figure with multiple subplots for each k value
        fig, axes = plt.subplots(n_hr_plots, 1, figsize=(12, 5*n_hr_plots), squeeze=False)
        
        for i, k in enumerate(hr_k_values):
            ax = axes[i, 0]
            metrics = hr_by_k[str(k)]
            
            # Check if we have both dense and reranked for this k value
            has_dense_k = any("Dense" in m for m in metrics)
            has_reranked_k = any("Re-Ranked" in m for m in metrics)
            has_comparison_k = has_dense_k and has_reranked_k
            
            x = np.arange(len(sorted_df))
            
            # Case 1: If we have both dense and reranked
            if has_comparison_k:
                dense_metric = f"HR@{k} (Dense)"
                reranked_metric = f"HR@{k} (Re-Ranked)"
                
                bar_width = 0.35
                # Apply model-based colors with different alpha for Dense vs Reranked
                dense_bars = ax.bar(x - bar_width/2, sorted_df[dense_metric], 
                                   width=bar_width, label="Dense", 
                                   color=[color_map[m] for m in sorted_df["Model"]], alpha=0.9)
                reranked_bars = ax.bar(x + bar_width/2, sorted_df[reranked_metric], 
                                      width=bar_width, label="Re-Ranked", 
                                      color=[color_map[m] for m in sorted_df["Model"]], alpha=0.6)
                
                # Add value labels (only for a reasonable number of models)
                if len(sorted_df) <= 8:
                    for j, bar in enumerate(dense_bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f"{height:.3f}", ha="center", va="bottom", fontsize=9)
                    
                    for j, bar in enumerate(reranked_bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f"{height:.3f}", ha="center", va="bottom", fontsize=9)
                
                if i == 0:  # Only add legend to first plot
                    # Create legend for dense vs reranked types
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc='upper right')
                    
                    # Only add model color legend to the first plot
                    if len(model_names) <= 8:  # Only if we have a reasonable number of models
                        model_patches = [plt.Rectangle((0,0),1,1, color=color_map[model]) 
                                         for model in model_names]
                        ax2 = ax.twinx()  # Create a second y-axis
                        ax2.set_yticks([])
                        ax2.legend(model_patches, model_names, loc='upper left', 
                                  bbox_to_anchor=(1.05, 1), title="Models")
                
            # Case 2: Single metric type
            else:
                metric = metrics[0]
                # Use model-based colors
                bars = ax.bar(x, sorted_df[metric], color=[color_map[m] for m in sorted_df["Model"]])
                
                # Add value labels (only for a reasonable number of models)
                if len(sorted_df) <= 8:
                    for j, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f"{height:.3f}", ha="center", va="bottom", fontsize=9)
                
                # Add model color legend to the first plot
                if i == 0 and len(model_names) <= 8:
                    model_patches = [plt.Rectangle((0,0),1,1, color=color_map[model]) 
                                     for model in model_names]
                    ax.legend(model_patches, model_names, loc='upper right',
                             title="Models")
            
            # Common formatting for this subplot
            ax.set_ylim(0, 1.0)
            ax.set_title(f"Hit Rate @ {k} Comparison", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_df["Model"], rotation=45, ha="right")
            ax.set_ylabel(f"HR@{k}")
            
            # Only show x-label on bottom plot
            if i == n_hr_plots-1:
                ax.set_xlabel("Model")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_hr_comparison.png"), dpi=300)
        plt.close()
        print("Generated combined HR comparison visualization")
        return True
    return False

def generate_ontsim_visualization(sorted_df, ont_by_k, model_names, color_map, output_dir):
    """
    Generates the OntSim@k visualizations with model-specific colors.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # Determine how many k-values we have for OntSim metrics
    ont_k_values = sorted([int(k) for k in ont_by_k.keys()])
    n_ont_plots = len(ont_k_values)
    
    if n_ont_plots > 0:
        # Create a figure with multiple subplots for each k value
        fig, axes = plt.subplots(n_ont_plots, 1, figsize=(12, 5*n_ont_plots), squeeze=False)
        
        for i, k in enumerate(ont_k_values):
            ax = axes[i, 0]
            metrics = ont_by_k[str(k)]
            
            # Check if we have both dense and reranked for this k value
            has_dense_k = any("Dense" in m for m in metrics)
            has_reranked_k = any("Re-Ranked" in m for m in metrics)
            has_comparison_k = has_dense_k and has_reranked_k
            
            x = np.arange(len(sorted_df))
            
            # Case 1: If we have both dense and reranked
            if has_comparison_k:
                dense_metric = f"OntSim@{k} (Dense)"
                reranked_metric = f"OntSim@{k} (Re-Ranked)"
                
                bar_width = 0.35
                # Apply model-based colors with different alpha for Dense vs Reranked
                dense_bars = ax.bar(x - bar_width/2, sorted_df[dense_metric], 
                                    width=bar_width, label="Dense", 
                                    color=[color_map[m] for m in sorted_df["Model"]], alpha=0.9)
                reranked_bars = ax.bar(x + bar_width/2, sorted_df[reranked_metric], 
                                       width=bar_width, label="Re-Ranked", 
                                       color=[color_map[m] for m in sorted_df["Model"]], alpha=0.6)
                
                # Add value labels (only for a reasonable number of models)
                if len(sorted_df) <= 8:
                    for j, bar in enumerate(dense_bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f"{height:.3f}", ha="center", va="bottom", fontsize=9)
                    
                    for j, bar in enumerate(reranked_bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f"{height:.3f}", ha="center", va="bottom", fontsize=9)
                
                if i == 0:  # Only add legend to first plot
                    # Create legend for dense vs reranked types
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc='upper right')
                    
                    # Only add model color legend to the first plot
                    if len(model_names) <= 8:  # Only if we have a reasonable number of models
                        model_patches = [plt.Rectangle((0,0),1,1, color=color_map[model]) 
                                         for model in model_names]
                        ax2 = ax.twinx()  # Create a second y-axis
                        ax2.set_yticks([])
                        ax2.legend(model_patches, model_names, loc='upper left', 
                                   bbox_to_anchor=(1.05, 1), title="Models")
                
            # Case 2: Single metric type
            else:
                metric = metrics[0]
                # Use model-based colors
                bars = ax.bar(x, sorted_df[metric], color=[color_map[m] for m in sorted_df["Model"]])
                
                # Add value labels (only for a reasonable number of models)
                if len(sorted_df) <= 8:
                    for j, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f"{height:.3f}", ha="center", va="bottom", fontsize=9)
                
                # Add model color legend to the first plot
                if i == 0 and len(model_names) <= 8:
                    model_patches = [plt.Rectangle((0,0),1,1, color=color_map[model]) 
                                     for model in model_names]
                    ax.legend(model_patches, model_names, loc='upper right',
                              title="Models")
            
            # Common formatting for this subplot
            ax.set_ylim(0, 1.0)
            ax.set_title(f"Ontology Similarity @ {k} Comparison", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_df["Model"], rotation=45, ha="right")
            ax.set_ylabel(f"OntSim@{k}")
            
            # Only show x-label on bottom plot
            if i == n_ont_plots-1:
                ax.set_xlabel("Model")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_ontsim_comparison.png"), dpi=300)
        plt.close()
        print("Generated combined OntSim comparison visualization")
        return True
    return False
