"""
Run benchmarks on all available models and compare their performance.
"""
import os
import subprocess
import pandas as pd
import time
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Define all models to benchmark
MODELS = [
    "FremyCompany/BioLORD-2023-M",
    "jinaai/jina-embeddings-v2-base-de",
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2"
]

def run_benchmark(model_name, similarity_threshold=0.1):
    """Run benchmark for a single model and extract results from stdout"""
    start_time = time.time()
    
    # Run the benchmark script
    cmd = ["python", "benchmark_rag.py", "--model-name", model_name, "--similarity-threshold", str(similarity_threshold)]
    logging.info(f"Running benchmark for {model_name}")
    logging.info(f"Command: {' '.join(cmd)}")
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logging.error(f"Benchmark failed for model {model_name}")
        logging.error(process.stderr)
        return None
    
    end_time = time.time()
    logging.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")
    
    # Extract results directly from stdout
    return extract_results_from_stdout(process.stdout, model_name)

def extract_results_from_stdout(stdout, model_name):
    """Extract benchmark results from the stdout of the benchmark script"""
    lines = stdout.split('\n')
    
    # Find the line with the model comparison table
    start_idx = None
    for i, line in enumerate(lines):
        if "Model Comparison:" in line:
            start_idx = i + 1
            break
    
    if start_idx is None:
        logging.error(f"Could not find model comparison table in output for {model_name}")
        return None
    
    # Extract the metrics from the model comparison lines
    metrics = {}
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith('==='):
            # The line might contain the model name and metrics
            parts = line.split()
            if len(parts) >= 8:  # We expect at least MRR, Precision, Recall, F1, HR@1, HR@3, HR@5, HR@10
                try:
                    # Extract metrics - the format is specific to the benchmark output
                    model_slug = parts[0]
                    metrics = {
                        "Model": model_name.split("/")[-1],  # Short name
                        "Original Model Name": model_name,
                        "Model Slug": model_slug,
                        "MRR": float(parts[1]),
                        "Precision": float(parts[2]),
                        "Recall": float(parts[3]),
                        "F1": float(parts[4]),
                        "Hit@1": float(parts[5]),
                        "Hit@3": float(parts[6]),
                        "Hit@5": float(parts[7]),
                        "Hit@10": float(parts[8])
                    }
                    return pd.DataFrame([metrics])
                except (ValueError, IndexError) as e:
                    logging.error(f"Error parsing metrics from line '{line}': {e}")
    
    logging.error(f"Could not extract metrics from output for {model_name}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on all models")
    parser.add_argument("--similarity-threshold", type=float, default=0.1, 
                        help="Similarity threshold for filtering results (default: 0.1)")
    parser.add_argument("--setup", action="store_true", 
                        help="Setup HPO index for models that don't have one")
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Setup indexes if requested
    if args.setup:
        for model in MODELS:
            logging.info(f"Setting up HPO index for {model}")
            setup_cmd = ["python", "setup_hpo_index.py", "--model-name", model]
            try:
                subprocess.run(setup_cmd, check=True)
                logging.info(f"Successfully set up index for {model}")
            except subprocess.CalledProcessError:
                logging.error(f"Failed to set up index for {model}")
    
    # Run benchmarks and collect results
    all_results = []
    for model in MODELS:
        df = run_benchmark(model, args.similarity_threshold)
        if df is not None:
            all_results.append(df)
            
            # Save individual results with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = model.split("/")[-1].replace("-", "_")
            save_path = f"benchmark_results/{model_short}_{timestamp}.csv"
            df.to_csv(save_path, index=False)
            logging.info(f"Saved individual results to {save_path}")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results)
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = f"benchmark_results/combined_results_{timestamp}.csv"
        combined_df.to_csv(combined_path, index=False)
        logging.info(f"All benchmark results saved to {combined_path}")
        
        # No need to aggregate since we already have one row per model
        summary = combined_df[[
            "Model", 
            "MRR", 
            "Precision", 
            "Recall", 
            "F1", 
            "Hit@1", 
            "Hit@3", 
            "Hit@5", 
            "Hit@10"
        ]]
        
        print("\n===== Combined Benchmark Results =====")
        print(f"Models evaluated: {len(all_results)}")
        print(f"Similarity threshold: {args.similarity_threshold}")
        print("\nModel Performance Comparison:")
        print(summary.to_string(index=False, float_format="%.4f"))
    else:
        logging.error("No benchmark results to combine")

if __name__ == "__main__":
    main()
