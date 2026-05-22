import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_pruning_results(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_distribution(data):
    # Keys are layer IDs as strings
    valid_keys = [k for k in data.keys() if k.isdigit()]
    sorted_keys = sorted(valid_keys, key=int)
    
    layers = []
    expert_counts = []
    
    for k in sorted_keys:
        layers.append(int(k))
        expert_counts.append(len(data[k].get("experts", [])))
        
    return layers, expert_counts

def main():
    parser = argparse.ArgumentParser(description="Plot expert distribution across layers from multiple pruning results.")
    parser.add_argument("--files", nargs="+", required=True, help="Path to one or more pruning_results.json files.")
    parser.add_argument("--labels", nargs="+", help="Labels for the legend (optional).")
    parser.add_argument("--output", type=str, default="/home/lab1008/data_disk_sdc/ywf/data/Pic/expert_distribution.png", help="Output image file path.")
    parser.add_argument("--title", type=str, default="Expert Distribution per Layer", help="Plot title.")
    
    args = parser.parse_args()
    
    plt.figure(figsize=(12, 6))
    
    for i, file_path in enumerate(args.files):
        data = load_pruning_results(file_path)
        if data is None:
            continue
            
        layers, counts = extract_distribution(data)
        
        label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(file_path)
        
        plt.plot(layers, counts, marker='o', linestyle='-', label=label)
        
    plt.xlabel("Layer ID")
    plt.ylabel("Number of Retained Experts")
    plt.title(args.title)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # Ensure x-axis shows all layers if not too many
    if len(layers) <= 64:
        plt.xticks(np.arange(min(layers), max(layers) + 1, step=max(1, len(layers)//20)))
    
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")
    plt.show()

if __name__ == "__main__":
    main()
