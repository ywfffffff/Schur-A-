# MoE Expert Pruning Toolkit

A comprehensive toolkit for pruning Mixture-of-Experts (MoE) Large Language Models, specifically optimized for Qwen3-30B-A3B and ERNIE-4.5-21B-A3B. This project implements a state-of-the-art pruning pipeline that combines dynamic expert count selection with a strictly optimal A* search algorithm for expert subset selection.

## 🌟 Key Features

-   **Strict Optimal Selection**: Uses a Best-First Branch & Bound (A* Search) algorithm with Schur-complement upper bounds to find the mathematically optimal expert subset.
-   **Dynamic Expert Count**: Automatically determines the ideal number of experts ($r$) per layer using Orthogonal Matching Pursuit (OMP) and Knee-point detection.
-   **Compensation Weights**: Calculates optimal scaling weights for retained experts to minimize reconstruction error.
-   **Seamless Patching**: Dynamically patches existing model architectures (Qwen3-30B-A3B, ERNIE-4.5-21B-A3B, etc.) to evaluate pruned models without full re-training.

## 🚀 Workflow Overview

The pruning process follows these primary stages:

1.  **Data Preparation**: Sampling calibration data (e.g., C4) to gather activation statistics.
2.  **Statistics Collection**: Running `Calibrate.py` to collect $y^2, u, G$ statistics using monkeypatching for maximum accuracy.
3.  **Expert Count Determination**: Using OMP to build a reconstruction curve and finding the "knee" to set the target $r$ for each layer.
4.  **Optimal Pruning**: Running the A* search to select the best $r$ experts and compute their compensation weights.
5.  **Evaluation**: Patching the model and measuring Perplexity (PPL) to verify performance.

## 📂 Project Structure

```text
.
├── algorithm/
│   ├── prepare_c4.py      # Calibration data sampling
│   ├── Calibrate.py       # Statistics collection (y2, u, G)
│   ├── pick_r.py          # OMP + Knee-point for expert count selection
│   ├── prune_astar.py     # Core A* search pruning algorithm
│   ├── select_forward.py  # Forward pass utility
│   ├── test.py            # Local testing script
│   └── ...
└── evaluate/
    ├── cal_ppl.py         # Perplexity evaluation with model patching
    ├── deploy.py          # Pruned model deployment logic
    ├── eval_ERNIE.py      # ERNIE-specific evaluation
    ├── eval_greedy.py     # Baseline greedy pruning comparison
    └── ...
```

## 🛠️ Usage Instructions

### 1. Data Preparation
Prepare the calibration dataset (defaults to C4 English).
```bash
python algorithm/prepare_c4.py
```

### 2. Statistics Collection
Collect the necessary statistics ($y^2, u, G$) for the pruning algorithm. This script supports both Qwen and ERNIE architectures.
```bash
python algorithm/Calibrate.py
```

### 3. Determine Expert Count per Layer
Analyze each layer's expert contributions and determine the optimal number of experts to keep ($r$).
```bash
python algorithm/pick_r.py --stats_dir <your_stats_path> --out_json r_per_layer.json
```

### 4. Optimal A* Search Pruning
Execute the strictly optimal pruning algorithm. This step generates the final expert mapping and compensation weights.
```bash
python algorithm/prune_astar.py --data_dir <your_stats_path> --output_file results/ --num_workers 28
```

### 5. Perplexity Evaluation
Apply the pruning results to the model and evaluate performance via PPL.
```bash
python evaluate/cal_ppl.py
```

## 🔬 Algorithm Highlights: A* Pruning

The core pruner (`prune_astar.py`) solves the optimization problem:

```math
\max_{S} F(S) = u_S^T G_S^{-1} u_S, \quad \text{s.t. } |S| = r
```

It uses an admissible **Schur-complement bound**:

```math
h(S, t) = \frac{\text{TopSum}_t(\tilde{u}_R^2)}{\lambda_{\min}(\Sigma_R) + \epsilon}
```

where $\Sigma_R$ is the Schur complement of the remaining experts. This ensures the search finds the global optimum while efficiently pruning the search space.

