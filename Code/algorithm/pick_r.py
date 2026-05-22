# knee_omp_warmstart_preselect_plot.py
# ------------------------------------------------------------
# Warm-start OMP + Knee-point:
#   1) 每层先按能量/收益分数预选 base_frac(=0.25) 个 expert 组成 S0 (size=32 for E=128)
#   2) 从 S0 出发，对剩余 expert 继续做 Gram-OMP，直到选满 E
#   3) 在 k ∈ [|S0|, E] 上做 knee-point（端点连线最大距离）
#   4) 输出每层 r_l (>=|S0|)；可保存曲线；可画所有层 L(k)/y2 论文图
# ------------------------------------------------------------

import argparse
import json
import os
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


# -------------------- Triangular solvers (no scipy) --------------------
def solve_lower(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    t = L.shape[0]
    x = np.empty_like(b, dtype=np.float64)
    for i in range(t):
        s = b[i] - np.dot(L[i, :i], x[:i])
        x[i] = s / L[i, i]
    return x


def solve_upper(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    t = U.shape[0]
    x = np.empty_like(b, dtype=np.float64)
    for i in range(t - 1, -1, -1):
        s = b[i] - np.dot(U[i, i + 1 :], x[i + 1 :])
        x[i] = s / U[i, i]
    return x


def cho_solve_from_L(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = solve_lower(L, b)
    x = solve_upper(L.T, y)
    return x


# -------------------- Smoothing + Knee --------------------
def smooth_ma(x: np.ndarray, w: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if w <= 1:
        return x
    w = min(w, x.size)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(xp, kernel, mode="valid")


def knee_point_on_segment(L_segment: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    在一个段上做“端点连线最大距离”：
      L_segment: shape [K], K>=1
    返回：
      r_seg: 1..K （段内的拐点位置）
      d: 每个点到端点连线距离
    """
    L = np.asarray(L_segment, dtype=np.float64)
    K = L.size
    if K < 3:
        return K, np.zeros_like(L)

    x = np.linspace(0.0, 1.0, K)
    y0, y1 = L[0], L[-1]
    denom = (y1 - y0)
    if abs(denom) < 1e-12:
        return K, np.zeros_like(L)

    y = (L - y0) / denom
    d = np.abs(y - x) / np.sqrt(2.0)
    r = int(np.argmax(d) + 1)
    return r, d


# -------------------- Base set preselection --------------------
def preselect_base_set(
    u_L: np.ndarray,
    G_L: np.ndarray,
    support_L: np.ndarray,
    frac: float,
    metric: str,
    ridge: float,
    use_supported_only: bool,
) -> np.ndarray:
    """
    返回 base_ids: np.ndarray[int64]，作为 warm-start 集合 S0
    metric:
      - "gain1": u^2/(Gii+ridge)（推荐）
      - "gii":   Gii
      - "abs_u": |u|
    """
    u_L = u_L.astype(np.float64)
    diag = np.diag(G_L).astype(np.float64)
    E = u_L.size

    base_mask = np.ones(E, dtype=bool)
    if use_supported_only:
        base_mask &= (support_L > 0)

    if metric == "gain1":
        score = (u_L * u_L) / (diag + ridge)
    elif metric == "gii":
        score = diag
    elif metric == "abs_u":
        score = np.abs(u_L)
    else:
        raise ValueError(f"Unknown base_metric={metric}")

    score = score.copy()
    score[~base_mask] = -np.inf

    M = int(np.ceil(frac * E))
    M = max(1, min(M, int(np.sum(base_mask))))
    idx = np.argsort(-score)[:M].astype(np.int64)
    return idx


def build_cholesky_for_set(G_L: np.ndarray, S: np.ndarray, ridge: float) -> np.ndarray:
    """
    返回 Lchol，使得 Lchol Lchol^T = G_S + ridge I
    """
    S = np.asarray(S, dtype=np.int64)
    GS = G_L[np.ix_(S, S)].astype(np.float64, copy=True)
    GS.flat[:: GS.shape[0] + 1] += ridge
    # cholesky
    return np.linalg.cholesky(GS)


# -------------------- Warm-start Gram-OMP --------------------
def warmstart_omp_curve(
    y2_L: float,
    u_L: np.ndarray,
    G_L: np.ndarray,
    support_L: np.ndarray,
    base_ids: np.ndarray,
    ridge: float,
    use_supported_only: bool,
) -> Tuple[List[int], np.ndarray, int]:
    """
    从 base_ids 作为初始集合 S0 出发，继续 Gram-OMP 选剩余 experts 直到满 E。
    输出：
      selected_order: 最终选中顺序（全局 expert id），长度=E（base_ids在前，后续按OMP追加）
      Lfull: shape [E]，其中
            - k<|S0| 的位置填 NaN
            - Lfull[|S0|-1] = g(S0)
            - Lfull[k-1] = g(S_k) for k>=|S0|
      base_size: |S0|
    """
    u_L = np.asarray(u_L, dtype=np.float64)
    G_L = np.asarray(G_L, dtype=np.float64)
    E = u_L.size

    base_ids = np.unique(np.asarray(base_ids, dtype=np.int64))
    base_ids = base_ids[(base_ids >= 0) & (base_ids < E)]
    base_size = int(base_ids.size)
    assert base_size >= 1, "base set must be non-empty"

    # remaining pool
    base_set = set(map(int, base_ids.tolist()))
    if use_supported_only:
        valid = set(map(int, np.flatnonzero(support_L > 0).tolist()))
    else:
        valid = set(range(E))
    remaining = sorted(list(valid - base_set))

    # initial chol + beta
    Lchol = build_cholesky_for_set(G_L, base_ids, ridge)
    uS = u_L[base_ids]  # [base_size]
    beta = cho_solve_from_L(Lchol, uS)

    # Lfull (NaN for k<base_size)
    Lfull = np.full((E,), np.nan, dtype=np.float64)
    g0 = float(y2_L - np.dot(uS, beta))
    Lfull[base_size - 1] = g0

    selected_order: List[int] = base_ids.tolist()

    eps_denom = 1e-12

    # incremental selection to reach E
    while len(selected_order) < E and remaining:
        S_idx = np.array(selected_order, dtype=np.int64)
        # refresh beta for current S
        uS = u_L[S_idx]
        beta = cho_solve_from_L(Lchol, uS)

        best_gain = -1.0
        best_e = None
        best_w = None
        best_den = None

        for e in remaining:
            a = G_L[S_idx, e]  # shape (t,)
            w = solve_lower(Lchol, a)
            den = float(G_L[e, e] + ridge - np.dot(w, w))
            if den <= eps_denom:
                continue
            num = float(u_L[e] - np.dot(a, beta))
            gain = (num * num) / den
            if gain > best_gain:
                best_gain = gain
                best_e = e
                best_w = w
                best_den = den

        if best_e is None:
            # fallback: pick first
            best_e = int(remaining[0])
            a = G_L[S_idx, best_e]
            best_w = solve_lower(Lchol, a)
            best_den = float(G_L[best_e, best_e] + ridge - np.dot(best_w, best_w))
            if best_den <= 0:
                best_den = eps_denom

        # update Cholesky
        new_diag = np.sqrt(max(best_den, eps_denom))
        tsize = Lchol.shape[0]
        Lnew = np.zeros((tsize + 1, tsize + 1), dtype=np.float64)
        Lnew[:tsize, :tsize] = Lchol
        Lnew[tsize, :tsize] = best_w
        Lnew[tsize, tsize] = new_diag
        Lchol = Lnew

        # add expert
        selected_order.append(int(best_e))
        remaining.remove(int(best_e))

        # compute g at new k
        k = len(selected_order)
        S_idx2 = np.array(selected_order, dtype=np.int64)
        uS2 = u_L[S_idx2]
        beta2 = cho_solve_from_L(Lchol, uS2)
        gk = float(y2_L - np.dot(uS2, beta2))
        Lfull[k - 1] = gk

    return selected_order, Lfull, base_size


# -------------------- Plotting (NaN-aware) --------------------
def plot_all_layers_L_over_y2(
    Lnorm_full: List[np.ndarray],   # each shape [E], NaN allowed for k<base
    layer_indices: List[int],       # layer ID for each curve
    out_path: str,
    quantile_band: bool,
    alpha: float,
    q_low: float = 0.10,
    q_high: float = 0.90,
):
    import matplotlib.pyplot as plt

    n = len(Lnorm_full)
    if n == 0:
        raise ValueError("No curves to plot.")
    E = Lnorm_full[0].size
    x = np.arange(1, E + 1)

    M = np.stack(Lnorm_full, axis=0)  # [n,E], may contain NaN

    plt.figure(figsize=(10, 6))  # Slightly larger for legend
    ax = plt.gca()

    # each layer curve
    # Use a colormap to distinguish layers
    cm = plt.get_cmap('tab20')
    
    for i in range(n):
        yi = M[i]
        valid = np.isfinite(yi)
        # Use higher alpha for visibility since we want to identify them
        # If alpha is very low (default 0.12), it might be hard to see colors.
        # We'll use max(alpha, 0.5) for labeled lines or just use alpha if user specified.
        # Assuming user wants to see them clearly now.
        color = cm(i % 20)
        ax.plot(x[valid], yi[valid], linewidth=1.5, alpha=max(alpha, 0.7), 
                label=f"Layer {layer_indices[i]}", color=color)

    # mean curve REMOVED as requested
    # mean_curve = np.nanmean(M, axis=0)
    # valid_m = np.isfinite(mean_curve)
    # ax.plot(x[valid_m], mean_curve[valid_m], linewidth=2.5, label="Mean across layers", color='black')

    # quantile band
    if quantile_band:
        ql = np.nanquantile(M, q_low, axis=0)
        qh = np.nanquantile(M, q_high, axis=0)
        valid_q = np.isfinite(ql) & np.isfinite(qh)
        ax.fill_between(
            x[valid_q], ql[valid_q], qh[valid_q],
            alpha=0.18, label=f"{int(q_low*100)}%-{int(q_high*100)}% band"
        )

    ax.set_xlabel("k (number of kept experts)")
    ax.set_ylabel("L(k) / y2")
    ax.set_title("MoE Reconstruction Residual Curve across Layers (warm-start at k=base)")
    ax.set_xlim(min(batch_sizes), E)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------- IO + Driver --------------------
def load_stats(stats_dir: str):
    y2 = np.load(os.path.join(stats_dir, "y2.npy"))
    u = np.load(os.path.join(stats_dir, "u.npy"))
    G = np.load(os.path.join(stats_dir, "G.npy"))
    support = np.load(os.path.join(stats_dir, "support.npy"))
    meta_path = os.path.join(stats_dir, "meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return y2, u, G, support, meta


def parse_layers(s: str, num_layers: int) -> List[int]:
    if s.strip().lower() == "all":
        return list(range(num_layers))
    items = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            items.extend(list(range(a, b + 1)))
        else:
            items.append(int(part))
    items = [i for i in items if 0 <= i < num_layers]
    return sorted(set(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_dir", type=str, default="/home/lab1008/data_disk_sdc/ywf/data/stats_4_ERNIE_1")
    ap.add_argument("--layers", type=str, default="all")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--smooth_w", type=int, default=5)
    ap.add_argument("--use_supported_only", action="store_true")

    # warm-start base set
    ap.add_argument("--base_frac", type=float, default=0.25, help="warm-start 集合比例（128->32）")
    ap.add_argument("--base_metric", type=str, default="gain1", choices=["gain1", "gii", "abs_u"])

    # outputs
    ap.add_argument("--out_json", type=str, default="/home/lab1008/data_disk_sdc/ywf/data/stats_4_ERNIE_1/r_per_layer.json")
    ap.add_argument("--save_curves_dir", type=str, default="/home/lab1008/data_disk_sdc/ywf/data/curves_3", help="保存每层 npz（含base_ids/selected/Lfull）")

    # plot
    ap.add_argument("--plot_path", type=str, default="/home/lab1008/data_disk_sdc/ywf/data/stats_4_ERNIE_1/L_over_y2.png")
    ap.add_argument("--plot_alpha", type=float, default=0.12)
    ap.add_argument("--plot_quantile_band", action="store_true")
    ap.add_argument("--plot_only_layers", type=str, default="1,7,14,21,27", help='只画指定层，如 "0,12,24,36,47"，空则画所有层')
    args = ap.parse_args()

    y2, u, G, support, meta = load_stats(args.stats_dir)
    num_layers = y2.shape[0]
    E = u.shape[1]

    layers = parse_layers(args.layers, num_layers)
    plot_only = set(parse_layers(args.plot_only_layers, num_layers)) if args.plot_only_layers.strip() else None

    if args.save_curves_dir:
        os.makedirs(args.save_curves_dir, exist_ok=True)

    out: Dict[str, Any] = {
        "stats_dir": args.stats_dir,
        "num_layers": int(num_layers),
        "num_experts": int(E),
        "ridge": float(args.ridge),
        "smooth_w": int(args.smooth_w),
        "use_supported_only": bool(args.use_supported_only),
        "base_frac": float(args.base_frac),
        "base_metric": args.base_metric,
        "r_per_layer": {},
    }

    # for plot
    Lnorm_full_list: List[np.ndarray] = []
    plot_layer_indices: List[int] = []

    print("=" * 90)
    print("Warm-start OMP + Knee-point")
    print(f"E={E} | base_frac={args.base_frac} | base_metric={args.base_metric} | ridge={args.ridge} | smooth_w={args.smooth_w}")
    if meta:
        print(f"meta: {meta}")
    print("=" * 90)

    for L in layers:
        y2_L = float(y2[L])
        u_L = u[L].astype(np.float64)
        G_L = G[L].astype(np.float64)
        sup_L = support[L].astype(np.int64)

        base_ids = preselect_base_set(
            u_L=u_L, G_L=G_L, support_L=sup_L,
            frac=args.base_frac, metric=args.base_metric,
            ridge=args.ridge, use_supported_only=args.use_supported_only
        )

        selected_order, Lfull, base_size = warmstart_omp_curve(
            y2_L=y2_L, u_L=u_L, G_L=G_L, support_L=sup_L,
            base_ids=base_ids, ridge=args.ridge, use_supported_only=args.use_supported_only
        )

        # knee only on segment k in [base_size..E] (1-indexed)
        seg = Lfull[base_size - 1 :].copy()
        # smooth segment (NaN-free by construction)
        seg_s = smooth_ma(seg, w=args.smooth_w)
        r_seg, dist = knee_point_on_segment(seg_s)
        r_global = base_size + r_seg - 1  # convert to k in [base_size..E]

        out["r_per_layer"][str(L)] = {
            "r": int(r_global),
            "y2": float(y2_L),
            "base_size": int(base_size),
            "base_ids": base_ids.tolist(),
            "selected_order": selected_order,   # length up to E
            "g_at_base": float(Lfull[base_size - 1]),
            "g_over_y2_at_base": float(Lfull[base_size - 1] / y2_L),
            "g_at_r": float(Lfull[r_global - 1]),
            "g_over_y2_at_r": float(Lfull[r_global - 1] / y2_L),
            "g_at_E": float(Lfull[E - 1]),
            "g_over_y2_at_E": float(Lfull[E - 1] / y2_L),
        }

        print(f"[Layer {L:02d}] base={base_size:3d}  r={r_global:3d}  "
              f"g(base)/y2={Lfull[base_size-1]/y2_L:.4e}  g(r)/y2={Lfull[r_global-1]/y2_L:.4e}  g(E)/y2={Lfull[E-1]/y2_L:.4e}")

        if args.save_curves_dir:
            np.savez(
                os.path.join(args.save_curves_dir, f"layer_{L:02d}_warmstart_curves.npz"),
                base_ids=base_ids.astype(np.int64),
                selected_order=np.array(selected_order, dtype=np.int64),
                Lfull=Lfull.astype(np.float64),
                base_size=np.array([base_size], dtype=np.int64),
                seg=seg.astype(np.float64),
                seg_s=seg_s.astype(np.float64),
                dist=dist.astype(np.float64),
                r_global=np.array([r_global], dtype=np.int64),
            )

        # collect for plot
        if args.plot_path:
            if (plot_only is None) or (L in plot_only):
                Lnorm_full_list.append((Lfull / y2_L).astype(np.float64))
                plot_layer_indices.append(L)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("=" * 90)
    print(f"Saved r_per_layer to: {args.out_json}")
    if args.save_curves_dir:
        print(f"Saved curves to: {args.save_curves_dir}")
    print("=" * 90)

    if args.plot_path:
        if len(Lnorm_full_list) == 0:
            print("No curves collected for plotting (check --plot_only_layers).")
        else:
            plot_all_layers_L_over_y2(
                Lnorm_full=Lnorm_full_list,
                layer_indices=plot_layer_indices,
                out_path=args.plot_path,
                quantile_band=args.plot_quantile_band,
                alpha=args.plot_alpha,
            )
            print(f"Saved plot to: {args.plot_path} (layers_plotted={len(Lnorm_full_list)}, E={E})")


if __name__ == "__main__":
    main()
